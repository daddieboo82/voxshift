"""Private speech-to-speech inference API for VoxShift."""
import hmac
import re
import io
import os
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4
import wave

from anyio import CapacityLimiter, WouldBlock, create_task_group, sleep, to_thread
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

MAX_FILE = 18 * 1024 * 1024
MAX_REQUEST = 2 * MAX_FILE + 65536
MAX_JOB_AUDIO = 3_000_000
MAX_JOBS = 12
MAX_PENDING = 3
JOB_RETENTION = 600


class ChatterboxEngine:
    def __init__(self):
        from chatterbox.vc import ChatterboxVC
        self.device = os.getenv("VOXSHIFT_DEVICE", "cpu")
        self.model = ChatterboxVC.from_pretrained(device=self.device)
        self.tts = None

    def synthesize(self, text, reference):
        import torchaudio
        from chatterbox.tts import ChatterboxTTS

        if self.tts is None:
            self.tts = ChatterboxTTS.from_pretrained(device=self.device)
        with TemporaryDirectory(prefix="voxshift-") as directory:
            reference_path = Path(directory) / "reference.wav"
            output_path = Path(directory) / "speech.wav"
            reference_path.write_bytes(reference)
            try:
                samples = self.tts.generate(text, audio_prompt_path=str(reference_path))
                torchaudio.save(str(output_path), samples, self.tts.sr, encoding="PCM_S", bits_per_sample=16)
                return output_path.read_bytes()
            finally:
                self.tts.conds = None

    def convert(self, source, reference):
        import torchaudio
        with TemporaryDirectory(prefix="voxshift-") as directory:
            source_path = Path(directory) / "source.wav"
            reference_path = Path(directory) / "reference.wav"
            output_path = Path(directory) / "converted.wav"
            source_path.write_bytes(source)
            reference_path.write_bytes(reference)
            try:
                samples = self.model.generate(str(source_path), target_voice_path=str(reference_path))
                torchaudio.save(str(output_path), samples, self.model.sr, encoding="PCM_S", bits_per_sample=16)
                return output_path.read_bytes()
            finally:
                self.model.ref_dict = None

    def convert_singing(self, source, reference):
        from pitch_lock import match_source_pitch
        return match_source_pitch(source, self.convert(source, reference))

    def enhance(self, source, _unused_reference):
        from vocal_enhance import enhance_vocal
        return enhance_vocal(source)


def validate_pcm(data, field, minimum, maximum):
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            channels = audio.getnchannels()
            rate = audio.getframerate()
            width = audio.getsampwidth()
            frames = audio.getnframes()
            if channels not in (1, 2) or not 8000 <= rate <= 48000 or width != 2:
                raise ValueError
            if not minimum <= frames / rate <= maximum:
                raise HTTPException(422, f"{field} duration must be {minimum}–{maximum} seconds")
            if len(audio.readframes(frames)) != frames * channels * width:
                raise ValueError
    except (ValueError, wave.Error, EOFError) as exc:
        raise HTTPException(422, f"{field} must be a complete 16-bit PCM WAV, mono or stereo") from exc



def valid_output_wav(data):
    """Only deliver complete PCM WAV output; reject invalid model bytes."""
    if not isinstance(data, bytes) or not 44 <= len(data) <= MAX_JOB_AUDIO:
        return False
    try:
        with wave.open(io.BytesIO(data), "rb") as audio:
            channels = audio.getnchannels()
            width = audio.getsampwidth()
            rate = audio.getframerate()
            frames = audio.getnframes()
            if channels not in (1, 2) or width != 2 or not 8000 <= rate <= 48000 or frames < 1:
                return False
            expected = frames * channels * width
            return expected <= len(data) and len(audio.readframes(frames)) == expected
    except (ValueError, wave.Error, EOFError):
        return False


async def read_upload(form, field, minimum, maximum):
    values = form.getlist(field)
    if len(values) != 1 or not isinstance(values[0], UploadFile):
        raise HTTPException(422, f"Provide one {field} WAV file")
    data = await values[0].read(MAX_FILE + 1)
    if len(data) > MAX_FILE:
        raise HTTPException(413, f"{field} is too large")
    validate_pcm(data, field, minimum, maximum)
    return data


def create_app(engine_factory=ChatterboxEngine):
    @asynccontextmanager
    async def lifespan(app):
        key = os.environ.get("VOXSHIFT_API_KEY", "")
        if len(key) < 32 or not key.isascii():
            raise RuntimeError("Set VOXSHIFT_API_KEY to a random ASCII secret of at least 32 characters")
        app.state.key = ("Bearer " + key).encode()
        app.state.engine = await to_thread.run_sync(engine_factory)
        app.state.slot = CapacityLimiter(1)
        app.state.jobs = {}
        async with create_task_group() as tasks:
            app.state.tasks = tasks
            try:
                yield
            finally:
                tasks.cancel_scope.cancel()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.get("/healthz")
    async def health():
        return {"status": "ready"}

    @app.post("/v1/convert")
    async def convert(request: Request):
        return await generate(request, speech=False)

    @app.post("/v1/speech")
    async def speech(request: Request):
        return await generate(request, speech=True)

    @app.post("/v1/enhance")
    async def enhance(request: Request):
        return await generate(request, speech=False, enhance=True)

    def authorize(request: Request):
        authorization = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(authorization, app.state.key):
            raise HTTPException(401, "Invalid provider credentials")

    @app.get("/v1/capabilities")
    async def capabilities(request: Request):
        authorize(request)
        return JSONResponse({"modes": ["speech", "convert", "sing", "enhance"]}, headers={"Cache-Control": "no-store"})

    def read_owner(request: Request):
        owner = request.headers.get("x-voxshift-owner", "")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", owner):
            raise HTTPException(400, "Invalid job owner")
        return owner

    def wav_response(wav):
        return Response(wav, media_type="audio/wav", headers={
            "Cache-Control": "no-store",
            "Content-Disposition": 'attachment; filename="voxshift-converted.wav"',
        })

    async def read_inputs(request: Request, speech, enhance=False):
        size = request.headers.get("content-length", "")
        if not size.isdecimal():
            raise HTTPException(411, "Content-Length required")
        if len(size) > 8 or int(size) > MAX_REQUEST:
            raise HTTPException(413, "Audio request is too large")
        if not request.headers.get("content-type", "").lower().startswith("multipart/form-data;"):
            raise HTTPException(415, "Expected multipart/form-data")
        async with request.form(max_files=2, max_fields=2, max_part_size=4096) as form:
            if form.getlist("consent") != ["confirmed"]:
                raise HTTPException(422, "Confirm permission to use the reference voice")
            if enhance:
                source = await read_upload(form, "source", 1, 60)
                return app.state.engine.enhance, source, b""
            reference = await read_upload(form, "reference", 3, 60)
            if speech:
                text = str(form.get("text", "")).strip()
                if not 1 <= len(text) <= 400:
                    raise HTTPException(422, "Text must contain 1–400 characters")
                return app.state.engine.synthesize, text, reference
            return app.state.engine.convert, await read_upload(form, "source", 1, 60), reference

    async def generate(request: Request, speech, enhance=False):
        authorize(request)
        try:
            app.state.slot.acquire_nowait()
        except WouldBlock as exc:
            raise HTTPException(503, "Voice engine is busy", headers={"Retry-After": "5"}) from exc
        try:
            operation, argument, reference = await read_inputs(request, speech, enhance=enhance)
            try:
                wav = await to_thread.run_sync(operation, argument, reference)
            except Exception as exc:
                raise HTTPException(502, "Voice conversion failed") from exc
            if not valid_output_wav(wav):
                raise HTTPException(502, "Voice provider returned invalid WAV audio")
            return wav_response(wav)
        finally:
            app.state.slot.release()

    async def run_job(identifier, operation, argument, reference):
        job = app.state.jobs[identifier]
        try:
            async with app.state.slot:
                job["status"] = "running"
                wav = await to_thread.run_sync(operation, argument, reference)
            if not valid_output_wav(wav):
                raise ValueError("Invalid WAV result")
            job.update(status="complete", wav=wav)
        except Exception:
            job["status"] = "failed"
        finally:
            # Do not retain raw inputs on the completed job.
            del argument, reference
        await sleep(JOB_RETENTION)
        app.state.jobs.pop(identifier, None)

    @app.post("/v1/jobs/{mode}", status_code=202)
    async def start_job(mode: str, request: Request):
        authorize(request)
        owner = read_owner(request)
        if mode not in ("speech", "convert", "sing", "enhance"):
            raise HTTPException(404, "Unknown generation mode")
        jobs = app.state.jobs
        pending = [job for job in jobs.values() if job["status"] in ("queued", "running", "receiving")]
        if len(jobs) >= MAX_JOBS or len(pending) >= MAX_PENDING or any(job["owner"] == owner for job in pending):
            raise HTTPException(503, "Voice engine is busy", headers={"Retry-After": "5"})
        identifier = uuid4().hex
        jobs[identifier] = {"owner": owner, "status": "receiving"}
        try:
            operation, argument, reference = await read_inputs(request, speech=mode == "speech", enhance=mode == "enhance")
            if mode == "sing":
                operation = app.state.engine.convert_singing
            if len(reference) > MAX_JOB_AUDIO or isinstance(argument, bytes) and len(argument) > MAX_JOB_AUDIO:
                raise HTTPException(413, "Studio audio is too large")
            jobs[identifier]["status"] = "queued"
            app.state.tasks.start_soon(run_job, identifier, operation, argument, reference)
        except BaseException:
            jobs.pop(identifier, None)
            raise
        return {"jobId": identifier, "status": "queued"}

    @app.get("/v1/jobs/{identifier}")
    async def job_status(identifier: str, request: Request):
        authorize(request)
        owner = read_owner(request)
        job = app.state.jobs.get(identifier)
        if not job or job["owner"] != owner:
            raise HTTPException(404, "Voice job not found or expired")
        if job["status"] == "complete":
            return wav_response(job["wav"])
        return JSONResponse({"status": job["status"]}, headers={"Cache-Control": "no-store"})

    return app


app = create_app()
