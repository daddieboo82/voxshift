# VoxShift AI Provider

Self-hosted speech-to-speech conversion for recorded takes. The Chatterbox VC
model receives a source take and an authorized target voice recording, then
returns generated audio as a WAV. This service uses downloadable model weights
and performs inference on its own host. Each request supplies its own reference.

## Run

Build with `docker build -t voxshift-provider .` from this directory.
Set `VOXSHIFT_API_KEY` to a random secret of 32 or more ASCII characters and
export it to your shell. Start the service with:

```sh
docker run --rm -p 127.0.0.1:8080:8080 -e VOXSHIFT_API_KEY \
  -v voxshift-models:/home/voxshift/.cache/huggingface voxshift-provider
```

The first start downloads model weights. `GET /healthz` returns ready once loaded.
The default image installs CPU wheels. For NVIDIA inference, build with
`--build-arg VOXSHIFT_TORCH_INDEX=https://download.pytorch.org/whl/cu124`
on a compatible host and container runtime, then run with `--gpus all`
and `-e VOXSHIFT_DEVICE=cuda`.
Use one worker per model instance. Inference is offline and serialized.

## Convert a Studio take

Export a dry 16-bit PCM WAV take and provide a clean, authorized voice sample
in the same format. The source must be 1–60 seconds and the reference 3–60
seconds, mono or stereo at 8–48 kHz. Test spoken vocals first. Singing fidelity
and clip alignment require evaluation on the intended tracks.

```sh
curl --fail-with-body -X POST http://127.0.0.1:8080/v1/convert \
  -H "Authorization: Bearer ${VOXSHIFT_API_KEY}" \
  -F 'consent=confirmed' -F 'source=@take.wav;type=audio/wav' \
  -F 'reference=@reference.wav;type=audio/wav' --output converted.wav
```

Import the returned WAV as a new track to keep the source intact.
The signed-in Studio integration sends bounded requests through its backend
over HTTPS with an account-level quota. Keep the provider key on the backend.
Configure the reverse proxy to
cap request bodies at 37 MiB and preserve Content-Length. Audio files exist
only in request-scoped temporary storage and are removed on completion.
A busy provider responds with HTTP 503 and Retry-After.

The bundled API test exercises request validation and audio response behavior
with a stand-in engine. Real model output must be tested on the inference host.

## Studio integration

Set `VOXSHIFT_PROVIDER_URL` to the HTTPS origin of the deployed container and
`VOXSHIFT_API_KEY` to its matching secret in the NaliChat Base44 runtime.
The `voxShiftGenerate` backend proxies authorized Studio requests.
For text-to-speech, the provider exposes `POST /v1/speech` with the same
authorization and reference form fields, replacing `source` with a `text`
field (up to 400 characters). The first speech request loads the TTS model.

## Inference smoke test

Both endpoints produced valid, non-silent mono 24 kHz WAVs using the real
models with synthetic speech fixtures. On the four-thread CPU test, a
5.21-second take returned a 5.22-second conversion in about 34 seconds.
The first text-to-speech request generated 3.62 seconds of audio in about
71 seconds, including lazy model initialization.

Studio starts a background job with `POST /v1/jobs/speech` for text-to-speech,
`POST /v1/jobs/convert` for a spoken take, or `POST /v1/jobs/sing` for
singing, then polls `GET /v1/jobs/{jobId}` until audio
is ready. Both routes require the provider bearer key and a trusted
`X-VoxShift-Owner` header. The Base44 proxy derives the owner from the
signed-in account. The provider runs one generation at a time with up
to three pending jobs. Use a single persistent worker: jobs are held
in memory, completed results expire after ten minutes, and restarts
clear pending jobs and results. Keep this service running behind HTTPS
and avoid request-triggered scale-to-zero for this job design.

Before enabling Voice Match for users, test cold and warm jobs on the
deployment host.

## Singing mode (beta)

Singing mode converts the reference voice, then follows the original lead's
measured pitch contour. It returns a clip with the source duration at the
provider's output sample rate. This mode is for isolated monophonic lead
vocals. It rejects conversion when too few voiced notes can be aligned or
when the converted duration differs by more than 120 ms.

In two five-second synthetic melody checks, raw conversion shifted median
note pitches by up to 300 cents. The pitch-preserving pass reduced median
note errors to 10 cents or less on those fixtures. This does not validate
real singing quality, timbre matching, or phrase-level timing.

Studio checks `GET /v1/capabilities` before enabling generation. The health
check uses the same provider bearer key and returns the available modes.
It sends no reference audio. If the check fails, confirm that the host is
running this version and that the Studio URL and key are configured.
