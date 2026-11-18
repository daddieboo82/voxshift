FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
ARG VOXSHIFT_TORCH_INDEX=https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir --index-url "$VOXSHIFT_TORCH_INDEX" torch==2.6.0 torchaudio==2.6.0 \
    && pip install --no-cache-dir -r requirements.txt
RUN useradd --create-home voxshift \
    && install -d -o voxshift -g voxshift /home/voxshift/.cache/huggingface \
    && install -o voxshift -g voxshift /dev/null /home/voxshift/.cache/huggingface/.voxshift-cache
USER voxshift
COPY app.py pitch_lock.py ./
ENV VOXSHIFT_DEVICE=cpu HF_HOME=/home/voxshift/.cache/huggingface
EXPOSE 8080
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--no-access-log"]
