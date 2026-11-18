# VoxShift HTTPS deployment

This directory is a **deployment template**, not a running service. GitHub stores the source. Supabase Edge Functions cannot run the PyTorch/Chatterbox model within their CPU/memory limits; keep NaliChat on Base44 and the model on a persistent CPU/GPU Docker host.

1. Provision a Linux host with Docker Engine and Compose, a public DNS name, TCP 80/443 open, and enough RAM/storage for the Chatterbox model weights. Point the domain at the host. For production speed, prefer a compatible GPU host.
2. Copy `.env.example` to `.env` in this directory. Replace `VOXSHIFT_DOMAIN` and generate a new random `VOXSHIFT_API_KEY` (32+ ASCII characters). Do not commit .env or reveal the key to the browser.
3. Run from this directory: `docker compose config` to validate, then `docker compose up -d --build`. First startup downloads model weights and may take a while.
4. Verify locally on the host: `docker compose ps`. Verify externally: `curl -i https://YOUR_DOMAIN/healthz`. Check authenticated modes: `curl -i -H "Authorization: Bearer YOUR_KEY" https://YOUR_DOMAIN/v1/capabilities`; never paste your live key into support messages or screenshots.
5. Set **Base44 NaliChat backend secrets** `VOXSHIFT_PROVIDER_URL=https://YOUR_DOMAIN` and `VOXSHIFT_API_KEY` to the exact matching secret. Do not put these in `VITE_*` variables or frontend files.
6. Open Studio → VoxShift Voice Match, confirm modes load, then do a permitted speech reference and source-take test. Confirm a new track appears and the original take remains. Only then expose generation to customers.

The server runs one worker. Job state is held in memory, expires after ten minutes, and disappears on restart; do not scale to multiple instances without persistent shared job storage. The API requires an authenticated Base44 backend proxy and an account owner ID for each job. Caddy handles HTTPS automatically when DNS and inbound ports work.

For GPU hosts, install NVIDIA Container Toolkit and use a GPU-capable Docker Compose configuration that reserves a GPU for the provider. Also configure `VOXSHIFT_DEVICE=cuda` and a CUDA-compatible Torch wheel index (check compatibility for the selected GPU and PyTorch version). Do not change the existing CPU template blindly.

## Security and cost

This is a private, authorization-gated voice service for vocals you own or have permission to process. The reference audio is not a license to impersonate another person. Restrict host access, rotate leaked provider keys, and do not enable anonymous processing. GPU servers, bandwidth, DNS, and cloud hosting can incur charges. The template provisions **no** infrastructure and incurs no cost on its own.
