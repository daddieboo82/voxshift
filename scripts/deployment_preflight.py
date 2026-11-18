"""Validate local VoxShift deployment settings without starting containers or spending money.

Run: python scripts/deployment_preflight.py --env deploy/.env
"""
import argparse
import re
import secrets
import shutil
import subprocess
import sys
from pathlib import Path


def parse_env(path):
    if not path.is_file():
        raise ValueError(f"Missing {path}; copy deploy/.env.example to deploy/.env")
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError("Invalid environment variable line")
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def validate(values):
    errors = []
    domain = values.get("VOXSHIFT_DOMAIN", "")
    if (
        not re.fullmatch(r"(?=.{1,253}$)(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,63}", domain)
        or domain.lower().endswith(".example.com")
    ):
        errors.append("VOXSHIFT_DOMAIN must be a real public DNS hostname, not the example.")
    key = values.get("VOXSHIFT_API_KEY", "")
    if (
        not key.isascii()
        or len(key) < 32
        or "replace" in key.lower()
        or len(set(key)) < 12
    ):
        errors.append("VOXSHIFT_API_KEY must be a unique generated ASCII secret (32+ characters).")
    device = values.get("VOXSHIFT_DEVICE", "cpu")
    if device not in ("cpu", "cuda"):
        errors.append("VOXSHIFT_DEVICE must be cpu or cuda.")
    if device == "cuda" and "cpu" in values.get("VOXSHIFT_TORCH_INDEX", "").lower():
        errors.append("CUDA device selected but the PyTorch package index still targets CPU.")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", type=Path, default=Path("deploy/.env"))
    parser.add_argument("--skip-docker", action="store_true", help="Validate .env only")
    args = parser.parse_args()
    try:
        errors = validate(parse_env(args.env))
    except (ValueError, OSError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    if not args.skip_docker:
        if shutil.which("docker") is None:
            errors.append("Docker not found; install Docker Engine and Compose on the inference host.")
        else:
            try:
                result = subprocess.run(
                    ["docker", "compose", "-f", str(args.env.parent / "compose.yaml"),
                     "--env-file", str(args.env), "config", "--quiet"],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                if result.returncode:
                    errors.append("Docker Compose configuration invalid; check docker compose config locally.")
            except (OSError, subprocess.TimeoutExpired):
                errors.append("Docker Compose validation could not complete.")
    for error in errors:
        print(f"FAIL: {error}", file=sys.stderr)
    if errors:
        return 1
    print("PASS: VoxShift deployment environment and configuration look valid.")
    print("This does not prove DNS, TLS, model inference, or Base44 integration is live.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
