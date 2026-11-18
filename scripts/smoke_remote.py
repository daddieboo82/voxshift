"""Read-only smoke test for a deployed VoxShift host (no audio, no generation cost).

Usage:
  VOXSHIFT_PROVIDER_URL=https://voice.example.org VOXSHIFT_API_KEY=... \
    python scripts/smoke_remote.py
"""
import hmac
import json
import os
import ssl
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def main():
    url = os.environ.get("VOXSHIFT_PROVIDER_URL", "").rstrip("/")
    key = os.environ.get("VOXSHIFT_API_KEY", "")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or len(key) < 32
    ):
        print("FAIL: Set the HTTPS provider origin and its 32+ character API key.")
        return 2
    try:
        with urlopen(Request(url + "/healthz"), timeout=12) as response:
            health = json.load(response)
        with urlopen(Request(url + "/v1/capabilities",
                             headers={"Authorization": "Bearer " + key}),
                     timeout=12) as response:
            capabilities = json.load(response)
        if health.get("status") != "ready":
            raise ValueError("Health endpoint is not ready")
        modes = capabilities.get("modes")
        if not isinstance(modes, list) or not {"speech", "convert", "sing"}.issubset(set(modes)):
            raise ValueError("Voice modes are unavailable")
    except (HTTPError, URLError, ValueError, TimeoutError, ssl.SSLError, json.JSONDecodeError) as error:
        print("FAIL: Provider health/capabilities checks failed: " + type(error).__name__)
        return 1
    print("PASS: Provider reports ready and authenticated Voice Match modes.")
    print("Note: This does not test actual Chatterbox inference or Base44 connection.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
