"""
Daily Kite Connect OAuth token refresh.

Run this ONCE each morning before 9:20 IST:
    python scripts/kite_auth.py

Steps:
  1. Opens Kite login URL in your browser
  2. You log in and are redirected to a localhost page
  3. Paste the full redirect URL (or just the request_token value)
  4. Script exchanges it for access_token and writes it to .env
"""

import os
import re
import sys
import webbrowser
from pathlib import Path

# Load .env so ZERODHA_API_KEY / ZERODHA_API_SECRET are available
from dotenv import load_dotenv, set_key

load_dotenv()

try:
    from kiteconnect import KiteConnect
except ImportError:
    sys.exit("kiteconnect not installed. Run: pip install kiteconnect")


def _get_required(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        sys.exit(f"Missing {name} in .env — add it and retry.")
    return val


def _extract_token(raw: str) -> str:
    # Accept full redirect URL or bare token string
    match = re.search(r"request_token=([A-Za-z0-9]+)", raw)
    if match:
        return match.group(1)
    stripped = raw.strip()
    if re.fullmatch(r"[A-Za-z0-9]+", stripped):
        return stripped
    sys.exit(
        "Could not parse request_token from input. Paste the full redirect URL or just the token."
    )


def main() -> None:
    api_key = _get_required("ZERODHA_API_KEY")
    api_secret = _get_required("ZERODHA_API_SECRET")

    kite = KiteConnect(api_key=api_key)
    login_url = kite.login_url()

    print("\n──────────────────────────────────────────────")
    print("Kite Connect daily token refresh")
    print("──────────────────────────────────────────────")
    print(f"\nOpening login URL in browser:\n{login_url}\n")
    webbrowser.open(login_url)

    print(
        "After login, Zerodha redirects to your redirect_uri with ?request_token=XXXX"
    )
    print("Paste that full URL (or just the token) below:\n")
    raw = input("request_token or redirect URL: ").strip()

    request_token = _extract_token(raw)

    data = kite.generate_session(request_token, api_secret=api_secret)
    access_token: str = data["access_token"]

    # Write back to .env
    env_path = Path(".env")
    if not env_path.exists():
        sys.exit(".env not found — run from project root.")

    set_key(str(env_path), "ZERODHA_ACCESS_TOKEN", access_token)
    print(f"\nAccess token written to .env: {access_token[:8]}…")
    print("Token valid until midnight IST. Run this script again tomorrow morning.\n")


if __name__ == "__main__":
    main()
