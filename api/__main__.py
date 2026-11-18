"""Start the PathSnap API: ``uv run python -m api --data-dir <trips folder>``.

Listens on all interfaces (0.0.0.0) so the UI laptop can reach it over the LAN.
"""

from __future__ import annotations

import argparse
import os
import socket

import uvicorn


def lan_addresses() -> list[str]:
    addrs = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))     # no packet is sent; picks the outward interface
            addrs.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        addrs.update(i[4][0] for i in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET))
    except OSError:
        pass
    return sorted(a for a in addrs if not a.startswith("127."))


def main() -> None:
    ap = argparse.ArgumentParser(description="PathSnap FastAPI backend")
    ap.add_argument("--data-dir", help="folder with one sub-folder per trip to serve "
                                       "(sets PATHSNAP_EVAL_DIR; default <PATHSNAP_DATA>/student_test)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    if args.data_dir:
        os.environ["PATHSNAP_EVAL_DIR"] = os.path.abspath(args.data_dir)
    print(f"PathSnap API on port {args.port}. Point the UI at one of:")
    for a in lan_addresses() or ["<this laptop's IP>"]:
        print(f"    http://{a}:{args.port}")
    print(f"    http://localhost:{args.port}   (same machine)")
    # Request lines come from the app's own middleware (which skips the UI's /logs polling).
    uvicorn.run("api.app:app", host=args.host, port=args.port, log_level="info", access_log=False)


if __name__ == "__main__":
    main()
