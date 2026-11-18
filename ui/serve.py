"""Serve the PathSnap evaluator UI (Laptop 2). Standard library only.

    python serve.py --api http://<laptop-1-ip>:8000 [--port 5173]

Then open http://localhost:5173 in a browser. The UI calls the API directly
from the browser; this server only hands out the static files.
"""

from __future__ import annotations

import argparse
import functools
import http.server
import json
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    api_url = ""

    def do_GET(self):
        if self.path.split("?")[0] == "/config.js" and self.api_url:
            body = f"window.PATHSNAP_API = {json.dumps(self.api_url)};\n".encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/javascript")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return
        super().do_GET()

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the PathSnap UI")
    ap.add_argument("--api", default="", help="PathSnap API base URL, e.g. http://192.168.1.10:8000")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=5173)
    ap.add_argument("--open", action="store_true", help="open the UI in the default browser")
    args = ap.parse_args()
    Handler.api_url = args.api.rstrip("/")
    handler = functools.partial(Handler, directory=str(HERE))
    with http.server.ThreadingHTTPServer((args.host, args.port), handler) as httpd:
        url = f"http://localhost:{args.port}"
        print(f"PathSnap UI on {url}" + (f"  (API: {Handler.api_url})" if Handler.api_url else ""))
        if args.open:
            webbrowser.open(url)
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
