"""Serve the 3D replay viewer: index.html from this folder, the assets (build_replay.py output) from ASSET_DIR, with
cross-origin isolation headers so the splat sorter can use shared memory.
usage: serve.py ASSET_DIR [--port 8792]   then open http://127.0.0.1:8792/"""
import functools, os, sys
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = sys.argv[1]
PORT = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8792


class Handler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        if path.split("?")[0].split("#")[0] in ("/", "/index.html"):
            return os.path.join(HERE, "index.html")                       # the page itself: live from the repo
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def log_message(self, *a):
        pass


print("replay viewer: http://127.0.0.1:%d/  (assets %s)" % (PORT, ASSETS), flush=True)
ThreadingHTTPServer(("127.0.0.1", PORT), functools.partial(Handler, directory=ASSETS)).serve_forever()
