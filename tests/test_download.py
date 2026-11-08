from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

from flybrain_snake.connectome import download


def test_download_resumes_after_truncated_response(tmp_path, monkeypatch):
    payload = b"flywire" * 512
    midpoint = len(payload) // 2
    attempts = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            requested = self.headers.get("Range")
            attempts.append(requested)
            if requested:
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {midpoint}-{len(payload) - 1}/{len(payload)}")
                self.send_header("Content-Length", str(len(payload) - midpoint))
                self.end_headers()
                self.wfile.write(payload[midpoint:])
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload[:midpoint])
                self.close_connection = True

    monkeypatch.setattr("flybrain_snake.connectome.time.sleep", lambda _: None)
    partial = tmp_path / "download.part"
    partial.touch()
    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            download(f"http://127.0.0.1:{server.server_port}/test", partial, len(payload))
        finally:
            server.shutdown()
            thread.join()
    assert attempts == [None, f"bytes={midpoint}-"]
    assert partial.read_bytes() == payload
