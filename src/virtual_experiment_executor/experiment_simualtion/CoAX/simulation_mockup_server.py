import json
import mimetypes
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import unquote, urlparse

from generate_strategy_sample_from_params import build_payload


ROOT = Path(__file__).resolve().parent
DEFAULT_HOST = os.environ.get("HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("PORT", "5000"))


def _strip_api_prefix(path: str) -> str:
    if path == "/api":
        return "/"
    if path.startswith("/api/"):
        return path.removeprefix("/api")
    return path


class MockupHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = _strip_api_prefix(parsed.path)
        if path == "/coax_simulator":
            self.serve_file(ROOT / "simulation_mockup" / "coax_simulator.html")
            return
        if path in {"/", "/mockup", "/case_by_case"}:
            self.serve_file(ROOT / "simulation_mockup" / "index.html")
            return
        if path == "/global_statistics":
            self.serve_file(ROOT / "simulation_mockup" / "xai_chart.html")
            return

        requested = (ROOT / unquote(path.lstrip("/"))).resolve()
        if ROOT not in requested.parents and requested != ROOT:
            self.send_error(403)
            return
        if requested.is_file():
            self.serve_file(requested)
            return
        self.send_error(404)

    def do_POST(self):
        if _strip_api_prefix(urlparse(self.path).path) != "/simulate":
            self.send_error(404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length).decode("utf-8"))
            args = SimpleNamespace(
                dataset=data["dataset"],
                xai_type=data["xai_type"],
                reasoning_strategy=data.get("reasoning_strategy"),
                tested=data["tested"],
                instance_number=int(data["instance_number"]),
                k=int(data["k"]),
                retrieval_threshold=float(data["retrieval_threshold"]),
                sensitivity=float(data["sensitivity"]),
                scaling_factor=float(data["scaling_factor"]),
                decay_param=float(data.get("decay_param", 0.5)),
                n_sessions=int(data.get("n_sessions", 2)),
                closest_k=int(data.get("closest_k", 7)),
                session=int(data.get("session", 1)),
                block_index=int(data.get("block_index", 0)),
                instance_id=data.get("instance_id"),
                seed=int(data.get("seed", 1234)),
                train_with_explanation=bool(data.get("train_with_explanation", True)),
                user_study_path=data.get("user_study_path", "data/user study results/3-datasets-jan-09-2026-trials.csv"),
                output_prefix=data.get("output_prefix", "mockup"),
            )
            payload = build_payload(args)
        except Exception as exc:
            self.send_json({"error": str(exc)}, status=400)
            return

        self.send_json(payload)

    def serve_file(self, path):
        content_type = mimetypes.guess_type(path)[0] or "application/octet-stream"
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload, status=200):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        return


def main():
    server = ThreadingHTTPServer((DEFAULT_HOST, DEFAULT_PORT), MockupHandler)
    print(f"Case by case: http://{DEFAULT_HOST}:{DEFAULT_PORT}/case_by_case")
    print(f"Global statistics: http://{DEFAULT_HOST}:{DEFAULT_PORT}/global_statistics")
    print(f"CoAX simulator: http://{DEFAULT_HOST}:{DEFAULT_PORT}/coax_simulator")
    print(f"Nginx/API case by case: http://{DEFAULT_HOST}:{DEFAULT_PORT}/api/case_by_case")
    print(f"Nginx/API global statistics: http://{DEFAULT_HOST}:{DEFAULT_PORT}/api/global_statistics")
    print(f"Nginx/API CoAX simulator: http://{DEFAULT_HOST}:{DEFAULT_PORT}/api/coax_simulator")
    server.serve_forever()


if __name__ == "__main__":
    main()
