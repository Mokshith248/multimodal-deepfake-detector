from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
import json
import uuid
import cgi
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.image_detector import detect_image
from inference.video_detector import detect_video
from inference.audio_detector import detect_audio
from inference.audiovisual_detector import detect_audiovisual

HOST = "127.0.0.1"
PORT = 8000

UPLOAD_DIR = os.path.join("backend", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)


def run_prediction(mode, file_path):
    if mode == "image":
        return detect_image(file_path)

    if mode == "video":
        return detect_video(file_path)

    if mode == "audio":
        return detect_audio(file_path)

    if mode == "audiovisual":
        return detect_audiovisual(file_path)

    raise ValueError("Unsupported detection mode")


class DetectionHandler(BaseHTTPRequestHandler):

    def send_json(self, data, status=200):
        response = json.dumps(data).encode("utf-8")

        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()

        self.wfile.write(response)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.end_headers()

    def do_GET(self):

        if self.path == "/":
            self.send_json({
                "message": "Multimodal Deepfake Detection API is running"
            })
            return

        if self.path == "/health":
            self.send_json({
                "status": "healthy"
            })
            return

        self.send_json({
            "error": "Endpoint not found"
        }, 404)

    def do_POST(self):

        routes = {
            "/predict/image": "image",
            "/predict/video": "video",
            "/predict/audio": "audio",
            "/predict/audiovisual": "audiovisual"
        }

        if self.path not in routes:
            self.send_json({
                "error": "Endpoint not found"
            }, 404)
            return

        mode = routes[self.path]

        try:
            content_type = self.headers.get("Content-Type", "")

            if "multipart/form-data" not in content_type:
                self.send_json({
                    "error": "Expected multipart/form-data upload"
                }, 400)
                return

            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": content_type,
                    "CONTENT_LENGTH": self.headers.get("Content-Length", "")
                }
            )

            if "file" not in form:
                self.send_json({
                    "error": "No file uploaded"
                }, 400)
                return

            uploaded_file = form["file"]

            if not uploaded_file.filename:
                self.send_json({
                    "error": "Invalid filename"
                }, 400)
                return

            original_name = os.path.basename(uploaded_file.filename)
            extension = os.path.splitext(original_name)[1]

            unique_name = str(uuid.uuid4()) + extension
            file_path = os.path.join(UPLOAD_DIR, unique_name)

            with open(file_path, "wb") as output:
                output.write(uploaded_file.file.read())

            print()
            print("=" * 50)
            print("DETECTION REQUEST")
            print("=" * 50)
            print("Mode :", mode)
            print("File :", original_name)
            print("Running model...")

            result = run_prediction(mode, file_path)

            print("Result:", result)
            print("=" * 50)

            self.send_json(result)

        except Exception as error:

            print()
            print("ERROR:", str(error))

            self.send_json({
                "error": str(error)
            }, 500)

        finally:

            if "file_path" in locals() and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass


if __name__ == "__main__":

    print()
    print("=" * 60)
    print("MULTIMODAL DEEPFAKE DETECTION API")
    print("=" * 60)
    print()
    print(f"Server running at: http://{HOST}:{PORT}")
    print("Health check:      http://127.0.0.1:8000/health")
    print()
    print("Endpoints:")
    print("  POST /predict/image")
    print("  POST /predict/video")
    print("  POST /predict/audio")
    print("  POST /predict/audiovisual")
    print()
    print("Press CTRL+C to stop the server.")
    print("=" * 60)

    server = ThreadingHTTPServer((HOST, PORT), DetectionHandler)
    server.serve_forever()