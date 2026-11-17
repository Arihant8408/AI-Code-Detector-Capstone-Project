"""
app.py - web server for the AI Code Checker (standard library only, no pip install needed).

Run:    python app.py                 -> http://127.0.0.1:8000
        python app.py --port 5000
        HOST=0.0.0.0 PORT=10000 python app.py      (for hosting platforms)
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import engine

try:  # image features are optional extras
    import imaging
except Exception:  # pragma: no cover
    imaging = None

# ---- Vercel support: cloud OCR (Tesseract is not available on Vercel) ----
import io
import urllib.request
import uuid

OCR_URL = "https://api.ocr.space/parse/image"
OCR_LIMIT = 1_000_000  # OCR.space free tier accepts files up to 1 MB


def _shrink(data: bytes) -> bytes:
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    for _ in range(8):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        if buf.tell() <= OCR_LIMIT:
            break
        img = img.resize((int(img.width * 0.75), int(img.height * 0.75)))
    return buf.getvalue()


def _remote_ocr(data: bytes, api_key: str) -> str:
    jpg = _shrink(data)
    boundary = uuid.uuid4().hex
    fields = {"apikey": api_key, "language": "eng", "OCREngine": "2", "scale": "true", "isOverlayRequired": "false"}
    body = b""
    for k, v in fields.items():
        body += f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode()
    body += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="img.jpg"\r\n'
             f'Content-Type: image/jpeg\r\n\r\n').encode() + jpg + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    req = urllib.request.Request(OCR_URL, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=25) as resp:
        out = json.loads(resp.read().decode("utf-8"))
    if out.get("IsErroredOnProcessing"):
        raise RuntimeError(str(out.get("ErrorMessage")))
    return "\n".join(r.get("ParsedText", "") for r in out.get("ParsedResults", [])).replace("\r\n", "\n").strip()


if imaging is not None and os.environ.get("OCR_SPACE_API_KEY"):
    _orig_process = imaging.process
    _orig_caps = imaging.capabilities

    def _process(data):
        res = _orig_process(data)
        ocr = res.get("ocr") or {}
        if not ocr.get("available"):
            try:
                text = _remote_ocr(data, os.environ["OCR_SPACE_API_KEY"])
                words = len(text.split())
                res["ocr"] = {**ocr, "available": True, "text": text, "words": words,
                              "lines": len([l for l in text.split("\n") if l.strip()]),
                              "confidence": 85 if words else 0, "note": ""}
            except Exception as exc:
                print("Remote OCR failed:", exc, file=sys.stderr)
        return res

    def _capabilities():
        caps = dict(_orig_caps())
        caps["ocr"] = True
        caps["hint"] = None
        return caps

    imaging.process = _process
    imaging.capabilities = _capabilities

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
MAX_BODY = 128 * 1024

CHECKER = engine.CodeChecker()
SAMPLES = json.loads((engine.DATA_DIR / "samples.json").read_text(encoding="utf-8"))

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Content-Security-Policy": "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; "
                               "img-src 'self' data: blob:; connect-src 'self'; base-uri 'none'; form-action 'none'",
}


class Handler(BaseHTTPRequestHandler):
    server_version = "CodeChecker/1.0"

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def log_message(self, fmt, *args):
        sys.stderr.write("[%s] %s\n" % (self.log_date_time_string(), fmt % args))

    def do_GET(self):  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                self._send(200, (STATIC_DIR / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/health":
                self._json(200, {"status": "ok", **CHECKER.info(), "image": imaging.capabilities() if imaging else {"image_support": False, "ocr": False}})
            elif path == "/api/rules":
                self._json(200, {"rules": CHECKER.list_rules(), "info": CHECKER.info()})
            elif path == "/api/samples":
                self._json(200, {"samples": SAMPLES})
            else:
                self._error(404, "Not found")
        except Exception as exc:  # pragma: no cover
            self._error(500, f"Server error: {exc}")

    do_HEAD = do_GET  # noqa: N815

    def _image(self) -> None:
        if imaging is None or not imaging.PIL_OK:
            return self._error(503, "Image support is not installed. Run: pip install pillow numpy (and install Tesseract for OCR).")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._error(400, "Invalid Content-Length")
        if length <= 0:
            return self._error(400, "Empty upload")
        if length > imaging.MAX_BYTES:
            return self._error(413, "Image is too large (limit 8 MB)")
        data = self.rfile.read(length)
        try:
            res = imaging.process(data)
            ocr = res["ocr"]
            res["code_check"], res["notes"] = None, []
            text = ocr["text"][: engine.MAX_CHARS]
            if not ocr["available"]:
                pass  # the UI shows the OCR note itself
            elif len(text.strip()) < 15 or ocr["confidence"] < 45 or ocr["words"] < 3:
                res["notes"].append("No readable code text was found in this image, so only the image check ran.")
            else:
                res["code_check"] = CHECKER.check(text)
                res["notes"].append("OCR can misread characters (quotes, brackets, indentation). Edit the text and re-check if needed.")
            self._json(200, res)
        except ValueError as exc:
            self._error(400, str(exc))
        except subprocess.TimeoutExpired:
            self._error(504, "OCR took too long. Try a smaller image.")
        except Exception as exc:  # pragma: no cover
            self._error(500, f"Server error: {exc}")

    def do_POST(self):  # noqa: N802
        if urlparse(self.path).path == "/api/image":
            return self._image()
        if urlparse(self.path).path != "/api/check":
            return self._error(404, "Not found")
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._error(400, "Invalid Content-Length")
        if length <= 0:
            return self._error(400, "Empty request body")
        if length > MAX_BODY:
            return self._error(413, "Request too large")
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._error(400, "Body must be valid JSON")
        if not isinstance(payload, dict) or not isinstance(payload.get("code"), str):
            return self._error(400, 'JSON body must look like {"code": "..."}')
        lang = payload.get("language", "auto")
        try:
            self._json(200, CHECKER.check(payload["code"], lang if isinstance(lang, str) else "auto"))
        except ValueError as exc:
            self._error(400, str(exc))
        except Exception as exc:  # pragma: no cover
            self._error(500, f"Server error: {exc}")


handler = Handler  # entry point for Vercel


def _stop(signum, frame):
    """`docker stop` sends SIGTERM. As PID 1 in a container Python ignores it (no default handler), so Docker would wait
    10 s and then kill the process. Raising KeyboardInterrupt reuses the normal clean shutdown below."""
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Code Checker web app")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8000")))
    args = parser.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    signal.signal(signal.SIGTERM, _stop)
    print(f"AI Code Checker running at http://{args.host}:{args.port}  (Ctrl+C to stop)")
    print(f"Rule set: {CHECKER.info()['rules']} rules")
    cap = imaging.capabilities() if imaging else {"image_support": False, "ocr": False}
    print(f"Image upload: {'on' if cap['image_support'] else 'off (pip install pillow numpy)'}, OCR: {'on' if cap['ocr'] else 'off (install Tesseract)'}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
