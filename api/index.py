"""Vercel entry point. Reuses app.Handler and adds cloud OCR (Tesseract is unavailable on Vercel)."""
import io
import json
import os
import sys
import urllib.request
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import app as _app  # noqa: E402

imaging = _app.imaging
handler = _app.Handler  # Vercel looks for this name

OCR_URL = "https://api.ocr.space/parse/image"
OCR_LIMIT = 1_000_000  # OCR.space free tier accepts files up to 1 MB


def _shrink(data: bytes) -> bytes:
    """Re-encode as JPEG under 1 MB so the free OCR tier accepts it."""
    from PIL import Image

    img = Image.open(io.BytesIO(data)).convert("RGB")
    for _ in range(8):
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=85)
        if buf.tell() <= OCR_LIMIT:
            return buf.getvalue()
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


if imaging is not None:
    _orig_process = imaging.process
    _orig_caps = imaging.capabilities

    def process(data):
        res = _orig_process(data)
        key = os.environ.get("OCR_SPACE_API_KEY")
        ocr = res.get("ocr") or {}
        if key and not ocr.get("available"):
            try:
                text = _remote_ocr(data, key)
                words = len(text.split())
                res["ocr"] = {**ocr, "available": True, "text": text, "words": words,
                              "lines": len([l for l in text.split("\n") if l.strip()]),
                              "confidence": 85 if words else 0, "note": ""}
            except Exception as exc:  # keep image check working even if OCR fails
                print("Remote OCR failed:", exc, file=sys.stderr)
        return res

    def capabilities():
        caps = dict(_orig_caps())
        if os.environ.get("OCR_SPACE_API_KEY"):
            caps["ocr"] = True
            caps["hint"] = None
        return caps

    imaging.process = process
    imaging.capabilities = capabilities
