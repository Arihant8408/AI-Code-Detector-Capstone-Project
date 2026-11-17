"""
imaging.py - image support for the AI Code Checker: OCR + AI-image signals.

Optional extras (the text checker still runs without them):
    pip install pillow numpy          and the `tesseract` program  (apt install tesseract-ocr / Windows installer)

1. OCR       : Tesseract reads a screenshot/photo of code. Word boxes (TSV) are used to rebuild indentation
               and blank lines, then the text goes to engine.CodeChecker.
2. AI image  : explainable signals (generator metadata, camera EXIF, typical generator sizes, noise level,
               frequency-domain peaks). Optionally combined with a pretrained classifier when the environment
               variable AI_IMAGE_MODEL is set (needs `pip install transformers torch`; NOT tested in the build sandbox).

IMPORTANT: no image detector is reliable. Metadata can be stripped or forged, and pixel statistics are weak evidence.
"""
from __future__ import annotations

import io
import os
import shutil
import statistics
import subprocess
import tempfile

try:
    import numpy as np
    from PIL import Image, ImageFilter, ImageOps
    PIL_OK = True
except ImportError:  # pragma: no cover
    PIL_OK = False

def _find_tesseract():
    """PATH first, then TESSERACT_CMD, then the usual install folders (Windows installers do not add it to PATH)."""
    for cand in (os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
                 r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                 os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
                 "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract"):
        if cand and os.path.isfile(cand):
            return cand
    return None


TESSERACT = _find_tesseract()
MAX_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 25_000_000
ALLOWED = {"PNG", "JPEG", "WEBP", "GIF", "BMP"}


def capabilities() -> dict:
    return {"image_support": PIL_OK, "ocr": bool(PIL_OK and TESSERACT),
            "pretrained_model": os.environ.get("AI_IMAGE_MODEL") or None,
            "hint": None if PIL_OK and TESSERACT else
            "Install: pip install pillow numpy  and the tesseract program (OCR needs both)."}


# --------------------------------------------------------------------------- loading
def open_image(data: bytes):
    if not PIL_OK:
        raise RuntimeError("Image support is not installed. Run: pip install pillow numpy")
    if not data:
        raise ValueError("Empty upload.")
    if len(data) > MAX_BYTES:
        raise ValueError(f"Image is too large (limit {MAX_BYTES // (1024 * 1024)} MB).")
    try:
        img = Image.open(io.BytesIO(data))
        fmt = img.format
        w, h = img.size
    except Exception:
        raise ValueError("Could not read this file as an image (PNG, JPEG, WebP, GIF, BMP are supported).")
    if fmt not in ALLOWED:
        raise ValueError(f"Unsupported image type: {fmt}.")
    if w * h > MAX_PIXELS:
        raise ValueError(f"Image is too many pixels ({w}x{h}). Limit is {MAX_PIXELS // 1_000_000} megapixels.")
    try:
        if getattr(img, "n_frames", 1) > 1:
            img.seek(0)
        img.load()
    except Exception:
        raise ValueError("The image file is damaged.")
    return img


def _flatten_rgb(img):
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        return Image.alpha_composite(bg, rgba).convert("RGB")
    return img.convert("RGB")


# --------------------------------------------------------------------------- OCR
def _prepare_for_ocr(img):
    g = ImageOps.grayscale(_flatten_rgb(img))
    if g.width < 1400:
        g = g.resize((g.width * 2, g.height * 2), Image.LANCZOS)
    if g.width > 4000:
        r = 4000 / g.width
        g = g.resize((4000, max(1, int(g.height * r))), Image.LANCZOS)
    if np.asarray(g).mean() < 120:          # dark theme -> dark text on light background
        g = ImageOps.invert(g)
    g = ImageOps.autocontrast(g, cutoff=1)
    return ImageOps.expand(g, border=24, fill=255)


def _tesseract_words(img) -> list[dict]:
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "in.png")
        img.save(path)
        env = dict(os.environ, OMP_THREAD_LIMIT="1")
        proc = subprocess.run([TESSERACT, path, "stdout", "--psm", "6", "-c", "preserve_interword_spaces=1", "tsv"],
                              capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env=env)
    if proc.returncode != 0:
        raise RuntimeError("Tesseract failed: " + proc.stderr.strip()[:200])
    words = []
    for row in proc.stdout.split("\n")[1:]:
        p = row.split("\t")
        if len(p) < 12 or p[0] != "5":
            continue
        text = p[11].strip()
        try:
            conf = float(p[10])
        except ValueError:
            continue
        if not text or conf < 0:
            continue
        words.append({"line": (int(p[2]), int(p[3]), int(p[4])), "left": int(p[6]), "top": int(p[7]),
                      "w": int(p[8]), "h": int(p[9]), "conf": conf, "text": text})
    return words


def _rebuild_text(words: list[dict]) -> str:
    """Use word boxes to put back indentation, spacing and blank lines (matters for code)."""
    if not words:
        return ""
    widths = [w["w"] / len(w["text"]) for w in words if len(w["text"]) >= 3]
    cw = statistics.median(widths) if widths else 10.0
    cw = max(cw, 1.0)
    lines: dict[tuple, list[dict]] = {}
    for w in words:
        lines.setdefault(w["line"], []).append(w)
    ordered = sorted(lines.values(), key=lambda ws: min(w["top"] for w in ws))
    min_left = min(w["left"] for w in words)
    heights = [statistics.median(w["h"] for w in ws) for ws in ordered]
    lh = statistics.median(heights) if heights else 10
    out, prev_bottom = [], None
    for ws in ordered:
        ws.sort(key=lambda w: w["left"])
        top = min(w["top"] for w in ws)
        if prev_bottom is not None and top - prev_bottom > lh * 1.1:
            out.append("")
        text = " " * round((ws[0]["left"] - min_left) / cw)
        prev_right = None
        for w in ws:
            if prev_right is not None:
                text += " " * int((w["left"] - prev_right) / cw + 0.15)   # 0 spaces = Tesseract split one token (e.g. calculate | _area)
            text += w["text"]
            prev_right = w["left"] + w["w"]
        out.append(text)
        prev_bottom = max(w["top"] + w["h"] for w in ws)
    return "\n".join(out)


_SMART = str.maketrans({"\u201c": '"', "\u201d": '"', "\u2018": "'", "\u2019": "'"})


def run_ocr(img) -> dict:
    if not TESSERACT:
        return {"available": False, "text": "", "confidence": 0, "words": 0, "lines": 0,
                "note": "Tesseract is not installed (or not found), so OCR is unavailable. Install it, then restart the app."}
    words = _tesseract_words(_prepare_for_ocr(img))
    text = _rebuild_text(words).translate(_SMART)   # code never contains curly quotes, so OCR's are misreads
    conf = sum(w["conf"] * len(w["text"]) for w in words) / max(1, sum(len(w["text"]) for w in words)) if words else 0
    return {"available": True, "text": text, "confidence": round(conf), "words": len(words),
            "lines": len([l for l in text.split("\n") if l.strip()]), "note": ""}


# --------------------------------------------------------------------------- AI-image signals
PROMPT_MARKERS = [b"negative prompt", b"cfg scale", b"sampler:", b"comfyui", b"automatic1111", b"novelai", b"invokeai"]
GENERATOR_MARKERS = [b"stable diffusion", b"midjourney", b"dall-e", b"dall\xc2\xb7e", b"adobe firefly", b"leonardo.ai",
                     b"openai", b"made with google ai"]
COMMON_SIZES = {(512, 512), (768, 768), (1024, 1024), (1024, 1536), (1536, 1024), (1792, 1024), (1024, 1792),
                (832, 1216), (1216, 832), (640, 640), (896, 1152), (1152, 896), (2048, 2048)}
WEAK_CAP = 30


def _metadata_signals(img, data: bytes) -> list[dict]:
    sig = []
    blob = (data[:262144] + repr(img.info).encode("utf-8", "ignore")[:200000]).lower()
    hit = [m.decode("latin1") for m in PROMPT_MARKERS if m in blob]
    if hit:
        sig.append({"name": "Generation parameters in metadata", "weight": 55, "kind": "ai", "group": "meta",
                    "detail": "Found " + ", ".join(hit) + " - settings written by image generators."})
    hit = [m.decode("utf-8", "ignore") for m in GENERATOR_MARKERS if m in blob]
    if hit:
        sig.append({"name": "Generator name in metadata", "weight": 45, "kind": "ai", "group": "meta",
                    "detail": "Found: " + ", ".join(hit) + "."})
    if b"c2pa" in blob or b"cabx" in blob:
        sig.append({"name": "Content Credentials (C2PA) present", "weight": 12, "kind": "ai", "group": "meta",
                    "detail": "Provenance data is embedded. Check who signed it: cameras and editors use C2PA too."})
    try:
        ex = img.getexif()
        sub = {}
        try:
            sub = ex.get_ifd(0x8769)
        except Exception:
            pass
        make, model = ex.get(271), ex.get(272)
        shot = any(k in sub for k in (33434, 34855, 36867))
        if (make or model) and shot:
            sig.append({"name": "Camera EXIF present", "weight": -25, "kind": "human", "group": "meta",
                        "detail": f"{(str(make or '') + ' ' + str(model or '')).strip()} with exposure/ISO/date. EXIF can be forged."})
        elif make or model:
            sig.append({"name": "Camera make/model in EXIF", "weight": -12, "kind": "human", "group": "meta",
                        "detail": "Partial camera data. EXIF can be forged."})
        elif not ex and not img.info.get("dpi") and img.format == "JPEG":
            sig.append({"name": "No metadata", "weight": 0, "kind": "info", "group": "meta",
                        "detail": "Stripped metadata is common for both AI images and re-shared real photos."})
    except Exception:
        pass
    return sig


def _is_graphic(rgb) -> tuple[bool, float]:
    """Screenshots and flat graphics have large areas of exactly identical pixels; photographs do not."""
    thumb = rgb.copy()
    thumb.thumbnail((400, 400), Image.NEAREST)
    g = np.asarray(ImageOps.grayscale(thumb)).astype(np.int16)
    same = (g[:-1, :-1] == g[:-1, 1:]) & (g[:-1, :-1] == g[1:, :-1])
    flat = float(same.mean()) if same.size else 0.0
    return flat > 0.6, flat


def _center_crop(arr, size):
    h, w = arr.shape
    s = min(size, h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    return arr[y0:y0 + s, x0:x0 + s]


def _noise_level(gray_img) -> tuple[float, float]:
    g = _center_crop(np.asarray(gray_img, dtype=np.float32), 512)
    crop = Image.fromarray(g.astype(np.uint8))
    resid = g - np.asarray(crop.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
    bs = 32
    h, w = (resid.shape[0] // bs) * bs, (resid.shape[1] // bs) * bs
    if h == 0 or w == 0:
        return float(resid.std()), 0.0
    blocks = resid[:h, :w].reshape(h // bs, bs, w // bs, bs).std(axis=(1, 3)).ravel()
    return float(np.median(blocks)), float(blocks.std() / (blocks.mean() + 1e-6))


def _spectral_peak(gray_img) -> float | None:
    """Largest z-score of a spectral peak relative to its frequency ring. Upsampling leaves periodic peaks."""
    g = np.asarray(gray_img, dtype=np.float32)
    s = 256
    if g.shape[0] < s or g.shape[1] < s:
        return None
    c = _center_crop(g, s)
    c = (c - c.mean()) * (np.hanning(s)[:, None] * np.hanning(s)[None, :])
    mag = np.log(np.abs(np.fft.fftshift(np.fft.fft2(c))) + 1e-3)
    yy, xx = np.indices(mag.shape)
    dy, dx = yy - s // 2, xx - s // 2
    r = np.sqrt(dx * dx + dy * dy).astype(int)
    cnt = np.bincount(r.ravel())
    mean = np.bincount(r.ravel(), mag.ravel()) / np.maximum(cnt, 1)
    var = np.bincount(r.ravel(), (mag ** 2).ravel()) / np.maximum(cnt, 1) - mean ** 2
    z = (mag - mean[r]) / np.sqrt(np.maximum(var[r], 1e-6))
    ok = (r > 40) & (r < 126)
    ok &= (np.abs(dx) > 3) & (np.abs(dy) > 3)
    return float(z[ok].max()) if ok.any() else None


def _hf_signal(img):
    name = os.environ.get("AI_IMAGE_MODEL")
    if not name:
        return None
    try:
        from transformers import pipeline  # type: ignore
        global _PIPE
        if "_PIPE" not in globals() or _PIPE[0] != name:
            _PIPE = (name, pipeline("image-classification", model=name))
        out = _PIPE[1](_flatten_rgb(img))
        p_ai = sum(o["score"] for o in out if any(k in o["label"].lower() for k in ("artificial", "ai", "fake", "generated")))
        return {"name": f"Pretrained classifier ({name})", "weight": round((p_ai - 0.5) * 80), "kind": "model", "group": "model",
                "detail": f"Model says {p_ai:.0%} AI-generated."}
    except Exception as exc:
        return {"name": "Pretrained classifier unavailable", "weight": 0, "kind": "info", "group": "model", "detail": str(exc)[:160]}


def analyse_image(img, data: bytes) -> dict:
    w, h = img.size
    signals = _metadata_signals(img, data)
    rgb = _flatten_rgb(img)
    graphic, flat = _is_graphic(rgb)
    info = {"width": w, "height": h, "format": img.format, "megapixels": round(w * h / 1e6, 2)}
    weak = []
    if graphic:
        signals.append({"name": "Screenshot / graphic", "weight": 0, "kind": "info", "group": "pixel",
                        "detail": f"{flat:.0%} of the image is perfectly flat colour. Pixel statistics only work on photographs, so they were skipped."})
    else:
        if (w, h) in COMMON_SIZES:
            weak.append({"name": "Common generator output size", "weight": 8, "kind": "ai", "group": "pixel",
                         "detail": f"{w}x{h} is a typical generator size (also a common crop size)."})
        elif w % 64 == 0 and h % 64 == 0 and min(w, h) >= 512:
            weak.append({"name": "Both sides multiples of 64", "weight": 5, "kind": "ai", "group": "pixel",
                         "detail": f"{w}x{h}: generators work in 64-pixel steps; cameras rarely do."})
        gray = ImageOps.grayscale(rgb)
        noise, cv = _noise_level(gray)
        info["noise_level"] = round(noise, 2)
        if noise < 0.9:
            weak.append({"name": "Very smooth (low noise)", "weight": 6, "kind": "ai", "group": "pixel",
                         "detail": f"Residual noise {noise:.2f}. Camera sensors add noise; generators are often too clean. Heavy compression or denoising also does this."})
        else:
            signals.append({"name": "Sensor-like noise present", "weight": 0, "kind": "info", "group": "pixel",
                            "detail": f"Residual noise {noise:.2f}, block variation {cv:.2f}."})
        # JPEG/WebP block grids create the same peaks as upsampling, so only lossless formats are tested
        peak = _spectral_peak(gray) if img.format in ("PNG", "BMP", "GIF") else None
        if peak is None and img.format in ("JPEG", "WEBP"):
            signals.append({"name": "Frequency check skipped", "weight": 0, "kind": "info", "group": "pixel",
                            "detail": f"{img.format} compression adds its own periodic pattern, so this test is only used on lossless images."})
        if peak is not None:
            info["spectral_peak_z"] = round(peak, 1)
            if peak > 5.5:
                weak.append({"name": "Strong periodic frequency peak", "weight": 12, "kind": "ai", "group": "pixel",
                             "detail": f"Spectral peak z={peak:.1f}. Upsampling layers in generators leave periodic patterns."})
            elif peak > 4.5:
                weak.append({"name": "Mild periodic frequency peak", "weight": 8, "kind": "ai", "group": "pixel",
                             "detail": f"Spectral peak z={peak:.1f}."})
        # weak pixel evidence is capped so it can never reach "Likely AI" alone
        total = sum(s["weight"] for s in weak)
        if total > WEAK_CAP:
            f = WEAK_CAP / total
            for s in weak:
                s["weight"] = round(s["weight"] * f, 1)
        signals.extend(weak)
    model = _hf_signal(img)
    if model:
        signals.append(model)

    score = max(0, min(100, round(30 + sum(s["weight"] for s in signals))))
    decisive = any(s["group"] in ("meta", "model") and abs(s["weight"]) >= 25 for s in signals)
    if graphic and not decisive:
        verdict, label = "na", "Not applicable (screenshot or graphic)"
    elif score >= 70:
        verdict, label = "ai", "Likely AI-generated"
    elif score >= 45:
        verdict, label = "possible", "Possibly AI-generated"
    elif score >= 20:
        verdict, label = "unsure", "Inconclusive"
    else:
        verdict, label = "real", "Likely a real photo"
    return {"label": label, "verdict": verdict, "score": score, "confidence": "high" if decisive else "low",
            "signals": signals, "info": info,
            "disclaimer": "Image detectors are unreliable. Metadata can be stripped or forged and pixel statistics are weak evidence. Treat this as a hint, not proof."}


# --------------------------------------------------------------------------- entry point
def process(data: bytes) -> dict:
    img = open_image(data)
    image = analyse_image(img, data)
    ocr = run_ocr(img)
    return {"image": image, "ocr": ocr}
