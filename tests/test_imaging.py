import io
import json
import sys
import threading
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402
import engine  # noqa: E402
import imaging  # noqa: E402

if imaging.PIL_OK:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

NEEDS_PIL = unittest.skipUnless(imaging.PIL_OK, "pillow/numpy not installed")
NEEDS_OCR = unittest.skipUnless(imaging.PIL_OK and imaging.TESSERACT, "tesseract not installed")
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CODE = ("# Initialize the counter\ncounter = 0\n# Loop through the items\nfor i in range(10):\n"
        "    # Add i to the counter\n    counter += i\n# Return the result\nprint(counter)\n")


def png(im, **kw):
    b = io.BytesIO()
    im.save(b, "PNG", **kw)
    return b.getvalue()


def photo(w=800, h=600, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = 120 + 40 * np.sin(xx / 70) * np.cos(yy / 90) + rng.normal(0, 12, (h, w))
    return Image.fromarray(np.clip(np.stack([base, base * .9, base * .8], -1), 0, 255).astype(np.uint8))


def code_image(code=CODE, dark=False):
    font = ImageFont.truetype(MONO, 16)
    lines = code.split("\n")
    bg, fg = ((30, 30, 36), (220, 220, 225)) if dark else ((255, 255, 255), (20, 20, 20))
    im = Image.new("RGB", (800, 24 * len(lines) + 30), bg)
    d = ImageDraw.Draw(im)
    for i, line in enumerate(lines):
        d.text((16, 14 + 24 * i), line, font=font, fill=fg)
    return im


def analyse(im_bytes):
    img = imaging.open_image(im_bytes)
    return imaging.analyse_image(img, im_bytes)


@NEEDS_PIL
class Loading(unittest.TestCase):
    def test_garbage_rejected(self):
        with self.assertRaises(ValueError):
            imaging.open_image(b"not an image")

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            imaging.open_image(b"")

    def test_too_many_pixels_rejected(self):
        with self.assertRaises(ValueError):
            imaging.open_image(png(Image.new("RGB", (6000, 5000), "white")))

    def test_too_many_bytes_rejected(self):
        with self.assertRaises(ValueError):
            imaging.open_image(b"\x89PNG" + b"0" * (imaging.MAX_BYTES + 1))

    def test_transparent_png_ok(self):
        imaging.open_image(png(Image.new("RGBA", (50, 50), (0, 0, 0, 0))))


@NEEDS_PIL
class Detector(unittest.TestCase):
    def test_generation_parameters_flag_ai(self):
        meta = PngImagePlugin.PngInfo()
        meta.add_text("parameters", "a cat\nNegative prompt: blurry\nSteps: 20, Sampler: Euler a, CFG scale: 7")
        r = analyse(png(photo(), pnginfo=meta))
        self.assertEqual(r["verdict"], "ai")
        self.assertEqual(r["confidence"], "high")

    def test_generator_name_flags_ai(self):
        meta = PngImagePlugin.PngInfo()
        meta.add_text("Software", "Midjourney")
        self.assertEqual(analyse(png(photo(), pnginfo=meta))["verdict"], "ai")

    def test_camera_exif_means_real(self):
        ex = Image.Exif()
        ex[271], ex[272] = "Canon", "EOS 80D"
        ex[0x8769] = {33434: (1, 125), 34855: 200, 36867: "2026:03:02 10:11:12"}
        b = io.BytesIO()
        photo().save(b, "JPEG", quality=92, exif=ex)
        self.assertEqual(analyse(b.getvalue())["verdict"], "real")

    def test_plain_photo_is_inconclusive_not_ai(self):
        b = io.BytesIO()
        photo().save(b, "JPEG", quality=92)
        self.assertIn(analyse(b.getvalue())["verdict"], ("unsure", "real"))

    def test_pixel_evidence_alone_never_reaches_likely_ai(self):
        yy, xx = np.mgrid[0:1024, 0:1024]
        sm = np.stack([128 + 90 * np.sin(xx / 200 + yy / 300), 100 + 80 * np.cos(yy / 250), 150 + 70 * np.sin(xx / 180)], -1)
        r = analyse(png(Image.fromarray(np.clip(sm, 0, 255).astype(np.uint8))))
        self.assertNotEqual(r["verdict"], "ai")
        self.assertLessEqual(sum(s["weight"] for s in r["signals"] if s["group"] == "pixel"), imaging.WEAK_CAP)

    def test_periodic_artifact_detected_on_png(self):
        rng = np.random.default_rng(2)
        yy, xx = np.mgrid[0:600, 0:800]
        a = 120 + 40 * np.sin(xx / 70) * np.cos(yy / 90) + rng.normal(0, 3, (600, 800)) + 12 * np.sin(2 * np.pi * xx / 4) * np.sin(2 * np.pi * yy / 4)
        r = analyse(png(Image.fromarray(np.clip(np.stack([a] * 3, -1), 0, 255).astype(np.uint8))))
        self.assertGreater(r["info"]["spectral_peak_z"], 4.5)

    def test_screenshot_is_not_applicable(self):
        self.assertEqual(analyse(png(code_image()))["verdict"], "na")

    def test_score_in_range(self):
        for im in (photo(), code_image(), Image.new("RGB", (64, 64), "red")):
            self.assertTrue(0 <= analyse(png(im))["score"] <= 100)

    def test_tiny_image_does_not_crash(self):
        analyse(png(Image.new("RGB", (8, 8), "gray")))


@NEEDS_OCR
class Ocr(unittest.TestCase):
    def test_reads_code_and_keeps_indentation(self):
        text = imaging.run_ocr(imaging.open_image(png(code_image())))["text"]
        self.assertIn("counter = 0", text)
        self.assertTrue(any(l.startswith("    ") and "counter" in l for l in text.split("\n")), text)

    def test_dark_theme(self):
        text = imaging.run_ocr(imaging.open_image(png(code_image(dark=True))))["text"]
        self.assertIn("for i in range", text)

    def test_ocr_text_feeds_checker(self):
        text = imaging.run_ocr(imaging.open_image(png(code_image())))["text"]
        r = engine.CodeChecker().check(text)
        self.assertIn("narration", {f["rule"] for f in r["findings"]})

    def test_blank_image_gives_no_text(self):
        o = imaging.run_ocr(imaging.open_image(png(Image.new("RGB", (400, 200), "white"))))
        self.assertEqual(o["text"].strip(), "")


@NEEDS_OCR
class Http(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.httpd = ThreadingHTTPServer(("127.0.0.1", 0), app.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, body):
        r = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/image", data=body, headers={"Content-Type": "image/png"})
        try:
            with urllib.request.urlopen(r) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_code_screenshot(self):
        status, d = self.post(png(code_image()))
        self.assertEqual(status, 200)
        self.assertIsNotNone(d["code_check"])
        self.assertEqual(d["image"]["verdict"], "na")

    def test_photo_has_no_code_check(self):
        status, d = self.post(png(photo()))
        self.assertEqual(status, 200)
        self.assertIsNone(d["code_check"])
        self.assertTrue(d["notes"])

    def test_garbage_400(self):
        self.assertEqual(self.post(b"hello")[0], 400)

    def test_empty_400(self):
        self.assertEqual(self.post(b"")[0], 400)

    def test_health_reports_image_support(self):
        with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/api/health") as r:
            self.assertTrue(json.loads(r.read())["image"]["ocr"])


if __name__ == "__main__":
    unittest.main()


@NEEDS_PIL
class OcrRebuild(unittest.TestCase):
    def word(self, text, left, line=1, top=0):
        return {"line": (1, 1, line), "left": left, "top": top, "w": len(text) * 19, "h": 20, "conf": 90.0, "text": text}

    def test_split_identifier_is_glued(self):
        # Tesseract split "calculate_area(" into two boxes only 11px (0.6 of a character) apart
        words = [self.word("def", 0), self.word("calculate", 76), self.word("_area(radius):", 76 + 171 + 11)]
        self.assertEqual(imaging._rebuild_text(words), "def calculate_area(radius):")

    def test_real_space_kept(self):
        words = [self.word("for", 0), self.word("i", 57 + 22)]
        self.assertEqual(imaging._rebuild_text(words), "for i")

    def test_smart_quotes_normalised(self):
        self.assertEqual("print(“hi”)".translate(imaging._SMART), 'print("hi")')
