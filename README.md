# AI Code Checker - spotting AI-style code with pattern matching

A web app (same stdlib-only architecture as the AI Fact Checker) that scans source code for the stylistic
fingerprints AI assistants leave behind, scores it 0-100, highlights the lines, and gives back a cleaned copy
with the redundant comments removed.

## Run it (Python 3.8+; the text checker needs no packages and no internet)

```bash
python app.py                       # http://127.0.0.1:8000
python app.py --port 5000
python check.py yourfile.py         # command-line mode
python -m unittest discover -s tests
python evaluate.py
```
Hosting (Render, Railway...): start command `python app.py`, set `HOST=0.0.0.0`; `PORT` is read automatically.

## Image features (OCR + AI-image check) - optional extras

```bash
pip install -r requirements.txt     # pillow, numpy
# plus the Tesseract program, a separate install that pip cannot provide (Windows: UB-Mannheim installer, macOS: brew install tesseract, Linux: apt install tesseract-ocr)
python app.py                       # console prints whether image upload and OCR are on
```
The app also finds Tesseract in its default Windows/macOS folders, or at the path in the `TESSERACT_CMD` environment variable. After installing it, restart `python app.py`.

Open the page and choose **Upload image** (drop a file, click, or paste a screenshot with Ctrl+V). Without the extras the text checker still works and
the image endpoint answers with an install message.

* **OCR** (`imaging.py`): Tesseract reads the picture; word boxes from its TSV output rebuild indentation, spacing and blank lines (important for code),
  and the text goes to the same code checker. Dark themes are inverted automatically, small images upscaled.
* **AI-image check**: explainable signals, each with a weight, starting from a neutral 30:
  generator metadata (SD "parameters", Midjourney/DALL-E names, +45 to +55), C2PA provenance (+12), camera EXIF (-12/-25),
  typical generator sizes (+5/+8), very low sensor noise (+6) and periodic frequency peaks from upsampling (+8/+12, lossless formats only).
  Pixel evidence is capped at +30, so without metadata the result can never exceed "Possibly AI-generated".
  Screenshots and flat graphics are reported "not applicable". Optional: `AI_IMAGE_MODEL=umm-maybe/AI-image-detector` adds a pretrained
  classifier (needs transformers + torch + internet on first run).

**Honest status of the image features**
* OCR was tested on screenshots rendered by us (light and dark, monospace). Photos of screens, handwriting and unusual fonts will be worse.
  OCR drops or garbles characters such as `"""` and `_`, so the code checker may report "did not parse"; the UI lets you correct the text and re-check.
* The detector was tested on synthetic images, on files carrying generator metadata/EXIF that we created, and on 5 real photos (no false alarms on the
  frequency test, but 3 of 5 trip the low-noise test, which is why that signal is weighted low). **We had no real AI-generated images to test on.**
* The pretrained-model hook is untested in our environment (no access to Hugging Face from the build sandbox).
* No image detector is reliable. Metadata can be stripped or forged; new generators leave different artefacts.

## How it works

| Step | Where | What it does |
|---|---|---|
| 1. Input | `static/index.html` | Paste code, pick language or auto-detect |
| 2. Preprocessing | `engine.py` `scan`, `detect_language`, `python_info` | Splits comments from code (strings respected), finds docstrings and functions with `ast` |
| 3. Pattern engine | `engine.py` comment rules + `data/patterns.json` | Narration, step comments, banners, assistant phrasing, placeholders, density |
| 4. Code patterns | `engine.py` code and docstring rules | Generic names, catch-and-print, assign-then-return, else-after-return, `== True`, status prints, template docstrings, emoji |
| 5. Scoring | `engine.py` `check` | Weighted rules with diminishing returns, normalised by file size, minus human-context signals |
| 6. Recovery | `engine.py` `cleaned` | Removes only flagged redundant comments; code is never touched |
| 7. Output | browser UI | Score, label, annotated code, findings with fixes, cleaned copy |

Labels: **Likely AI-generated style** (score >= 55), **Mixed - worth reviewing** (28-54), **Likely human-written style** (< 28).
Comments that explain *why* (because, workaround, TODO, upstream...) count as human signals and lower the score.

## Honest limitations
* It detects **style, not authorship**. Beginners, tutorials and "comment every line" teaching produce the same patterns; edited AI code loses them.
* Clean, uncommented AI code will score low. The tool mostly catches *unedited* assistant output.
* Rules are hand-tuned. The 16-snippet test set in `data/testset.json` was written by the author alongside the rules, so the result below shows the rules work as designed, not that they generalise.
* Measured on that set: AI 7/7, human 7/7, **mixed 0/2** (14/16 overall). The mixed band is the weakest part.
* Python gets deeper checks (AST docstrings, type hints); other languages use the comment and regex rules only.
* Input limit 30,000 characters. Never use the score as sole evidence of misconduct.

## Linux and container concepts in this project

| Term | Where it is in this project |
|---|---|
| **Process and port** | `python app.py` starts one process that listens on a TCP port (8000). `HOST=127.0.0.1` accepts only this computer; `0.0.0.0` accepts every network interface (needed inside a container). |
| **Environment variables** | `HOST`, `PORT` (server), `TESSERACT_CMD` (OCR location), `AI_IMAGE_MODEL` (optional model), `OMP_THREAD_LIMIT` (set for the OCR child process). |
| **PATH** | `shutil.which("tesseract")` searches the folders listed in the `PATH` variable. When the Windows installer does not add its folder to PATH, the program is "not found", which is why we also check the default folders. |
| **File paths** | `pathlib` builds paths from `BASE_DIR` (the folder of `app.py`) so the app works from any directory. OCR writes the image to a temporary directory (`/tmp` on Linux) that is deleted afterwards. |
| **Child process (`subprocess`)** | The app runs `tesseract` as a separate program, waits at most 60 s (`timeout`), reads its `stdout`, and checks its exit code. |
| **stdin / stdout / pipes** | `check.py -` reads code from standard input, so `cat file.py | python check.py -` works. Tesseract writes its result to `stdout`, which we capture. |
| **Exit codes** | `check.py` exits with code 2 on wrong usage (`SystemExit(2)`); 0 means success. The HTTP server's 400/413/503 codes play the same role for web requests. |
| **Signals** | Ctrl+C sends SIGINT and `docker stop` sends SIGTERM. `app.py` handles both, so the server closes its socket and exits cleanly instead of being force-killed (SIGKILL) after 10 s. |
| **Procfile** | One line (`web: python app.py`) telling hosting platforms which process to start. |

### Docker (files: `Dockerfile`, `.dockerignore`, `compose.yaml`)
* **Image**: a read-only template built from the `Dockerfile`: Python + the Tesseract program + our code. Each instruction is a cached **layer**; `requirements.txt` is copied before the code so editing code does not reinstall packages.
* **Container**: a running instance of an image. It is an ordinary Linux process isolated with *namespaces* (own view of processes, network, filesystem) and limited with *cgroups* (memory, CPU). Unlike a virtual machine, it shares the host's kernel, so it starts in seconds.
* **Port mapping** `-p 8000:8000` connects host port 8000 to container port 8000.
* **Why we use it**: Tesseract is not a pip package, so installs differ per computer. The image contains it, so the app runs identically on Windows, Mac and Linux.
* **Hardening in `compose.yaml`**: runs as a non-root user, read-only filesystem with `/tmp` in RAM (`tmpfs`), 512 MB memory limit, all Linux capabilities dropped, `no-new-privileges`. `HEALTHCHECK` marks the container healthy only while `/api/health` answers.

```bash
docker build -t ai-code-checker .
docker run --rm -p 8000:8000 ai-code-checker      # then open http://localhost:8000
docker compose up --build                          # same, with the hardening options
```
Status: the Dockerfile was written but **not built** in our test environment (no Docker engine). Compose syntax, the health-check command and a non-root run of the app were checked separately. The first `docker build` on your machine is the real test.

## Extending it
Add phrases or placeholder regexes in `data/patterns.json`. Add a rule by registering it in `RULES` (title, category, weight, description)
and calling `add("rule_id", line)` inside `CodeChecker.check`. Re-run the tests and `evaluate.py`.

## Future work
Token-level perplexity features, a larger labelled corpus of real human vs AI files, per-language rule sets, a VS Code extension,
git-history signals (one huge commit of uniformly styled code).
