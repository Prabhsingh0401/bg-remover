"""
Background-removal web service: same BiRefNet pipeline as Singh-Seals/scripts/remove-bg.py,
served over HTTP with a small drag-and-drop frontend in static/.

Run locally:
    python -m venv .venv
    .venv/Scripts/python -m pip install -r requirements.txt
    .venv/Scripts/python -m uvicorn server:app --port 8000

Env:
    BG_MODEL      rembg model name            (default birefnet-general)
    BG_WORKERS    images processed at once    (default 1 - each inference already uses every core)
    BG_QUEUE      max requests waiting        (default 8, beyond that -> 503)
    U2NET_HOME    where rembg caches models   (default ~/.u2net)
"""

import asyncio
import os
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from PIL import Image, ImageOps, UnidentifiedImageError
from rembg import new_session, remove

MODEL = os.getenv("BG_MODEL", "birefnet-general")
WORKERS = int(os.getenv("BG_WORKERS", "1"))
QUEUE = int(os.getenv("BG_QUEUE", "8"))
MAX_UPLOAD = 25 * 1024 * 1024  # bytes
MAX_INPUT_SIDE = 4096          # px, inputs are downscaled to this before inference
Image.MAX_IMAGE_PIXELS = 80_000_000  # refuse decompression bombs

STATIC = Path(__file__).resolve().parent / "static"

state = {"session": None, "waiting": 0}
slots = asyncio.Semaphore(WORKERS)


@asynccontextmanager
async def lifespan(app):
    # Loads (and on first run downloads, ~1 GB) the model before accepting requests.
    state["session"] = await run_in_threadpool(new_session, MODEL)
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)


def process(data, trim, pad, max_side, fmt):
    with Image.open(BytesIO(data)) as im:
        im = ImageOps.exif_transpose(im)  # phone photos carry rotation in EXIF
        im = im.convert("RGB")
    im.thumbnail((MAX_INPUT_SIDE, MAX_INPUT_SIDE), Image.LANCZOS)

    img = remove(im, session=state["session"], post_process_mask=True).convert("RGBA")

    if trim:
        bbox = img.getchannel("A").point(lambda a: 255 if a > 8 else 0).getbbox()
        if bbox:
            img = img.crop(bbox)
        p = int(max(img.size) * pad)
        canvas = Image.new("RGBA", (img.width + 2 * p, img.height + 2 * p), (0, 0, 0, 0))
        canvas.paste(img, (p, p))
        img = canvas
    if max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)

    buf = BytesIO()
    if fmt == "webp":
        img.save(buf, "WEBP", quality=90, method=6)
    else:
        img.save(buf, "PNG", optimize=True)
    return buf.getvalue()


@app.get("/api/health")
async def health():
    return {"ok": state["session"] is not None, "model": MODEL, "waiting": state["waiting"]}


@app.post("/api/remove")
async def remove_bg(
    file: UploadFile = File(...),
    trim: bool = Form(True),
    pad: float = Form(0.04),
    max_side: int = Form(0),
    fmt: str = Form("png"),
):
    if fmt not in ("png", "webp"):
        raise HTTPException(400, "fmt must be png or webp")
    pad = min(max(pad, 0.0), 0.5)
    max_side = min(max(max_side, 0), 8192)

    data = await file.read(MAX_UPLOAD + 1)
    if len(data) > MAX_UPLOAD:
        raise HTTPException(413, "Image is larger than 25 MB")

    if state["waiting"] >= QUEUE:
        raise HTTPException(503, "Server is busy, try again in a moment")
    state["waiting"] += 1
    try:
        async with slots:
            out = await run_in_threadpool(process, data, trim, pad, max_side, fmt)
    except (UnidentifiedImageError, Image.DecompressionBombError):
        raise HTTPException(400, "Not a supported image")
    finally:
        state["waiting"] -= 1

    return Response(out, media_type=f"image/{fmt}", headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")
