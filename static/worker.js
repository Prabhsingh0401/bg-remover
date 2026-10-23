// Runs the background-removal model in the browser with ONNX Runtime Web, so the page needs no server.
// Same pipeline as the old server.py: downscale to 4096 px, predict a 1024x1024 mask,
// apply it at full size, then trim / pad / max size / encode.
//
// Model: ISNet general-use (DIS), the same model as rembg's isnet-general-use. BiRefNet was tried
// first but needs ~6.6 GB at 1024 px, more than browsers give WebAssembly (4 GB), and fails on
// GPUs with 16 storage buffers per shader (e.g. Intel iGPUs).
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs";

const REPO = "https://huggingface.co/imgly/isnet-general-onnx/resolve/main/onnx/";
const SIZE = 1024;            // model input
const MAX_INPUT_SIDE = 4096;  // px, inputs are downscaled to this before inference

let sessionPromise = null;

async function gpu() {
  try {
    const adapter = await navigator.gpu?.requestAdapter();
    if (adapter) return { f16: adapter.features.has("shader-f16") };
  } catch {}
  return null;
}

// Downloads once, then serves from Cache Storage on later visits.
async function fetchModel(url, id) {
  let cache;
  try {
    cache = await caches.open("bg-remover-models");
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch {}
  const res = await fetch(url);
  if (!res.ok) throw new Error(`Model download failed (${res.status})`);
  const total = +res.headers.get("content-length") || 0;
  const reader = res.body.getReader(), parts = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    loaded += value.length;
    postMessage({ type: "progress", id, loaded, total });
  }
  const blob = new Blob(parts);
  try { await cache?.put(url, new Response(blob)); } catch {}
  return new Uint8Array(await blob.arrayBuffer());
}

async function createSession(id) {
  const g = await gpu();
  if (g) {
    try {
      const model = await fetchModel(REPO + (g.f16 ? "model_fp16.onnx" : "model.onnx"), id);
      const session = await ort.InferenceSession.create(model, { executionProviders: ["webgpu"] });
      postMessage({ type: "device", device: "webgpu" });
      return session;
    } catch (err) {
      console.warn("WebGPU failed, falling back to CPU", err);
    }
  }
  const model = await fetchModel(REPO + "model.onnx", id);
  const session = await ort.InferenceSession.create(model, { executionProviders: ["wasm"] });
  postMessage({ type: "device", device: "wasm" });
  return session;
}

function session(id) {
  sessionPromise ??= createSession(id).catch((err) => { sessionPromise = null; throw err; });
  return sessionPromise;
}

async function cutout(id, file, { trim, pad, maxSide, fmt }) {
  let bmp;
  try { bmp = await createImageBitmap(file, { imageOrientation: "from-image" }); }
  catch { throw new Error("Not a supported image"); }

  const s = Math.min(1, MAX_INPUT_SIDE / Math.max(bmp.width, bmp.height));
  const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
  const src = new OffscreenCanvas(w, h);
  const ctx = src.getContext("2d", { willReadFrequently: true });
  ctx.imageSmoothingQuality = "high";
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();

  const sess = await session(id);
  postMessage({ type: "status", id, text: "Removing…" });

  // Model input: 1024x1024 RGB, NCHW, (v - 128) / 256.
  const ic = new OffscreenCanvas(SIZE, SIZE);
  const ictx = ic.getContext("2d", { willReadFrequently: true });
  ictx.imageSmoothingQuality = "high";
  ictx.drawImage(src, 0, 0, SIZE, SIZE);
  const px = ictx.getImageData(0, 0, SIZE, SIZE).data, N = SIZE * SIZE;
  const x = new Float32Array(3 * N);
  for (let i = 0; i < N; i++) {
    x[i] = (px[i * 4] - 128) / 256;
    x[N + i] = (px[i * 4 + 1] - 128) / 256;
    x[2 * N + i] = (px[i * 4 + 2] - 128) / 256;
  }
  const out = await sess.run({ [sess.inputNames[0]]: new ort.Tensor("float32", x, [1, 3, SIZE, SIZE]) });
  const pred = out[sess.outputNames[0]];

  // Prediction -> alpha mask (min-max normalised like rembg), then let the canvas scale it onto the image.
  const [mh, mw] = pred.dims.slice(-2), raw = pred.data;
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < mw * mh; i++) { if (raw[i] < lo) lo = raw[i]; if (raw[i] > hi) hi = raw[i]; }
  const range = hi - lo || 1;
  const mask = new ImageData(mw, mh);
  for (let i = 0; i < mw * mh; i++) mask.data[i * 4 + 3] = (255 * (raw[i] - lo)) / range;
  const mc = new OffscreenCanvas(mw, mh);
  mc.getContext("2d").putImageData(mask, 0, 0);
  ctx.globalCompositeOperation = "destination-in";
  ctx.drawImage(mc, 0, 0, w, h);
  ctx.globalCompositeOperation = "source-over";

  let sx = 0, sy = 0, cw = w, ch = h, p = 0;
  if (trim) {
    const a = ctx.getImageData(0, 0, w, h).data;
    let x0 = w, y0 = h, x1 = -1, y1 = -1;
    for (let yy = 0; yy < h; yy++) for (let xx = 0; xx < w; xx++) {
      if (a[(yy * w + xx) * 4 + 3] > 8) {
        if (xx < x0) x0 = xx; if (xx > x1) x1 = xx;
        if (yy < y0) y0 = yy; if (yy > y1) y1 = yy;
      }
    }
    if (x1 >= 0) { sx = x0; sy = y0; cw = x1 - x0 + 1; ch = y1 - y0 + 1; }
    p = Math.trunc(Math.max(cw, ch) * pad);
  }
  const ow = cw + 2 * p, oh = ch + 2 * p;
  const k = maxSide && Math.max(ow, oh) > maxSide ? maxSide / Math.max(ow, oh) : 1;

  const canvas = new OffscreenCanvas(Math.max(1, Math.round(ow * k)), Math.max(1, Math.round(oh * k)));
  const octx = canvas.getContext("2d");
  octx.imageSmoothingQuality = "high";
  octx.drawImage(src, sx, sy, cw, ch, p * k, p * k, cw * k, ch * k);
  return canvas.convertToBlob({ type: `image/${fmt}`, quality: 0.9 });   // Safari falls back to PNG for WebP
}

onmessage = async ({ data: { id, file, opts } }) => {
  try {
    const blob = await cutout(id, file, opts);
    postMessage({ type: "done", id, blob });
  } catch (err) {
    console.error(err);
    postMessage({ type: "error", id, message: err?.message || String(err) });
  }
};
