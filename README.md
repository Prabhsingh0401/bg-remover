# Background Remover

A static web app that removes image backgrounds **in the browser**. The model runs on the visitor's GPU (WebGPU), or on the CPU (WebAssembly) when WebGPU isn't available. Images never leave the device, and no server is needed.

- `static/index.html`: drag/drop/paste UI, batch queue, before/after peek, ZIP download.
- `static/worker.js`: inference in a Web Worker with [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/). The model is **ISNet general-use** (same as rembg's `isnet-general-use`), loaded from [`imgly/isnet-general-onnx`](https://huggingface.co/imgly/isnet-general-onnx) on Hugging Face: fp16 (88 MB) on GPUs that support it, fp32 (176 MB) otherwise. It downloads on the first image and is then kept in Cache Storage.
- Pipeline is the same as before: downscale to 4096 px, 1024×1024 mask, trim to subject, pad, optional max size, PNG/WebP.

Speed measured on a laptop with an Intel integrated GPU (gen 12): ~1 s per image on WebGPU, ~12 s on CPU (single thread). On Vercel the COOP/COEP headers enable multithreaded WASM, so CPU runs faster.

Why not BiRefNet any more: in the browser it needs about 6.6 GB of memory at 1024 px, more than WebAssembly can address (4 GB). Its WebGPU path also fails on GPUs that allow only 16 storage buffers per shader, such as Intel iGPUs. For BiRefNet quality, run it on a server GPU instead (e.g. fal.ai or Replicate behind a Vercel function).
