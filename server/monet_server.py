# Monetworking diffusion server.
#
# Streams camera frames from the web app through an image-to-image diffusion
# pipeline so the scene is *reinterpreted* — not just repainted — as a Monet:
# structure preserved, materials replaced (the "car hidden as a forest" trick).
#
#   browser ──JPEG frame──▶ websocket ──▶ diffusion pipeline ──▶ JPEG ──▶ browser
#
# Modes (pick per available hardware):
#   --mock          no ML at all; cheap posterize filter. Validates the loop.
#   (default)       stabilityai/sd-turbo img2img, 1-3 steps. Fast: structure
#                   survives via low-ish strength. ~15-25 fps on a decent GPU.
#   --controlnet    SD1.5 + LCM-LoRA + ControlNet-depth. The full camouflage
#                   recipe: a depth map locks the silhouette so strength can go
#                   high (0.8+) and materials fully dissolve into the prompt.
#                   ~5-10 fps on a decent GPU.
#
# Protocol: client sends JSON {"type":"settings", prompt, strength, structure}
# and binary JPEG frames; server answers every frame with a binary JPEG.
# Serial processing + client-side backpressure = newest frame always wins.

import argparse
import asyncio
import io
import json
import math
import time

from PIL import Image, ImageFilter

try:
    import websockets
except ImportError:
    raise SystemExit("pip install websockets pillow  (see server/requirements.txt)")

SETTINGS = {
    "prompt": "an impressionist oil painting of a water lily pond by Claude Monet, "
              "soft morning light, thick impasto brushstrokes, pastel palette",
    "negative": "photo, photorealistic, sharp focus, text, watermark, frame",
    "strength": 0.55,
    "structure": 0.6,
}


def parse_args():
    p = argparse.ArgumentParser(description="Monetworking diffusion server")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--size", type=int, default=512, help="long edge of working frames")
    p.add_argument("--mock", action="store_true", help="no ML; posterize filter to test the loop")
    p.add_argument("--controlnet", action="store_true",
                   help="SD1.5 + LCM + ControlNet-depth (strong structure lock)")
    p.add_argument("--model", default=None, help="override base model id")
    p.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    return p.parse_args()


class Painter:
    """Wraps whichever backend is active behind a synchronous paint(image)."""

    def __init__(self, args):
        self.args = args
        self.kind = "mock" if args.mock else ("controlnet" if args.controlnet else "turbo")
        self.device = "cpu"
        if not args.mock:
            self._load()

    def _load(self):
        import torch
        from diffusers import AutoPipelineForImage2Image

        self.torch = torch
        self.device = self.args.device or (
            "cuda" if torch.cuda.is_available()
            else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
            else "cpu")
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"[server] loading {self.kind} pipeline on {self.device} ({dtype}) …")

        if self.kind == "turbo":
            model = self.args.model or "stabilityai/sd-turbo"
            self.pipe = AutoPipelineForImage2Image.from_pretrained(
                model, torch_dtype=dtype, safety_checker=None)
        else:
            from diffusers import (ControlNetModel, LCMScheduler,
                                   StableDiffusionControlNetImg2ImgPipeline)
            from transformers import pipeline as hf_pipeline

            model = self.args.model or "Lykon/dreamshaper-8"
            controlnet = ControlNetModel.from_pretrained(
                "lllyasviel/control_v11f1p_sd15_depth", torch_dtype=dtype)
            self.pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                model, controlnet=controlnet, torch_dtype=dtype, safety_checker=None)
            self.pipe.scheduler = LCMScheduler.from_config(self.pipe.scheduler.config)
            self.pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
            self.pipe.fuse_lora()
            self.depth = hf_pipeline(
                "depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf",
                device=0 if self.device == "cuda" else -1)

        self.pipe = self.pipe.to(self.device)
        if self.device == "cuda":
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        # fixed seed: per-frame noise stays identical, which tames flicker
        self.generator = self.torch.Generator(self.device).manual_seed(7)
        print("[server] pipeline ready")

    def _fit(self, img):
        long_edge = self.args.size
        w, h = img.size
        s = long_edge / max(w, h)
        w, h = max(8, int(w * s) // 8 * 8), max(8, int(h * s) // 8 * 8)
        return img.resize((w, h), Image.LANCZOS)

    def paint(self, jpeg_bytes, settings):
        img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
        img = self._fit(img)

        if self.kind == "mock":
            out = (img.quantize(16, dither=Image.Dither.NONE).convert("RGB")
                      .filter(ImageFilter.ModeFilter(7))
                      .filter(ImageFilter.GaussianBlur(0.6)))
        elif self.kind == "turbo":
            strength = min(max(float(settings["strength"]), 0.15), 0.95)
            steps = max(2, math.ceil(1.0 / strength))   # diffusers needs steps*strength ≥ 1
            out = self.pipe(
                prompt=settings["prompt"], negative_prompt=settings["negative"],
                image=img, strength=strength, num_inference_steps=steps,
                guidance_scale=0.0, generator=self.generator).images[0]
        else:
            strength = min(max(float(settings["strength"]), 0.3), 0.98)
            depth = self.depth(img)["depth"].convert("RGB").resize(img.size)
            out = self.pipe(
                prompt=settings["prompt"], negative_prompt=settings["negative"],
                image=img, control_image=depth,
                controlnet_conditioning_scale=float(settings["structure"]) * 1.2,
                strength=strength, num_inference_steps=5,
                guidance_scale=1.3, generator=self.generator).images[0]

        buf = io.BytesIO()
        out.save(buf, "JPEG", quality=87)
        return buf.getvalue()


async def main():
    args = parse_args()
    painter = Painter(args)
    loop = asyncio.get_running_loop()
    lock = asyncio.Lock()   # one generation at a time, even with several clients

    async def handle(ws):
        peer = ws.remote_address
        print(f"[server] client connected: {peer}")
        settings = dict(SETTINGS)
        await ws.send(json.dumps({
            "type": "hello", "backend": painter.kind, "device": painter.device,
            "size": args.size,
        }))
        try:
            async for msg in ws:
                if isinstance(msg, (bytes, bytearray)):
                    t0 = time.time()
                    async with lock:
                        out = await loop.run_in_executor(None, painter.paint, bytes(msg), settings)
                    await ws.send(out)
                    print(f"[server] frame {len(msg)/1e3:.0f}kB → {len(out)/1e3:.0f}kB "
                          f"in {time.time()-t0:.2f}s", end="\r")
                else:
                    data = json.loads(msg)
                    if data.get("type") == "settings":
                        for k in ("prompt", "negative", "strength", "structure"):
                            if k in data:
                                settings[k] = data[k]
        except websockets.ConnectionClosed:
            pass
        print(f"\n[server] client left: {peer}")

    async with websockets.serve(handle, "127.0.0.1", args.port, max_size=8 * 1024 * 1024):
        print(f"[server] {painter.kind} backend on ws://127.0.0.1:{args.port} "
              f"(device: {painter.device})")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
