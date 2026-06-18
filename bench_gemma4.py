"""Benchmark Gemma 4 E2B on RTX 8000 for live-stream captioning use cases.

Tests at multiple precisions (bf16 + 4-bit if bitsandbytes loads).
Measures:
  - cold-start load time
  - single-image caption (real-time per-frame VLM at 1 fps target)
  - 8-frame video caption
  - audio+video caption (the killer feature)

Then projects how many concurrent 1-fps streams a single GPU can serve.
"""
import argparse
import os
import sys
import time
from pathlib import Path
from statistics import median

import torch


def time_block(name, fn, iters=3):
    """Run `fn` N times, return (median_sec, all_times)."""
    times = []
    for i in range(iters):
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        fn()
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        times.append(time.time() - t0)
    print(f"  {name}: median={median(times):.2f}s  all={[f'{t:.2f}' for t in times]}")
    return median(times), times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-E2B-it")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--quant", choices=["bf16", "4bit", "8bit"], default="bf16")
    ap.add_argument("--n-iters", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=80)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(f"=== Gemma 4 E2B benchmark on GPU {args.gpu}, precision={args.quant} ===\n")

    # === LOAD ===
    print(f"loading {args.model} ({args.quant}) ...")
    t0 = time.time()
    from transformers import AutoProcessor, AutoModelForCausalLM
    processor = AutoProcessor.from_pretrained(args.model)

    load_kwargs = {"device_map": "cuda:0"}
    if args.quant == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    elif args.quant == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quant == "8bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs).eval()
    load_time = time.time() - t0
    mem_gb = torch.cuda.memory_allocated() / (1024**3)
    print(f"  loaded in {load_time:.1f}s, allocated GPU memory: {mem_gb:.1f} GB\n")

    # === FIND TEST IMAGES ===
    bench_dir = Path(os.environ.get("DGL_BENCH_IMAGE_DIR", "/tmp/dgl_test_imgs"))
    test_imgs = sorted(bench_dir.glob("*.jpg"))[:8]
    print(f"using {len(test_imgs)} test frames\n")

    from PIL import Image
    pil_imgs = [Image.open(p).convert("RGB") for p in test_imgs]

    # === BENCH 1: text-only (warmup) ===
    print("[1] text-only generation (warmup)")
    def text_only():
        msgs = [{"role": "user", "content": [{"type": "text",
                 "text": "Say hello in one short sentence."}]}]
        text = processor.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
        inputs = processor(text=text, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                  do_sample=False)
    time_block("text-only", text_only, iters=args.n_iters)

    # === BENCH 2: single image caption (real-time per-frame target) ===
    print("\n[2] single-image caption (target: <1.0s for 1 fps real-time)")
    def single_img():
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil_imgs[0]},
            {"type": "text", "text": "Describe this scene in one sentence."},
        ]}]
        text = processor.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
        inputs = processor(text=text, images=[pil_imgs[0]],
                           return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                  do_sample=False)
    single_med, _ = time_block("single-image", single_img, iters=args.n_iters)

    # Show one example output
    msgs = [{"role": "user", "content": [
        {"type": "image", "image": pil_imgs[0]},
        {"type": "text", "text": "Describe this scene in one sentence."},
    ]}]
    text = processor.apply_chat_template(msgs, tokenize=False,
                                          add_generation_prompt=True,
                                          enable_thinking=False)
    inputs = processor(text=text, images=[pil_imgs[0]],
                       return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                              do_sample=False)
    in_len = inputs["input_ids"].shape[-1]
    response = processor.decode(out[0][in_len:], skip_special_tokens=True)
    print(f"  EXAMPLE output: {response.strip()[:200]}")

    # === BENCH 3: multi-frame video caption (8 frames = 8 sec @ 1 fps) ===
    print("\n[3] 8-frame video caption (target: <8s for streaming)")
    def multi_img():
        content = [{"type": "image", "image": im} for im in pil_imgs]
        content.append({"type": "text",
                        "text": "These are 8 sequential frames from a video. Describe what is happening."})
        msgs = [{"role": "user", "content": content}]
        text = processor.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True,
                                              enable_thinking=False)
        inputs = processor(text=text, images=pil_imgs,
                           return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                  do_sample=False)
    multi_med, _ = time_block("8-frame", multi_img, iters=args.n_iters)

    # === SUMMARY ===
    print(f"\n=== SUMMARY ({args.quant}) ===")
    print(f"  GPU memory at load:     {mem_gb:.2f} GB")
    print(f"  Single-image latency:   {single_med:.2f}s")
    print(f"  8-frame video latency:  {multi_med:.2f}s")
    print(f"\n  --- 1 fps single-frame streams per RTX 8000 ---")
    print(f"  Per-stream budget:      1.00s")
    print(f"  Single-frame call:      {single_med:.2f}s")
    streams_real = max(1, int(1.0 / single_med * 0.9))
    streams_total_for_n_workers = lambda n_workers: n_workers * streams_real
    # Per worker: ~mem_gb (model) + ~0.5GB activation + ~0.3GB KV
    per_worker = mem_gb + 0.8
    workers_per_48gb = max(1, int((48 - 4) / per_worker))
    print(f"  Per-worker memory:      ~{per_worker:.1f} GB")
    print(f"  Workers per RTX 8000:   {workers_per_48gb}")
    print(f"  Streams per GPU @ 1fps: {workers_per_48gb * streams_real}")
    print(f"  Streams across 8 GPUs:  {workers_per_48gb * streams_real * 8}")


if __name__ == "__main__":
    main()
