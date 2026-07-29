"""Consolidate an FSDP (DCP) training checkpoint into a full pipeline dir that
create_pipeline / from_pretrained can load for inference.

train.py saves the transformer with:
    dcp.save(state_dict={"model": get_model_state_dict(transformer)}, checkpoint_id=<ckpt>/transformer)
i.e. a sharded torch.distributed.checkpoint (the __*.distcp files). That format
is NOT loadable by HunyuanVideo_1_5_Pipeline.create_pipeline, which expects a
diffusers-style transformer/ dir (config.json + diffusion_pytorch_model.safetensors).

This script (single process) rebuilds the base transformer, dcp.load's the tuned
weights into it, save_pretrained's it, and symlinks the rest of the pipeline
(vae/text_encoder/scheduler/...) from the base so inference gets a complete dir.
"""
import os
import sys
import glob
import argparse

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    set_model_state_dict,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hyvideo.models.transformers.hunyuanvideo_1_5_transformer import (
    HunyuanVideo_1_5_DiffusionTransformer,
)


def latest_checkpoint(output_dir: str) -> str:
    cks = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    cks = [c for c in cks if os.path.isdir(os.path.join(c, "transformer"))]
    if not cks:
        raise FileNotFoundError(f"no checkpoint-*/transformer under {output_dir}")
    return sorted(cks, key=lambda p: int(p.rsplit("-", 1)[1]))[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_pipeline", required=True, help="base ckpts dir (full pipeline layout)")
    ap.add_argument("--checkpoint", required=True,
                    help="a checkpoint-N dir, OR an output_dir to auto-pick the latest checkpoint-N")
    ap.add_argument("--out", required=True, help="output tuned pipeline dir")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = ap.parse_args()

    ckpt = args.checkpoint
    if not os.path.isdir(os.path.join(ckpt, "transformer")):
        ckpt = latest_checkpoint(ckpt)
    print(f"[export] using checkpoint: {ckpt}", flush=True)

    # DCP orchestration needs a (single-rank) process group.
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29566")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")

    base_tf = os.path.join(args.base_pipeline, "transformer")
    ckpt_tf = os.path.join(ckpt, "transformer")
    out_tf = os.path.join(args.out, "transformer")

    print(f"[export] loading base transformer (bf16) from {base_tf} ...", flush=True)
    tf = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(base_tf, torch_dtype=torch.bfloat16)
    tf = tf.to(args.device)

    print(f"[export] dcp.load tuned weights from {ckpt_tf} ...", flush=True)
    sd = get_model_state_dict(tf)
    dcp.load(state_dict={"model": sd}, checkpoint_id=ckpt_tf)
    set_model_state_dict(tf, sd)

    os.makedirs(out_tf, exist_ok=True)
    print(f"[export] save_pretrained -> {out_tf} ...", flush=True)
    tf.save_pretrained(out_tf, safe_serialization=True)

    # Symlink the rest of the pipeline (vae / text_encoder / scheduler / ...).
    os.makedirs(args.out, exist_ok=True)
    for name in sorted(os.listdir(args.base_pipeline)):
        if name == "transformer":
            continue
        src = os.path.join(args.base_pipeline, name)
        dst = os.path.join(args.out, name)
        if not os.path.exists(dst):
            os.symlink(src, dst)
            print(f"[export] symlink {name}", flush=True)

    print(f"[export] DONE -> {args.out}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
