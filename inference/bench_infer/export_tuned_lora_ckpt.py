"""Consolidate a LoRA FSDP (DCP) training checkpoint into a full, merged pipeline dir
that create_pipeline / from_pretrained can load for inference WITHOUT any LoRA code.

Why a separate script from export_tuned_ckpt.py:
  The LoRA run saved the transformer with dcp.save(get_model_state_dict(transformer))
  AFTER add_adapter(), so the dcp `transformer/` holds base + LoRA (lora_A/lora_B)
  weights whose keys only exist once the adapter modules are attached. Loading that
  dcp into a plain base transformer (as export_tuned_ckpt.py does) would mismatch keys
  and silently drop the LoRA delta. So here we:
    1. build the base transformer,
    2. add_adapter() with the SAME r/alpha/target_modules used in training
       (read from the checkpoint's lora/default/adapter_config.json),
    3. dcp.load the tuned (base+lora) weights -> keys now match,
    4. fuse_lora() to bake the LoRA delta into the base weights,
    5. unload_lora() (a.k.a. delete adapters) so the saved module is pure base,
    6. save_pretrained() -> a normal diffusers transformer/ dir,
    7. symlink the rest of the pipeline from the base.

Usage:
  python3 export_tuned_lora_ckpt.py \
    --base_pipeline <ckpts dir> \
    --checkpoint <output_dir OR a checkpoint-N dir> \
    --out <tuned_pipeline_lora_stepN> [--device cuda]
"""
import os
import sys
import glob
import json
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
    cks = [c for c in cks if os.path.isdir(os.path.join(c, "transformer"))
           and os.path.exists(os.path.join(c, "transformer", ".metadata"))]
    if not cks:
        raise FileNotFoundError(f"no complete checkpoint-*/transformer under {output_dir}")
    return sorted(cks, key=lambda p: int(p.rsplit("-", 1)[1]))[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_pipeline", required=True, help="base ckpts dir (full pipeline layout)")
    ap.add_argument("--checkpoint", required=True,
                    help="a checkpoint-N dir, OR an output_dir to auto-pick the latest checkpoint-N")
    ap.add_argument("--out", required=True, help="output merged pipeline dir")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    # Optional overrides; by default we read them from the checkpoint's adapter_config.json.
    ap.add_argument("--lora_r", type=int, default=None)
    ap.add_argument("--lora_alpha", type=int, default=None)
    args = ap.parse_args()

    ckpt = args.checkpoint
    if not os.path.isdir(os.path.join(ckpt, "transformer")):
        ckpt = latest_checkpoint(ckpt)
    print(f"[export-lora] using checkpoint: {ckpt}", flush=True)

    # Read the LoRA config that was actually used for this checkpoint.
    adapter_cfg_path = os.path.join(ckpt, "lora", "default", "adapter_config.json")
    if not os.path.exists(adapter_cfg_path):
        raise FileNotFoundError(
            f"adapter_config.json not found at {adapter_cfg_path}; this checkpoint does "
            f"not look like a LoRA run. Use export_tuned_ckpt.py for full-tune checkpoints."
        )
    acfg = json.load(open(adapter_cfg_path))
    r = args.lora_r if args.lora_r is not None else acfg["r"]
    alpha = args.lora_alpha if args.lora_alpha is not None else acfg["lora_alpha"]
    target_modules = acfg.get("target_modules", "all-linear")
    lora_dropout = acfg.get("lora_dropout", 0.0)
    print(f"[export-lora] LoRA config: r={r} alpha={alpha} dropout={lora_dropout} "
          f"target_modules={target_modules if isinstance(target_modules, str) else f'{len(target_modules)} modules'}",
          flush=True)

    # DCP orchestration needs a (single-rank) process group.
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29577")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group("gloo")

    base_tf = os.path.join(args.base_pipeline, "transformer")
    ckpt_tf = os.path.join(ckpt, "transformer")
    out_tf = os.path.join(args.out, "transformer")

    print(f"[export-lora] loading base transformer (bf16) from {base_tf} ...", flush=True)
    tf = HunyuanVideo_1_5_DiffusionTransformer.from_pretrained(base_tf, torch_dtype=torch.bfloat16)
    tf = tf.to(args.device)

    # Attach adapter so the dcp keys (base + lora_A/lora_B) match the module structure.
    from peft import LoraConfig
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION",
    )
    print("[export-lora] add_adapter('default') ...", flush=True)
    tf.add_adapter(lora_config, adapter_name="default")

    print(f"[export-lora] dcp.load tuned (base+lora) weights from {ckpt_tf} ...", flush=True)
    sd = get_model_state_dict(tf)
    dcp.load(state_dict={"model": sd}, checkpoint_id=ckpt_tf)
    set_model_state_dict(tf, sd)

    # Bake LoRA into base weights, then strip the adapter so the saved model is pure base.
    print("[export-lora] fuse_lora() + unload ...", flush=True)
    fused = False
    if hasattr(tf, "fuse_lora"):
        tf.fuse_lora()
        fused = True
    # Remove adapter modules so save_pretrained emits a clean base-shaped state dict.
    if hasattr(tf, "unload_lora"):
        tf.unload_lora()
    elif hasattr(tf, "delete_adapters"):
        tf.delete_adapters(["default"])
    if not fused:
        raise RuntimeError("transformer has no fuse_lora(); cannot bake LoRA into base weights")

    os.makedirs(out_tf, exist_ok=True)
    print(f"[export-lora] save_pretrained -> {out_tf} ...", flush=True)
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
            print(f"[export-lora] symlink {name}", flush=True)

    print(f"[export-lora] DONE -> {args.out}", flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
