# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.
"""
Batch T2V inference for OmniWeaving over a benchmark manifest shard.

Loads the pipeline ONCE, then generates a video for every case assigned to this
shard (case index % shard_total == shard_idx). Single-process, single-GPU
(WORLD_SIZE=1 -> SP=1, uses LOCAL_RANK / CUDA_VISIBLE_DEVICES). Run 8 of these,
one per GPU, to fan out across a node.

Manifest: list of {case_id, caption, video, ...}. We read `caption` as the T2V
prompt and write the result to <dataset_dir>/<case_id>/<out_name>.

Idempotent: skips a case whose output already exists and is non-trivial.
Fault-tolerant: a failing case is logged and skipped, never kills the shard.

Env respected (set per worker):
  CUDA_VISIBLE_DEVICES  which physical GPU this process uses
"""
import os
if 'PYTORCH_CUDA_ALLOC_CONF' not in os.environ:
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import sys
import json
import time
import argparse
import traceback
from types import SimpleNamespace

import einops
import imageio
import torch

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state

# SP=1 single process; pick GPU from LOCAL_RANK (0 within the masked device set).
initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


def save_video(video, path, fps=24):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=fps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--dataset_dir', required=True,
                    help='root that <case_id>/ dirs live under (outputs written here)')
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--shard_total', type=int, default=1)
    ap.add_argument('--out_name', default='omniweaving.mp4',
                    help='per-case output filename (kept separate from source video.mp4)')
    ap.add_argument('--pipeline_config', default='omniweaving')
    ap.add_argument('--prompt_key', default='caption')
    ap.add_argument('--video_length', type=int, default=81)
    ap.add_argument('--num_inference_steps', type=int, default=50)
    ap.add_argument('--aspect_ratio', type=str, default='16:9')
    ap.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp32'])
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--fps', type=int, default=None)
    ap.add_argument('--negative_prompt', type=str, default='')
    ap.add_argument('--limit', type=int, default=0,
                    help='cap cases processed by THIS shard (0=all); for debugging')
    args = ap.parse_args()

    tag = f"[shard {args.shard_idx}/{args.shard_total}]"
    transformer_dtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32

    # ---- select this shard's cases ----
    manifest = json.load(open(args.manifest))
    cases = [c for i, c in enumerate(manifest) if i % args.shard_total == args.shard_idx]
    if args.limit and args.limit > 0:
        cases = cases[:args.limit]
    print(f"{tag} {len(cases)} cases of {len(manifest)} total", flush=True)

    # ---- load pipeline ONCE (offloading on: fits a single 80G card) ----
    t0 = time.time()
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_dtype=transformer_dtype,
        device=torch.device('cpu'),                 # offloading path
        transformer_init_device=torch.device('cpu'),
        pipeline_config=args.pipeline_config,
    )
    infer_state = initialize_infer_state(SimpleNamespace(
        sage_blocks_range="0-53", no_cache_block_id="53", use_sageattn=False,
        enable_torch_compile=False, enable_cache=False, cache_type="deepcache",
        cache_start_step=11, cache_end_step=45, total_steps=args.num_inference_steps,
        cache_step_interval=4, use_fp8_gemm=False, quant_type="fp8-per-token-sgl",
        include_patterns="double_blocks",
    ))
    offloading_config = HunyuanVideo_1_5_Pipeline.get_offloading_config()
    pipe.apply_infer_optimization(
        infer_state=infer_state,
        enable_offloading=True,
        enable_group_offloading=offloading_config['enable_group_offloading'],
        overlap_group_offloading=True,
    )
    print(f"{tag} pipeline loaded in {time.time()-t0:.0f}s", flush=True)

    fps = args.fps if args.fps is not None else (16 if args.video_length <= 81 else 24)
    done = skipped = failed = 0
    for j, c in enumerate(cases):
        cid = c['case_id']
        prompt = c.get(args.prompt_key) or ''
        out_dir = os.path.join(args.dataset_dir, cid)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, args.out_name)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 5000:
            skipped += 1
            print(f"{tag} ({j+1}/{len(cases)}) skip existing {cid}", flush=True)
            continue
        if not prompt.strip():
            failed += 1
            print(f"{tag} ({j+1}/{len(cases)}) EMPTY prompt {cid}, skip", flush=True)
            continue

        t = time.time()
        try:
            out = pipe(
                prompt=prompt,
                aspect_ratio=args.aspect_ratio,
                num_inference_steps=args.num_inference_steps,
                video_length=args.video_length,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                output_type="pt",
                task_type='t2v',
            )
            tmp = out_path + '.tmp.mp4'
            save_video(out.videos, tmp, fps=fps)
            os.replace(tmp, out_path)
            done += 1
            print(f"{tag} ({j+1}/{len(cases)}) OK {cid} {time.time()-t:.0f}s -> {out_path}", flush=True)
        except Exception as e:
            failed += 1
            print(f"{tag} ({j+1}/{len(cases)}) FAIL {cid}: {repr(e)}", flush=True)
            traceback.print_exc()
            torch.cuda.empty_cache()

    print(f"{tag} DONE done={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == '__main__':
    main()
