# Licensed under the TENCENT HUNYUAN COMMUNITY LICENSE AGREEMENT.
"""
Batch VIDEO-EDITING inference for OmniWeaving over a benchmark manifest shard.

Second baseline: instead of text->video (batch_infer.py), this uses OmniWeaving's
`editing` task_type. For each case it takes the grayscale clay ref (video.mp4) as
the condition video, guides the edit with that case's edit_prompt.txt (the same
restyle prompt fed to Seedance), and writes the result to
<dataset_dir>/<case_id>/<out_name>.

Mirrors generate.py's editing path (encode_video_to_latents -> pipe(task_type=
'editing', condition_video_latents=..., condition_videos=[path])), but loops a
manifest shard, loads the pipeline once, and is idempotent + fault-tolerant like
batch_infer.py.

Env respected: CUDA_VISIBLE_DEVICES (which physical GPU), LOCAL_RANK, WORLD_SIZE.
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

import numpy as np
import einops
import imageio
import torch
from torchvision import transforms

from hyvideo.pipelines.hunyuan_video_pipeline import HunyuanVideo_1_5_Pipeline
from hyvideo.commons import auto_offload_model
from hyvideo.utils.data_utils import generate_crop_size_list
from hyvideo.commons.parallel_states import initialize_parallel_state
from hyvideo.commons.infer_state import initialize_infer_state

initialize_parallel_state(sp=int(os.environ.get('WORLD_SIZE', '1')))
torch.cuda.set_device(int(os.environ.get('LOCAL_RANK', '0')))


def save_video(video, path, fps=24):
    if video.ndim == 5:
        assert video.shape[0] == 1
        video = video[0]
    vid = (video * 255).clamp(0, 255).to(torch.uint8)
    vid = einops.rearrange(vid, 'c f h w -> f h w c')
    imageio.mimwrite(path, vid, fps=fps)


def encode_video_to_latents(pipe, video_path, max_frames=None):
    """Encode a video into VAE latents — identical flow to generate.py."""
    from decord import VideoReader
    vr = VideoReader(video_path)
    ori_len = len(vr)
    if max_frames is not None:
        ori_len = min(ori_len, max_frames)
    if ori_len < 33:
        raise ValueError(f"condition video too short: {video_path} has {ori_len} frames (<33)")
    elif ori_len > 161:
        tgt_len = 161
    else:
        tgt_len = (ori_len - 1) // 8 * 8 + 1
    imgs = vr.get_batch(list(range(tgt_len)))
    px = torch.from_numpy(imgs.asnumpy()).permute(0, 3, 1, 2).contiguous() / 255.
    crop_size_list = generate_crop_size_list(base_size=640)
    ratios = np.array([round(float(h) / float(w), 5) for h, w in crop_size_list])
    H, W = px.size(-2), px.size(-1)
    cid = np.abs(ratios - float(H) / float(W)).argmin()
    cs = crop_size_list[cid]
    if cs[0] / H > cs[1] / W:
        rs = cs[0], int(W * cs[0] / H)
    else:
        rs = int(H * cs[1] / W), cs[1]
    tf = transforms.Compose([
        transforms.Resize(rs, antialias=True),
        transforms.CenterCrop(cs),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ])
    px = tf(px).unsqueeze(0).transpose(1, 2).to(pipe.execution_device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16, enabled=True), \
         auto_offload_model(pipe.vae, pipe.execution_device, enabled=pipe.enable_offloading), \
         pipe.vae.memory_efficient_context():
        latents = pipe.vae.encode(px).latent_dist.mode()
    return latents, tgt_len


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model_path', required=True)
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--shard_idx', type=int, default=0)
    ap.add_argument('--shard_total', type=int, default=1)
    ap.add_argument('--out_name', default='omniweaving_edit.mp4')
    ap.add_argument('--cond_name', default='video.mp4',
                    help='per-case condition (clay ref) filename')
    ap.add_argument('--prompt_file', default='edit_prompt.txt',
                    help='per-case restyle prompt filename (guides the edit)')
    ap.add_argument('--pipeline_config', default='omniweaving')
    ap.add_argument('--video_length', type=int, default=81)
    ap.add_argument('--num_inference_steps', type=int, default=50)
    ap.add_argument('--aspect_ratio', type=str, default='16:9')
    ap.add_argument('--dtype', type=str, default='bf16', choices=['bf16', 'fp32'])
    ap.add_argument('--seed', type=int, default=123)
    ap.add_argument('--negative_prompt', type=str, default='')
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    tag = f"[edit shard {args.shard_idx}/{args.shard_total}]"
    tdtype = torch.bfloat16 if args.dtype == 'bf16' else torch.float32

    manifest = json.load(open(args.manifest))
    cases = [c for i, c in enumerate(manifest) if i % args.shard_total == args.shard_idx]
    if args.limit and args.limit > 0:
        cases = cases[:args.limit]
    print(f"{tag} {len(cases)} cases of {len(manifest)} total", flush=True)

    t0 = time.time()
    pipe = HunyuanVideo_1_5_Pipeline.create_pipeline(
        pretrained_model_name_or_path=args.model_path,
        transformer_dtype=tdtype,
        device=torch.device('cpu'),
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
    off = HunyuanVideo_1_5_Pipeline.get_offloading_config()
    pipe.apply_infer_optimization(
        infer_state=infer_state, enable_offloading=True,
        enable_group_offloading=off['enable_group_offloading'],
        overlap_group_offloading=True,
    )
    print(f"{tag} pipeline loaded in {time.time()-t0:.0f}s", flush=True)

    done = skipped = failed = 0
    for j, c in enumerate(cases):
        cid = c['case_id']
        out_dir = os.path.join(args.dataset_dir, cid)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, args.out_name)
        cond_path = os.path.join(out_dir, args.cond_name)
        prompt_path = os.path.join(out_dir, args.prompt_file)

        if os.path.exists(out_path) and os.path.getsize(out_path) > 5000:
            skipped += 1
            print(f"{tag} ({j+1}/{len(cases)}) skip existing {cid}", flush=True); continue
        if not os.path.exists(cond_path) or os.path.getsize(cond_path) < 5000:
            failed += 1
            print(f"{tag} ({j+1}/{len(cases)}) MISSING/bad cond video {cid}, skip", flush=True); continue
        # restyle prompt: prefer edit_prompt.txt, strip the --resolution... tail if present
        prompt = ''
        if os.path.exists(prompt_path):
            raw = open(prompt_path, encoding="utf-8").read().strip()
            prompt = raw.split('--resolution')[0].strip()
        if not prompt:
            prompt = c.get('caption') or ''
        if not prompt.strip():
            failed += 1
            print(f"{tag} ({j+1}/{len(cases)}) EMPTY prompt {cid}, skip", flush=True); continue

        t = time.time()
        try:
            cond_latents, used_frames = encode_video_to_latents(
                pipe, cond_path, max_frames=args.video_length)
            # fps from the source clay video
            from decord import VideoReader
            vr = VideoReader(cond_path); fps = round(vr.get_avg_fps()); del vr
            out = pipe(
                prompt=prompt,
                aspect_ratio=args.aspect_ratio,
                num_inference_steps=args.num_inference_steps,
                video_length=args.video_length,
                negative_prompt=args.negative_prompt,
                seed=args.seed,
                output_type="pt",
                task_type='editing',
                condition_video_latents=cond_latents,
                condition_videos=[cond_path],
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
