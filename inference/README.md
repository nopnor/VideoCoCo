# VideoCoCo Inference

Batch inference for the VideoCoCo tuned transformer on top of
[Tencent HY-OmniWeaving](https://github.com/Tencent-Hunyuan/OmniWeaving)
(HunyuanVideo-1.5).

VideoCoCo does **not** fork the OmniWeaving model code. This directory ships only
our own scripts plus a small patch against the upstream repo, so you run the
official codebase with our tuned weights.

## Contents

```
inference/
├── bench_infer/
│   ├── batch_infer.py            # text -> video baseline over a manifest shard
│   ├── batch_infer_edit.py       # video editing: clay proxy + edit prompt -> photoreal
│   ├── export_tuned_ckpt.py      # export a full-SFT training checkpoint to a pipeline
│   ├── export_tuned_lora_ckpt.py # export a LoRA checkpoint to a full transformer
│   ├── run_node_infer.sh         # 8-GPU fan-out runner (text->video)
│   └── run_node_infer_edit.sh    # 8-GPU fan-out runner (editing)
└── patches/
    └── videococo-omniweaving.patch  # our diffs to upstream hyvideo/ and train.py
```

## Setup

1. Clone the official OmniWeaving and set up its environment (see its README):

   ```bash
   git clone https://github.com/Tencent-Hunyuan/OmniWeaving.git
   cd OmniWeaving
   pip install -r requirements.txt
   ```

2. Apply the VideoCoCo patch (small changes to `hyvideo/` and `train.py`):

   ```bash
   git apply /path/to/videococo/inference/patches/videococo-omniweaving.patch
   ```

3. Copy our inference scripts into the repo and download the tuned weights:

   ```bash
   cp -r /path/to/videococo/inference/bench_infer ./bench_infer
   huggingface-cli download mickyhimself/VideoCoCo --local-dir ./VideoCoCo-weights
   ```

   Then assemble a pipeline directory: take the base OmniWeaving pipeline
   (VAE, text/vision encoders, scheduler, upsampler) and replace its
   `transformer/` with `VideoCoCo-weights/transformer/`.

## Run (video editing)

`batch_infer_edit.py` takes each case's clay proxy (`video.mp4`) as the
condition video and its `edit_prompt.txt` as the restyle instruction, writing a
photoreal result per case. It expects a JSONL/JSON manifest and a dataset
directory laid out like `data/toy_cases/` in this repo.

Single GPU:

```bash
python bench_infer/batch_infer_edit.py \
  --model_path  /path/to/assembled_pipeline \
  --manifest    /path/to/data/toy_cases/manifest.jsonl \
  --dataset_dir /path/to/data/toy_cases \
  --out_name    omniweaving_edit.mp4 \
  --video_length 81 --num_inference_steps 50
```

Full 8-GPU node (shards the manifest across all visible GPUs):

```bash
ENV_DIR=/path/to/python_env \
REPO=$PWD \
CKPTS=/path/to/assembled_pipeline \
MANIFEST=/path/to/data/toy_cases/manifest.jsonl \
DSROOT=/path/to/data/toy_cases \
LOGDIR=./_logs \
bash bench_infer/run_node_infer_edit.sh
```

Text-to-video (`batch_infer.py` / `run_node_infer.sh`) follows the same pattern
but uses a caption manifest instead of proxy videos.

## License

These scripts are written against OmniWeaving and are licensed under the
**Tencent HY Community License Agreement** (see the header in each file and the
[upstream LICENSE](https://github.com/Tencent-Hunyuan/OmniWeaving/blob/main/LICENSE)).
The tuned weights are a Model Derivative under the same agreement.
