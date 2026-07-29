#!/usr/bin/env bash
# One hope worker owns a full 8-GPU node. This fans out 8 python processes,
# one pinned per GPU, each rendering shard i of 8 of the manifest. Every process
# loads the OmniWeaving pipeline once, then loops its prompts.
#
# Env in (set by the .hope worker.script):
#   ENV_DIR   conda env prefix (shared-disk)                REQUIRED
#   REPO      OmniWeaving repo dir                          REQUIRED
#   CKPTS     model weights dir                             REQUIRED
#   MANIFEST  benchmark manifest.json                       REQUIRED
#   DSROOT    dataset dir (<case_id>/ live here)            REQUIRED
#   LOGDIR    where per-gpu logs go                         REQUIRED
#   VIDEO_LENGTH / STEPS / OUT_NAME                         optional
set -u
NGPU=$(nvidia-smi --list-gpus | wc -l)
: "${VIDEO_LENGTH:=81}"
: "${STEPS:=50}"
: "${OUT_NAME:=omniweaving.mp4}"
: "${DEBUG_LIMIT:=0}"
mkdir -p "$LOGDIR"
cd "$REPO"

echo "[runner] node has $NGPU GPUs; launching $NGPU shards. manifest=$MANIFEST dsroot=$DSROOT"
pids=()
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i LOCAL_RANK=0 WORLD_SIZE=1 RANK=0 \
    "$ENV_DIR/bin/python" -u bench_infer/batch_infer.py \
      --model_path "$CKPTS" \
      --manifest "$MANIFEST" \
      --dataset_dir "$DSROOT" \
      --shard_idx "$i" --shard_total "$NGPU" \
      --out_name "$OUT_NAME" \
      --video_length "$VIDEO_LENGTH" \
      --num_inference_steps "$STEPS" \
      --limit "$DEBUG_LIMIT" \
      > "$LOGDIR/gpu_${i}.log" 2>&1 &
  pids+=($!)
  echo "[runner] gpu $i -> pid ${pids[-1]}, log $LOGDIR/gpu_${i}.log"
done

# wait for all shards; report any failures
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[runner] all shards finished rc=$rc"
exit $rc
