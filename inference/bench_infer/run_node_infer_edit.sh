#!/usr/bin/env bash
# Editing-baseline runner: one hope worker owns a full 8-GPU node and fans out
# 8 python processes (one per GPU), each editing shard i of 8 of the manifest.
# Same structure as run_node_infer.sh but calls batch_infer_edit.py (task_type
# 'editing'): each case's clay ref video.mp4 is the condition, guided by its
# edit_prompt.txt, output -> <case>/omniweaving_edit.mp4.
#
# Env in (set by the .hope worker.script):
#   ENV_DIR REPO CKPTS MANIFEST DSROOT LOGDIR   REQUIRED
#   VIDEO_LENGTH / STEPS / OUT_NAME / DEBUG_LIMIT   optional
set -u
NGPU=$(nvidia-smi --list-gpus | wc -l)
: "${VIDEO_LENGTH:=81}"
: "${STEPS:=50}"
: "${OUT_NAME:=omniweaving_edit.mp4}"
: "${DEBUG_LIMIT:=0}"
mkdir -p "$LOGDIR"
cd "$REPO"

echo "[edit-runner] node has $NGPU GPUs; launching $NGPU shards. manifest=$MANIFEST dsroot=$DSROOT"
pids=()
for i in $(seq 0 $((NGPU-1))); do
  CUDA_VISIBLE_DEVICES=$i LOCAL_RANK=0 WORLD_SIZE=1 RANK=0 \
    "$ENV_DIR/bin/python" -u bench_infer/batch_infer_edit.py \
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
  echo "[edit-runner] gpu $i -> pid ${pids[-1]}, log $LOGDIR/gpu_${i}.log"
done

rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "[edit-runner] all shards finished rc=$rc"
exit $rc
