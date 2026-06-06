#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="/home/leizongru/lzr_ws/VINGS-Mono"
DATA_ROOT="/home/leizongru/lzr_ws/railway_data"
BASE_CONFIG="$ROOT_DIR/configs/railway.yaml"
PYTHON_BIN="/home/leizongru/miniconda3/envs/vings_vio/bin/python"
LOG_ROOT="$ROOT_DIR/logs"
RESERVER_PID_FILE="/home/leizongru/lzr_ws/gpu_reserver/hold_gpus.pid"
GPU_CSV="2,3"
STOP_RESERVER=1
DRY_RUN=0

SEQUENCES=(
  scene_05_train
  scene_11_train
  scene_13_train
  scene_14_train
  scene_16_train
  scene_17_train
  scene_19_train
)

usage() {
  cat <<USAGE
Usage: $0 [options]

Run the seven railway VO sequences with a two-GPU dynamic queue.

Options:
  --gpus 2,3          Physical GPU ids to use. Default: 2,3
  --keep-reserver     Do not stop /home/leizongru/lzr_ws/gpu_reserver before running.
  --config PATH       Base railway config. Default: $BASE_CONFIG
  --python PATH       Python executable. Default: $PYTHON_BIN
  --dry-run           Create configs/log folders and print commands without running VINGS-Mono.
  -h, --help          Show this help.

Outputs:
  Results: $ROOT_DIR/output/<sequence_name>/
  Logs:    $ROOT_DIR/logs/railway_7seq_<timestamp>/
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPU_CSV="$2"
      shift 2
      ;;
    --keep-reserver)
      STOP_RESERVER=0
      shift
      ;;
    --config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

IFS=',' read -r -a GPUS <<< "$GPU_CSV"
if [[ ${#GPUS[@]} -eq 0 ]]; then
  echo "No GPU ids provided." >&2
  exit 2
fi

RUN_ID="railway_7seq_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$LOG_ROOT/$RUN_ID"
CONFIG_DIR="$RUN_DIR/configs"
STATUS_DIR="$RUN_DIR/status"
QUEUE_FILE="$RUN_DIR/queue.txt"
LOCK_FILE="$RUN_DIR/queue.lock"
MASTER_LOG="$RUN_DIR/master.log"

mkdir -p "$CONFIG_DIR" "$STATUS_DIR"
printf '%s\n' "${SEQUENCES[@]}" > "$QUEUE_FILE"
: > "$LOCK_FILE"
: > "$MASTER_LOG"

log_msg() {
  local message="$1"
  printf '[%s] %s\n' "$(date '+%F %T')" "$message" | tee -a "$MASTER_LOG"
}

pids=()
cleanup_on_signal() {
  local signal="$1"
  log_msg "Received $signal; stopping background workers."
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  log_msg "Stopped after $signal. Check partial logs under $RUN_DIR"
  exit 130
}
trap 'cleanup_on_signal SIGINT' INT
trap 'cleanup_on_signal SIGTERM' TERM

stop_gpu_reserver() {
  if [[ "$STOP_RESERVER" -ne 1 ]]; then
    log_msg "Keeping gpu_reserver process if it exists."
    return 0
  fi

  if [[ ! -f "$RESERVER_PID_FILE" ]]; then
    log_msg "No gpu_reserver PID file found."
    return 0
  fi

  local pid
  pid="$(tr -d '[:space:]' < "$RESERVER_PID_FILE")"
  if [[ -z "$pid" ]]; then
    log_msg "gpu_reserver PID file is empty."
    return 0
  fi

  if ! kill -0 "$pid" 2>/dev/null; then
    log_msg "gpu_reserver PID $pid is not running; removing stale PID file."
    rm -f "$RESERVER_PID_FILE"
    return 0
  fi

  local cmdline
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  if [[ "$cmdline" != *"hold_gpus.py"* ]]; then
    log_msg "PID $pid does not look like hold_gpus.py; leaving it untouched. cmdline=$cmdline"
    return 0
  fi

  log_msg "Stopping gpu_reserver PID $pid before starting real VO jobs."
  kill "$pid"
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$RESERVER_PID_FILE"
      log_msg "gpu_reserver stopped."
      return 0
    fi
    sleep 0.5
  done

  log_msg "gpu_reserver PID $pid did not exit after SIGTERM; sending SIGKILL."
  kill -9 "$pid" 2>/dev/null || true
  rm -f "$RESERVER_PID_FILE"
}

write_sequence_config() {
  local seq="$1"
  local cfg_out="$2"
  "$PYTHON_BIN" - "$BASE_CONFIG" "$DATA_ROOT" "$seq" "$cfg_out" <<'PY'
import sys
from pathlib import Path
import yaml

base_config, data_root, sequence, output_path = sys.argv[1:5]
with open(base_config, 'r', encoding='utf-8') as f:
    cfg = yaml.safe_load(f)

cfg['dataset']['root'] = str(Path(data_root) / sequence) + '/'
cfg.setdefault('output', {})['save_dir'] = '/home/leizongru/lzr_ws/VINGS-Mono/output/'
cfg['output']['save_rgbdnua'] = False
cfg['output']['save_legacy_outputs'] = False
cfg['output']['render_height'] = 2504
cfg['output']['render_width'] = 4112

with open(output_path, 'w', encoding='utf-8') as f:
    yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
PY
}

next_sequence() {
  local next=""
  exec 9<>"$LOCK_FILE"
  flock 9
  if [[ -s "$QUEUE_FILE" ]]; then
    next="$(head -n 1 "$QUEUE_FILE")"
    tail -n +2 "$QUEUE_FILE" > "$QUEUE_FILE.tmp"
    mv "$QUEUE_FILE.tmp" "$QUEUE_FILE"
  fi
  flock -u 9
  echo "$next"
}

run_sequence() {
  local seq="$1"
  local gpu="$2"
  local cfg="$CONFIG_DIR/${seq}.yaml"
  local log_file="$RUN_DIR/${seq}_gpu${gpu}.log"
  local output_dir="$ROOT_DIR/output/$seq"
  local status=0

  log_msg "START seq=$seq gpu=$gpu output=$output_dir log=$log_file"
  {
    echo "Run id: $RUN_ID"
    echo "Sequence: $seq"
    echo "Physical GPU: $gpu"
    echo "CUDA_VISIBLE_DEVICES=$gpu"
    echo "Config: $cfg"
    echo "Output: $output_dir"
    echo "Start time: $(date '+%F %T')"
    echo
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi -i "$gpu" || true
      echo
    fi

    echo "Writing sequence config..."
    write_sequence_config "$seq" "$cfg" || status=$?

    if [[ "$status" -ne 0 ]]; then
      echo "Failed to write sequence config: $cfg"
    elif [[ "$DRY_RUN" -eq 1 ]]; then
      echo "DRY RUN: cd $ROOT_DIR && CUDA_VISIBLE_DEVICES=$gpu PYTHONUNBUFFERED=1 $PYTHON_BIN scripts/run.py $cfg --prefix $seq"
    else
      cd "$ROOT_DIR"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONUNBUFFERED=1 "$PYTHON_BIN" scripts/run.py "$cfg" --prefix "$seq" || status=$?
    fi
    echo
    echo "End time: $(date '+%F %T')"
    echo "Exit status: $status"
  } > "$log_file" 2>&1

  if [[ $status -eq 0 ]]; then
    echo "$seq gpu=$gpu log=$log_file" > "$STATUS_DIR/${seq}.ok"
    log_msg "DONE  seq=$seq gpu=$gpu"
  else
    echo "$seq gpu=$gpu status=$status log=$log_file" > "$STATUS_DIR/${seq}.fail"
    log_msg "FAIL  seq=$seq gpu=$gpu status=$status log=$log_file"
  fi
  return "$status"
}

worker() {
  local gpu="$1"
  local failures=0
  local seq
  log_msg "Worker for GPU $gpu is ready."
  while true; do
    seq="$(next_sequence)"
    if [[ -z "$seq" ]]; then
      break
    fi
    if ! run_sequence "$seq" "$gpu"; then
      failures=$((failures + 1))
    fi
  done
  log_msg "Worker for GPU $gpu finished with $failures failure(s)."
  return "$failures"
}

log_msg "Run directory: $RUN_DIR"
log_msg "Base config: $BASE_CONFIG"
log_msg "Python: $PYTHON_BIN"
log_msg "GPUs: ${GPUS[*]}"
log_msg "Sequences: ${SEQUENCES[*]}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  log_msg "Python executable is missing or not executable: $PYTHON_BIN"
  exit 2
fi
if [[ ! -f "$BASE_CONFIG" ]]; then
  log_msg "Base config not found: $BASE_CONFIG"
  exit 2
fi
for seq in "${SEQUENCES[@]}"; do
  if [[ ! -d "$DATA_ROOT/$seq" ]]; then
    log_msg "Dataset sequence directory not found: $DATA_ROOT/$seq"
    exit 2
  fi
done

stop_gpu_reserver

if command -v nvidia-smi >/dev/null 2>&1; then
  log_msg "GPU snapshot before launch:"
  nvidia-smi | tee -a "$MASTER_LOG" >/dev/null || true
fi

for gpu in "${GPUS[@]}"; do
  worker "$gpu" > "$RUN_DIR/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

ok_count=$(find "$STATUS_DIR" -name '*.ok' | wc -l)
fail_count=$(find "$STATUS_DIR" -name '*.fail' | wc -l)
log_msg "Summary: ok=$ok_count fail=$fail_count run_dir=$RUN_DIR"

if [[ "$fail_count" -gt 0 || "$failures" -gt 0 ]]; then
  log_msg "Some sequences failed. Check logs under $RUN_DIR"
  exit 1
fi

log_msg "All railway sequences finished successfully."
