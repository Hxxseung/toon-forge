#!/usr/bin/env bash
# GPU 2장에 컷을 반씩 나눠 같은 명령을 동시에 실행.
# 예) bash tools/run_dual_gpu.sh image --axis character --strength weak medium strong
#     bash tools/run_dual_gpu.sh both  --axis character --strength medium --text-level full
# 로그: logs/<명령>_gpu{0,1}.log
set -euo pipefail
cmd="$1"; shift
mkdir -p logs
pids=()
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g python -m replica.cli "$cmd" --shard "$g/2" "$@" \
    > "logs/${cmd}_gpu${g}.log" 2>&1 &
  pids+=($!)
  echo "GPU$g 시작 (pid ${pids[-1]}) → logs/${cmd}_gpu${g}.log"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ $fail -eq 0 ] && echo "완료" || { echo "실패한 GPU가 있습니다. 로그를 확인하세요."; exit 1; }
