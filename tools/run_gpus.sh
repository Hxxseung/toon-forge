#!/usr/bin/env bash
# 모든 GPU에 프로세스를 하나씩 띄워 작업 큐를 함께 처리 (먼저 끝난 GPU가 남은 일을 가져감).
# 명령을 여러 개 주면 GPU마다 순서대로 이어서 실행 → GPU가 놀지 않음.
#
#   bash tools/run_gpus.sh "image --axis character --strength weak medium strong" \
#                          "both  --axis character --strength medium --text-level full"
#
#   GPUS="0 1" bash tools/run_gpus.sh "…"     # 쓸 GPU 지정 (기본: 전부)
#   CPU_WORKERS=4 bash tools/run_gpus.sh "ocr"  # GPU 없이 CPU 프로세스 N개 (ocr·text용)
#
# 로그: logs/<run>/w<i>.log      진행 보기: tail -f logs/<run>/*.log   GPU 보기: watch -n1 nvidia-smi
set -uo pipefail
[ $# -ge 1 ] || { sed -n 2,11p "$0"; exit 1; }

RUN="run_$(date +%m%d_%H%M%S)"
mkdir -p "logs/$RUN"

if [ -n "${CPU_WORKERS:-}" ]; then
  WORKERS=(); for ((i=0; i<CPU_WORKERS; i++)); do WORKERS+=(""); done
else
  GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}"
  read -ra WORKERS <<< "$GPUS"
fi
N=${#WORKERS[@]}

# RAM 확인: 오프로드 방식은 프로세스마다 모델을 RAM에 올림 (대략 30~35GB/프로세스)
if [ -z "${CPU_WORKERS:-}" ]; then
  avail=$(awk '/MemAvailable/ {printf "%d", $2/1024/1024}' /proc/meminfo)
  need=$((N * 32))
  echo "사용 가능 RAM ${avail}GB / 예상 필요 ~${need}GB (프로세스 ${N}개)"
  if [ "$avail" -lt "$need" ]; then
    echo "⚠ RAM이 부족할 수 있습니다. 스왑이 생기면 매우 느려집니다 → GPUS=\"0\" 으로 한 장만 쓰거나 RAM 확보"
  fi
fi

pids=()
for ((i=0; i<N; i++)); do
  dev="${WORKERS[$i]}"
  (
    [ -n "$dev" ] && export CUDA_VISIBLE_DEVICES="$dev"
    for cmd in "$@"; do
      echo "=== $(date +%T) 시작: $cmd"
      # 모든 명령이 같은 큐 공유 (작업 ID가 서로 달라 겹치지 않음) → image 중인 부모를 both가 중복 생성 안 함
      # shellcheck disable=SC2086
      python -m replica.cli $cmd --queue "$RUN" --workers "$N" || { echo "!!! 실패: $cmd"; exit 1; }
    done
    echo "=== $(date +%T) 모두 완료"
  ) > "logs/$RUN/w$i.log" 2>&1 &
  pids+=($!)
  echo "작업자 $i ${dev:+(GPU $dev)} 시작 → logs/$RUN/w$i.log"
done

echo "진행 보기: tail -f logs/$RUN/*.log"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ $fail -eq 0 ]; then echo "완료 ($RUN)"; else echo "실패한 작업자가 있습니다. logs/$RUN 확인"; exit 1; fi