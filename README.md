# webtoon-genai-replica

원작 웹툰 컷에서 생성형 AI 2차창작형 변형본(Replica)을 만들고, 원작↔변형 정답 짝을 매니페스트로 기록한다.
"플롯 고정" 원칙: 구도·포즈·서사는 그대로 두고 외형/텍스트만 바꾼다.

| 변형 | 바뀌는 것 | 유지되는 것 |
|---|---|---|
| `image` | 캐릭터 / 배경 / 화풍 (weak·medium·strong) | 글자 획 픽셀(원본에서 다시 붙임) |
| `text`  | 대사 속 이름·지명·아이템(조사 보정), 효과음, 크레딧 (+선택: LLM 의역) | 그림 |
| `both`  | `image` 결과 위에 `text` 변형 | — (`parent_variant_id`로 이미지 변형과 연결) |

## 폴더
```
data/originals/<work_id>/<episode>/*.jpg   # 입력: 세로 스크롤 조각 (파일명 순 = 위→아래)
configs/default.yaml                        # 경로·모델·프롬프트 템플릿
configs/works/<work_id>.yaml                # 작품별 치환표·재설계표 (_example.yaml 복사)
outputs/ocr/<cut_id>.json  outputs/variants/{image,text,both}/…   # 원본 컷 이미지는 저장하지 않음
# 원본 컷 = originals.jsonl 의 source_dir 조각들을 이어 붙여 y_range 만큼 자른 것 (replica.cli.load_original)
outputs/manifests/{originals,variants}.jsonl (+ export 시 .csv)
```

## 실행 순서
```bash
pip install -r requirements.txt && sudo apt install fonts-nanum
python -m replica.cli cut                     # 회차 → 컷, originals.jsonl
python -m replica.cli ocr                     # PaddleOCR → outputs/ocr/*.json (사람이 고쳐도 됨, 변형은 이 JSON만 읽음)
python -m replica.cli text  --text-level full # names | full | paraphrase
python -m replica.cli image --axis character --strength weak medium strong --backend qwen
python -m replica.cli both  --axis character --strength medium --text-level full
python -m replica.cli export                  # originals.csv / variants.csv
```
공통 옵션: `--works A B`, `--limit N`, `--overwrite`. 이미 매니페스트에 있는 ID는 건너뜀(중단 후 재실행 가능).

## 매니페스트
- **cut_id** = `<work>__<episode>__c###`
- **variant_id** = `<cut_id>__<type>-<axis>-<strength>-s<seed>` — 시드는 (cut_id, axis, strength) 해시로 고정 → 재현 가능
- variants 한 줄에 기록: `original_cut_id`(정답 짝), 유형·축·강도·시드, 경로·sha256,
  `image_edit`(backend, model_id, prompt, steps, guidance, 참조 이미지, 텍스트 보호 여부),
  `text_edit`(표 버전, 블록별 before→after·bbox·적용 규칙), `parent_variant_id`, `qc.pixel_change_ratio`, 코드 커밋, 시각
- `status`: `ok` / `skipped_no_text`(바꿀 텍스트 없음) / `skipped_no_ocr`

## 편집 백엔드
| 이름 | 모델 | 라이선스 |
|---|---|---|
| `qwen` | Qwen/Qwen-Image-Edit | Apache-2.0 |
| `qwen_plus` | Qwen/Qwen-Image-Edit-2509 (캐릭터 참조 이미지 입력) | Apache-2.0 |
| `flux_kontext` | black-forest-labs/FLUX.1-Kontext-dev | **비상업** — 공개 배포 전 확인 |
| `dummy` | 색만 바꿈 (테스트용) | — |

VRAM이 부족하면 해당 백엔드의 `cpu_offload: true`.

### RTX 3080 ×2 (VRAM 10~12GB)
- 백엔드 `qwen_nunchaku` (Qwen-Image-Edit-2509 + Nunchaku INT4, 블록 단위 오프로드). `pip install nunchaku` 필요, 시스템 RAM 64GB 이상 권장
- GPU 2장은 메모리가 합쳐지지 않으므로 **컷을 반씩 나눠 GPU마다 프로세스 1개**:
  ```bash
  bash tools/run_dual_gpu.sh image --axis character --strength weak medium strong
  ```
  (= `CUDA_VISIBLE_DEVICES=g python -m replica.cli image --shard g/2 …` 두 개). 매니페스트는 파일 잠금으로 안전하게 함께 기록됨
- `cut`·`ocr`·`text`는 한 프로세스로 충분

## 테스트 (GPU·OCR 없이)
```bash
python tests/test_synthetic.py /path/to/korean_font.ttf
```

## 알려진 한계
- 효과음/크레딧 판정은 규칙 기반 → OCR JSON의 `category`를 사람이 고칠 수 있게 둠
- 호격 `야`(→아)와 서술격 `야`(→이야)는 구분 불가 → 호격으로 처리, 규칙에 `(ambiguous)` 표시
- 텍스트 보호는 원본 글자 획을 다시 붙이는 방식이라, 말풍선 모양이 크게 바뀌는 strong 편집에선 글자가 어색하게 뜰 수 있음 → QC로 걸러야 함
- 폰트는 원작 폰트와 다름 (나눔고딕)
