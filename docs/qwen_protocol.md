# Qwen Direct-Semantic Protocol

> 2026-07-12 scope decision: Qwen production batch는 workshop 본문 일정에서 보류한다. 현재 main comparison은
> V-JEPA Physical Safety Query와 LTX Physical Safety Query에 집중하며, 이 문서는 후속 실험 protocol로
> 보존한다.

## 1. 목적

Qwen3.6 multimodal에 고정 CCTV observation prefix만 제공하고, 미래 5초 이내의 lateral tip/wheel-lift
outcome을 직접 JSON으로 예측하게 한다.

```text
CCTV prefix RGB
-> Qwen3.6-27B-FP8
-> {predicted_outcome, tip_probability}
```

이 실험은 future RGB를 생성하는 LTX나 trainable readout을 붙이는 V-JEPA J1/J2와 다르다. Foundation
model의 native direct-semantic answer를 측정하며 별도 safety head를 학습하지 않는다.

Primary questions:

1. Qwen의 `tip_probability`가 definitive test runs의 미래 outcome을 rank할 수 있는가?
2. 같은 command의 Gold pair에서 실제 tip endpoint에 더 높은 probability를 주는가?
3. 성능이 speed/route prevalence와 단순 영상 shortcut을 넘는가?

## 2. 현재 상태

| 항목 | 상태 |
|---|---|
| Qwen checkpoint download | complete |
| Local preprocessing/parser | tested |
| Historical 5090 compatibility smoke | complete, production 결과에서 제외 |
| DGX Spark launcher | ready |
| DGX Spark production smoke | pending |
| Resumable full batch runner | pending implementation |

현재 repository에는 single-request smoke client가 있다. 이 문서는 batch 구현 전에 selection, prompt,
metric을 동결하기 위한 protocol이다. 문서 작성 시점에는 `safety_probing.qwen_batch`가 아직 없으므로 존재하지
않는 batch command를 실행하면 안 된다.

## 3. Model and Runtime Contract

| 항목 | 값 |
|---|---|
| Model | `Qwen/Qwen3.6-27B-FP8` |
| Pinned revision | `e89b16ebf1988b3d6befa7de50abc2d76f26eb09` |
| Hardware | DGX Spark, GB10, 128GB unified memory |
| Backend | NVIDIA ARM64 vLLM container `26.06-py3` |
| Served name | `qwen3.6-27b-fp8` |
| Context limit | 4096 |
| Max sequences | 1 |
| GPU memory utilization | 0.80 |
| CPU offload | 0, fallback prohibited |
| Thinking | disabled |
| Temperature | 0.0 |
| Sampling | disabled |
| Max output tokens | 160 |

RTX 5090에서 CPU offload로 본 실험을 실행하지 않는다. 5090의 과거 smoke는 API와 kernel compatibility
기록일 뿐이며, production result는 DGX Spark full-model load에서만 만든다.

## 4. Input Contract

| 항목 | 값 |
|---|---|
| Camera | `main_fixed_northwest` |
| Observation | 0.0-2.7083 s |
| Source frames | 65 at 24 FPS, indices 0-64 |
| Model frames | 20 uniformly sampled frames |
| Video count/request | 1 |
| Max total video pixels | 2,293,760 |
| Transport | base64 MP4 data URI |

Observation은 2.0초 steering onset 이후 약 0.7초의 초기 yaw response를 포함하지만 가장 빠른 definitive
event 2.958초보다 앞선다. Future RGB, dense log, mass, CoM, route label, seed, answer label은 request에
포함하지 않는다.

vLLM processor가 65-frame MP4에서 정확히 20 frames를 읽는지 DGX smoke에서 확인한다. 가능하면 local
preprocessing audit의 uniform indices와 server processor log를 비교한다. Frame count가 다르면 본 batch를
시작하지 않는다.

## 5. Primary Prompt

Prompt source of truth는 `configs/experiments/qwen_3_6_direct.json`이다.

```text
You are given only a fixed CCTV video prefix from 0.0 to 2.7083 seconds.
Predict whether the same forklift will exhibit a sustained lateral tip or
wheel-lift instability before 5.0 seconds. Do not infer or invent hidden
payload metadata. Return exactly one JSON object and no other text:
{"predicted_outcome": "observed_tip" or "confirmed_recovered",
 "tip_probability": number from 0 to 1}.
```

Main protocol은 video-only다. Future steering schedule을 text로 제공하지 않는다. LTX action-aware prompt와
직접 비교할 필요가 생기면 동일 future-action description을 Qwen과 V-JEPA에도 제공한 별도 ablation으로
분리한다. 그 ablation을 video-only main result와 합치지 않는다.

Validation/test output을 본 뒤 prompt wording이나 label definition을 바꾸면 새 protocol version과 새 output
root를 사용한다.

## 6. Output Schema

vLLM strict JSON schema:

```json
{
  "predicted_outcome": "observed_tip",
  "tip_probability": 0.73
}
```

Rules:

- `predicted_outcome`: `observed_tip` 또는 `confirmed_recovered`
- `tip_probability`: finite number in `[0,1]`
- Additional fields: forbidden
- Free text, explanation, markdown: forbidden

Primary ranking score는 `tip_probability`다. `predicted_outcome`은 semantic consistency check에 사용한다.
`tip_probability >= 0.5`인데 recovered를 출력하거나 그 반대인 경우 contradiction으로 집계하되 probability를
사후 수정하지 않는다.

## 7. Evaluation Cohorts

### Run-level cohort

Qwen을 Gold 10 pair의 20 endpoints에만 실행하지 않는다. Canonical condition 전체를 한 번씩 실행한다.

| Split | Canonical runs | Definitive binary | Tip | Recovered | Censored |
|---|---:|---:|---:|---:|---:|
| Validation | 98 | 98 | 58 | 40 | 0 |
| Test | 102 | 99 | 60 | 39 | 3 |

Primary inference batch는 validation 98 + test 102 = **200 requests**다. Test AUROC/AUPRC는 definitive 99
runs에서 계산한다. Right-censored 3 runs의 prediction은 보존하지만 positive/negative로 강제하지 않는다.

Temperature 0이므로 model seed 반복 batch는 사용하지 않는다. 대신 validation에서 10개 condition을 골라
각 3회 요청하는 repeatability audit을 수행하고 probability 및 JSON의 exact match rate를 보고한다.

### Pair-level cohort

| Main split | All pairs | Gold | Silver | Bronze | Reject | Censored |
|---|---:|---:|---:|---:|---:|---:|
| Train | 426 | 31 | 85 | 37 | 257 | 16 |
| Validation | 84 | 17 | 22 | 4 | 41 | 0 |
| Test | 94 | 10 | 30 | 12 | 38 | 4 |

- Primary causal pair metric: test Gold 10
- Validation pair model-selection diagnostic: Gold 17
- Secondary severity control: test Silver 30 + Bronze 12
- Excluded from primary: Reject 38, Censored 4

Silver/Bronze 42 pairs는 두 endpoint의 binary outcome이 같다. 따라서 이들을 Gold처럼 `tip vs recovered`
pair accuracy에 넣지 않는다. Qwen probability가 future margin 또는 realized roll consequence의 순서를
따르는지 보는 secondary continuous-severity control로만 사용한다.

## 8. 왜 Gold Test Pair가 10개뿐인가

### Local pair funnel

| 단계 | Pairs | 남는 비율 |
|---|---:|---:|
| Generated local shard 01 | 604 | 100.0% |
| Both endpoints definitive | 584 | 96.7% |
| Different definitive outcome, Gold | 58 | 9.6% |
| Main split test, all tiers | 94 | 15.6% |
| Main split test Gold | 10 | 1.7% of all local pairs |

604개 모두 same-action pair다. 문제는 action mismatch가 아니라 **대부분 A/B가 같은 결과를 낸다**는 점이다.

| Local transition | Pairs |
|---|---:|
| Both tip | 308 |
| Both recovered | 218 |
| Tip/recovered transition | 58 |
| At least one censored | 20 |

Definitive 584 pairs 중 526개, 즉 **90.1%**가 same-outcome이다. Slow route는 400/400 runs이 recovered이고,
fast route는 398 definitive runs 중 389가 tip이다. 이런 포화 regime에서 mass/CoM을 한 단계 바꿔도
binary boundary를 넘지 않으므로 Gold가 생기지 않는다. Gold는 주로 medium-speed boundary regime에
몰린다.

그 다음 leakage-safe split이 pair ID를 무작위로 자르는 대신 shared endpoint와 mirror component 전체를
train/validation/test에 배정한다. Local 604 pairs 중 test가 94개이고 그 안에 Gold가 10개다. 이 split을
느슨하게 바꿔 Gold test 수를 억지로 늘리면 shared physical condition leakage가 다시 생긴다.

### Run 수가 188이 아니라 102인 이유

Test 94 pairs에는 endpoint instance가 188개 있지만 인접 mass/CoM pair가 같은 physical endpoint를 공유한다.
전체 local 1,208 run instances는 canonical dedup 후 670 conditions가 된다. Test도 188 instances가 102
canonical conditions로 줄어든다. 이것은 데이터 손실이 아니라 같은 영상을 여러 번 세어 confidence를
부풀리지 않기 위한 중복 제거다.

### 쓸 수 있는 데이터가 정말 10개뿐인 것은 아니다

- Run-level binary test: 99 definitive conditions
- Gold causal test: 10 pairs
- Accepted test severity pairs: 52 = Gold 10 + Silver 30 + Bronze 12
- Censored test runs: survival/abstention analysis 3

따라서 Qwen의 일반 분류 성능은 99 runs에서 평가한다. 다만 논문의 핵심 질문인 "같은 command에서 hidden
intervention 때문에 outcome이 갈리는가"에 대한 깨끗한 test pair가 10개뿐인 것이다.

## 9. Primary Metrics

### Run-level definitive test

- AUROC, using raw `tip_probability`
- AUPRC
- Balanced accuracy
- F1
- Brier score
- ECE, 10 equal-width bins
- Confusion matrix

두 threshold를 별도로 보고한다.

1. Native threshold `0.5`
2. Validation에서 balanced accuracy를 최대화한 threshold

Test로 threshold를 고르지 않는다. Raw probability의 Platt/isotonic recalibration은 zero-shot main에
적용하지 않는다. 필요하면 validation-only calibrated ablation으로 분리한다.

### Gold pair

Gold pair에서 실제 tip endpoint probability가 recovered endpoint보다 높으면 correct다. Exact tie는 0.5
credit을 준다.

- Overall pair accuracy
- Left/right
- Load mass/CoM shift
- Pair score gap `p_tip - p_recovered`

Test Gold가 10개이므로 Wilson interval과 exact binomial test를 함께 보고한다. 여러 request retry를 독립
pair 표본으로 세지 않는다.

### Secondary severity control

Silver/Bronze는 binary pair metric과 분리한다.

- Probability difference와 future outward ZMP-margin difference의 Spearman correlation
- Margin-defined risk ordering accuracy
- Silver/Bronze 별도 집계

ZMP sign을 binary label로 사용하지 않는다. 이 분석은 direct-semantic probability가 같은-outcome 안의
severity까지 반영하는지 보는 exploratory control이다.

## 10. Runtime and Reliability Metrics

- HTTP/schema success rate
- Parse-valid rate
- Outcome/probability contradiction count
- Request latency median, p90, p95
- Prompt/completion/total tokens
- Retry count and terminal failure count
- DGX unified-memory peak from server monitoring
- Repeatability exact-match rate

Transport timeout이나 HTTP 5xx만 최대 3회 retry한다. Schema-valid response는 결과가 마음에 들지 않아도
재생성하지 않는다. 각 condition result는 atomic sidecar로 저장하고 같은 config hash에서만 resume한다.

## 11. Required Result Record

Condition별 sidecar는 최소 다음을 보존한다.

```text
condition_id
run_id
split
safety_outcome
binary_tip_label
future_prediction_eligible
pair_id
run_label
turn_direction
intervention_type
video_path
prompt_sha256
experiment_sha256
model_revision
backend/container
predicted_outcome
tip_probability
raw_text
usage
latency_s
retry_count
created_at_utc
```

Batch root에는 `batch_manifest.jsonl`, `batch_summary.json`, config snapshot을 둔다. MP4와 model checkpoint는
복사하지 않고 path/hash provenance만 기록한다.

## 12. Leakage and Prompt Controls

1. `canonical_runs.jsonl`만 사용해 duplicate condition을 반복 inference하지 않는다.
2. Video 외 mass, CoM, speed label, route, seed, ground truth를 request에 넣지 않는다.
3. Validation/test output을 본 뒤 prompt를 수정하지 않는다.
4. Threshold는 validation에서만 선택한다.
5. Test Gold pair는 prompt selection에 사용하지 않는다.
6. Future action text를 쓰는 실험은 별도 action-conditioned ablation으로 표시한다.

Recommended negative controls:

- Prompt-only/no-video label-prior audit
- Last-frame repeated video
- Temporally reversed observation
- Speed/route prevalence baseline
- Left/right mirror subgroup

## 13. Execution

### DGX Spark server

```bash
cd ~/safety_probing
scripts/start_qwen_vllm.sh
```

Server startup에서 다음을 확인한다.

```text
architecture = aarch64
container = nvcr.io/nvidia/vllm:26.06-py3
cpu_offload_gb = 0
served_model_name = qwen3.6-27b-fp8
port = 8016
```

### Workstation smoke client

```bash
cd ~/isaac_ws/safety_probing
export QWEN_VLLM_SERVER_URL=http://dgx-spark.local:8016

curl --fail "$QWEN_VLLM_SERVER_URL/health"
.venv/bin/safety-qwen-vllm-smoke --outcome confirmed_recovered
.venv/bin/safety-qwen-vllm-smoke --outcome observed_tip
```

Smoke가 schema, sampled frame count, latency, memory 조건을 통과한 뒤 batch runner를 구현하고 200 canonical
requests를 실행한다.

## 14. Interpretation Rules

주장 가능:

- Qwen raw semantic score의 run-level discrimination과 calibration
- Same-command Gold pair에서의 risk ordering
- Parse/reliability/runtime 특성

주장 불가:

- High run AUROC만으로 hidden mass/CoM reasoning을 증명
- Silver/Bronze를 binary Gold pair처럼 합산
- 10 Gold pair의 point estimate만으로 일반화 확정
- Right-censored를 stable negative로 처리
- Action-aware LTX와 video-only Qwen을 동일 조건으로 직접 비교

## 15. Gold Pair를 늘리는 올바른 방법

Split을 오염시키거나 Reject pair를 Gold로 이름만 바꾸면 안 된다. 추가 simulator generation에서
near-boundary adaptive sampling을 사용해야 한다.

1. Route/seed/load geometry를 고정한다.
2. Mass 또는 lateral CoM을 sweep해 recovered/tip transition bracket을 찾는다.
3. Bracket 안에서 bisection 또는 촘촘한 local sampling을 한다.
4. Boundary 양쪽의 visually similar matched pair를 생성한다.
5. Mirror counterpart를 같은 split component에 묶는다.
6. 새 untouched test component를 확보한다.

현재 full grid의 나머지 shard를 합치면 표본은 늘지만 Gold 비율이 shard 01과 비슷하다면 test Gold가
충분히 커지지 않을 수 있다. Full-shard statistics를 먼저 재계산하고, 그 뒤 boundary-focused 추가 생성량을
결정한다.

## 16. Source Files

- Experiment: `configs/experiments/qwen_3_6_direct.json`
- Model registry: `configs/model_registry.json`
- Hardware policy: `configs/hardware_profiles.json`
- Server launcher: `scripts/start_qwen_vllm_dgx_spark.sh`
- vLLM adapter: `src/safety_probing/adapters/qwen_vllm.py`
- Parser/local adapter: `src/safety_probing/adapters/qwen_direct.py`
- Smoke client: `src/safety_probing/qwen_vllm_smoke.py`
- DGX runbook: `docs/DGX_SPARK_RUNBOOK_KO.md`
