# LTX Decode-Re-encode Round-trip Experiment

작성일: 2026-07-14
상태: 실험 완료, 현재 논문 `main`에는 미반영

## 1. 목적

기존 LTX safety-query 결과에서 다음 두 원인이 섞여 있는지 확인한다.

1. 생성 미래가 물리 정보를 충분히 보존하지 못함
2. GT posterior latent에서 학습한 고정 PSQ가 native generated latent 분포에 맞지 않음

이를 위해 GT와 generated 양쪽을 같은 LTX VAE `decode -> re-encode` 경로에 통과시킨 뒤, 기존
GT-trained PSQ checkpoint를 재학습 없이 적용했다.

```text
GT native        -> PSQ
GT native        -> decode -> re-encode -> PSQ
Generated native -> PSQ
Generated native -> decode -> re-encode -> PSQ
```

## 2. 실험 계약

| 항목 | 값 |
|---|---|
| Conditions | validation 120 + test 120 = 240 |
| Decoder seeds | 17, 29, 43, 71 |
| Jobs | pathway당 960, 총 1,920 |
| Fixed PSQ seeds | 17, 29, 43 |
| Source frames | 120 RGB frames, 내부 121 frames |
| Full latent | 16 temporal tokens |
| Evaluated future | latent indices `9:16`, shape `[7, 10, 18, 128]` |
| Pixel transport | memory tensor, MP4 codec 없음 |
| VAE dtype | bfloat16 |
| Foundation model | frozen |
| PSQ weights | frozen, 재학습 없음 |

Generated round-trip은 각 rollout의 원래 seed를 사용했다. GT round-trip은 같은 네 seed를 사용해 decoder
stochasticity 조건을 맞췄다. 각 condition의 네 decoder-seed 예측을 먼저 평균한 뒤 평가했다.

## 3. 실행 결과

### Primary test metrics

| Pathway | Mass MAE (kg) | CoM MAE (m) | Roll MAE (deg) | ZMP MAE (m) | Tip AUROC |
|---|---:|---:|---:|---:|---:|
| GT native | 14.47 | 0.0318 | 2.25 | 0.0316 | 1.0000 |
| GT round-trip | 18.19 | 0.0428 | 2.35 | 0.0351 | 0.9999 |
| Generated native | 95.50 | 0.1828 | 16.16 | 0.1114 | 0.7612 |
| Generated round-trip | 97.72 | 0.1833 | 16.34 | 0.1089 | 0.7700 |

### Matched intervention direction

| Pathway | Mass direction | CoM direction |
|---|---:|---:|
| GT native | 1.0000 | 0.9938 |
| GT round-trip | 1.0000 | 0.9877 |
| Generated native | 0.8507 | 0.5864 |
| Generated round-trip | 0.8681 | 0.6111 |

### Gap persistence

`(generated round-trip - GT round-trip) / (generated native - GT native)`로 계산한 gap persistence:

| Target | Persistence |
|---|---:|
| Mass MAE | 98.14% |
| CoM MAE | 93.05% |
| Roll MAE | 100.53% |
| ZMP MAE | 92.39% |

Decode-re-encode 이후에도 generated fixed-readout gap은 대부분 유지되었다.

## 4. 사전 Gate 판정

전체 gate는 **partial fail**이다.

| Check | Result |
|---|---|
| Tip AUROC drop <= 0.02 | pass, drop 0.00010 |
| Mass MAE ratio <= 1.25 | fail, 1.2569 |
| CoM MAE ratio <= 1.25 | fail, 1.3468 |
| Roll MAE ratio <= 1.25 | pass, 1.0474 |
| ZMP MAE ratio <= 1.25 | pass, 1.1126 |
| Mass direction drop <= 0.05 | pass, 0.0000 |
| CoM direction drop <= 0.05 | pass, 0.0062 |

GT round-trip 자체가 mass와 CoM readout을 사전 기준보다 크게 훼손했으므로, 이 실험만으로 generated latent에
물리 정보가 없다고 결론내릴 수 없다. 허용되는 해석은 더 제한적이다.

> 대칭적인 decode-re-encode를 적용해도 generated representation의 fixed-readout transfer gap은 회복되지
> 않았다. 다만 GT control도 일부 물성 readout을 훼손하므로, content loss와 representation mismatch는 아직
> 완전히 분리되지 않았다.

## 5. 재실행 명령

```bash
cd ~/isaac_ws/safety_probing

PYTHONPATH=src external/LTX-2/.venv/bin/python \
  -m safety_probing.ltx_roundtrip --source gt

PYTHONPATH=src external/LTX-2/.venv/bin/python \
  -m safety_probing.ltx_roundtrip --source generated

PYTHONPATH=src external/LTX-2/.venv/bin/python \
  -m safety_probing.ltx_roundtrip_evaluate
```

Runner는 per-job fingerprint 기반 resume를 지원한다. 기존 완료 part를 다시 계산하려면 명시적으로
`--overwrite`를 사용해야 한다.

## 6. 산출물

- Config: `configs/experiments/ltx_roundtrip_control.json`
- Cache runner: `src/safety_probing/ltx_roundtrip.py`
- Evaluator: `src/safety_probing/ltx_roundtrip_evaluate.py`
- Tests: `tests/test_ltx_roundtrip.py`
- GT cache: `artifacts/features/ltx_2_3_full_gt_roundtrip_future.pt`
- Generated cache: `artifacts/features/ltx_2_3_full_generated_roundtrip_future.pt`
- Full result: `outputs/ltx_roundtrip_control/full_prompt_v1/result.json`

이 문서는 새 실험의 독립 기록이다. 현재 `/home/jude/CWM_paper/Safety_Probing_paper/main.tex`과 본문 결과에는
이 실험을 인용하거나 통합하지 않는다.
