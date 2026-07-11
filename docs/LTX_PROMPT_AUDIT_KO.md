# LTX Prompt Audit

## 목적

LTX rollout의 failure가 잘못된 text prompt 때문인지, 모델의 물리 continuation 능력 때문인지 분리한다.
Main safety metric은 outcome label을 제공하지 않는 동일한 neutral prompt만 사용한다. Forced outcome
prompt는 controllability upper bound이며 AUROC 입력으로 사용하지 않는다.

## Prompt v1 baseline

- Config: `configs/experiments/ltx_2_3_continuation_prompt_v1.json`
- Output: `outputs/ltx_v2/part_01`
- Completion: 200 conditions x 4 seeds = 800/800
- 문제: 실제 바닥은 blue floor + thin white grid인데 prompt가 `checkerboard`라고 기술
- 40 sampled observed-tip final-frame audit: visible rollover 0
- 주요 artifact: forklift/part duplication, 문자와 기호, checkerboard 변형, object fragmentation

이 audit은 표본 검사이며 800개 전체에 대한 exhaustive visible-tip detector 결과는 아니다.

## Action-aware neutral candidate

- Active config: `configs/experiments/ltx_2_3_continuation.json`
- Pilot output: `outputs/ltx_prompt_v2_pilot`
- Selection: observed-tip 2, recovered 2, right-censored 2; seed 17
- Completion: 6/6
- Visible rollover: 0/6
- 결과: 바닥 설명은 정확해졌지만 기존 duplication/text/background artifact가 사라지지 않음

따라서 corrected prompt 전체 800 rollout은 아직 실행하지 않는다.

## 다음 판정

동일 prefix에 대해 다음 세 prompt를 비교한다.

1. `neutral`: outcome을 강제하지 않는 현재 candidate
2. `forced-stable`: upright continuation을 명시
3. `forced-tip`: 회전 바깥쪽 visible rollover를 명시

- forced-tip도 실패: LTX가 이 scene에서 rollover를 렌더링하지 못함
- forced-tip만 성공: 렌더링은 가능하지만 neutral future inference가 실패
- neutral risk가 label과 연동: pixel-generative safety prediction 가능성 확인

Stable/tip oracle 결과는 생성능력 진단에만 사용하고 main prediction table에는 넣지 않는다.

현재 candidate는 향후 steering schedule을 text로 제공한다. 이를 main protocol로 채택할 경우 Qwen에는
동일한 action text를, V-JEPA predictor에는 동일 command embedding을 제공해야 한다. 다른 두 arm이
video-only라면 이 LTX 결과는 `action-conditioned` ablation으로 별도 보고한다.
