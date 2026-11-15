종합 평가

현재 상태는 이 워크숍 기준으로 Weak Accept–Borderline Accept, 약 6/10 정도입니다. 주제 적합성은 매우 높고 핵심 메시지도 좋습니다. 특히 “최종 사고 여부를 어느 정도 맞히는 것”과 “그 사고를 유발하는 물리 상태를 실제로 보존하는 것”을 분리한다는 주장은 논문 전체를 끌고 갈 만큼 명확합니다. PSQ의 고정 리드아웃 비교와 ForkSafe-CF의 대응된 counterfactual pair 설계도 안전한 world model 평가라는 워크숍 주제에 정확히 맞습니다.

다만 지금 그대로 제출하면, 리뷰어가 “물리 정보를 잊은 것이 아니라 generated latent와 VAE-encoded latent 사이의 분포 차이를 측정한 것 아닌가?”라고 물을 가능성이 큽니다. 이 부분을 통제하지 않으면 논문의 가장 강한 결론과 제목이 흔들립니다.

항목	현재 평가
워크숍 적합성	5/5
문제 설정과 중요성	4.5/5
방법론적 새로움	3/5
기술적 엄밀성	3/5
실험 완결성	2.5/5
글의 명확성	4/5
종합	6/10, Weak Accept
리뷰 신뢰도	4/5

메인 ECCV 트랙 기준으로는 현재 실험 범위와 통제가 부족하지만, 이 워크숍의 포스터·프로시딩 논문으로는 충분히 경쟁력 있습니다. 다만 6편 oral이나 award 후보가 되려면 아래 핵심 통제 중 적어도 하나가 더 필요합니다.

잘된 점
1. 워크숍과의 적합성이 매우 높습니다

워크숍이 요구하는 물리적 일관성, safety-critical evaluation, counterfactual reasoning, generative stress testing, embodied systems를 거의 모두 직접 건드립니다. 단순히 “영상 생성이 물리를 잘하나”가 아니라, payload mass, lateral CoM, contact, support geometry, ZMP margin이라는 안전 의사결정에 직접 연결되는 상태량을 평가 대상으로 삼은 것도 좋습니다.

2. 논문의 중심 메시지가 명확합니다

가장 좋은 부분은 다음 구분입니다.

Outcome-correlated signal이 남아 있는 것과 intervention-sensitive physical state가 보존되는 것은 다르다.

Generated latent의 tip AUROC는 어느 정도 남아 있지만, lateral CoM 방향성과 ZMP 위험 순서는 거의 무너진다는 결과가 이 주장을 잘 받칩니다. 결과 크기도 작지 않습니다. mass MAE는 14.5→95.5 kg, CoM은 0.032→0.183 m, roll은 2.25→16.16°, ZMP margin은 0.032→0.111 m로 악화됩니다. Counterfactual CoM 방향 정확도가 0.994에서 0.586으로 떨어지는 결과도 메시지가 선명합니다.

특히 16.16° roll error는 20° tip threshold와 비교하면 물리적으로도 상당한 오차입니다. 이처럼 단위를 가진 결과는 perceptual metric보다 훨씬 설득력이 있습니다.

3. ForkSafe-CF의 paired design이 좋습니다

동일한 scene, camera, route, steering, environment, simulator seed를 유지하면서 mass 또는 lateral CoM 하나만 바꾸는 설계는 shortcut을 크게 줄입니다.

동일 endpoint나 좌우 mirror condition이 train/test를 넘지 않도록 connected-component split을 만든 것도 꼼꼼합니다. 랜덤 pair split의 누수 가능성을 인식하고 막았다는 점은 리뷰어에게 긍정적으로 보일 부분입니다.

4. 결론을 비교적 조심스럽게 쓰고 있습니다

“고정된 decoder가 실패했다고 해서 다른 decoder도 정보를 복구하지 못한다는 뜻은 아니다”라는 한계를 반복해서 명시한 것은 좋습니다. 이런 문장은 저자들이 probing 결과를 representation의 절대적 정보 부재로 과도하게 해석하지 않는다는 인상을 줍니다.

5. 글의 구조와 문장력은 전반적으로 좋습니다

Introduction → PSQ → counterfactual benchmark → fixed-readout result → outcome ranking과 physical preservation의 차이라는 흐름이 자연스럽습니다. 14페이지 본문과 2페이지 참고문헌으로 페이지 제한에도 맞습니다. 현재 글에서 독자가 핵심 문제를 이해하지 못할 가능성은 낮습니다.

수락을 막을 수 있는 핵심 문제
1. 가장 큰 문제: GT-future latent와 generated latent가 정말 비교 가능한가

현재 실험은 다음 두 입력을 같은 probe에 넣습니다.

Ground-truth future: simulator RGB를 frozen LTX VAE로 encode한 latent
Generated future: continuation model의 final denoised latent를 직접 사용

논문은 이 둘이 같은 feature coordinates를 가진다고 주장하지만, 그것만으로 충분하지 않습니다. VAE posterior에서 얻은 latent와 diffusion model이 생성한 최종 latent는 다음 이유로 다른 분포를 가질 수 있습니다.

latent scaling 및 normalization 차이
VAE posterior sample/mean과 diffusion x
0
	

 출력의 차이
prefix와 future를 temporal VAE에 넣는 방식의 차이
sequence boundary와 temporal receptive field 차이
generated latent의 단순한 affine/statistical shift
diffusion output 특유의 artifact

따라서 현재 결과는 엄밀히 말하면 physical information loss보다는 “GT-future latent에서 학습한 리드아웃과 generated latent의 비호환성”을 보여줍니다. 논문도 이를 일부 인정하지만, 제목의 “what world models forget”과 본문 일부의 “do not preserve”는 그보다 강한 주장입니다. 실험 파이프라인은 논문에 분명히 기술되어 있습니다.

반드시 추가할 통제

가장 우선순위가 높은 것은 다음입니다.

Generated latent → decode to RGB → 동일한 frozen LTX VAE로 re-encode → 동일한 frozen PSQ 적용

이 결과를 현재 direct-latent 결과와 같이 보여주세요.

direct와 re-encoded가 모두 나쁘면: 생성된 시각적·물리적 내용 자체가 손실되었다는 주장이 강해집니다.
direct만 나쁘고 re-encoded는 좋으면: 물리 손실이 아니라 latent-coordinate mismatch일 가능성이 큽니다.
별도 probe를 generated latent에 학습했을 때는 좋고 fixed probe만 나쁘면: 정보는 있으나 representation alignment가 깨진 것입니다.
generated-trained probe도 나쁘면: “forget” 주장에 가장 강한 근거가 됩니다.

시간상 하나만 할 수 있다면 decode–re-encode control이 최우선입니다. 이 통제가 없으면 전문 리뷰어가 메인 결론을 계속 공격할 수 있습니다.

또한 GT future를 VAE에 넣을 때 future 55 frames만 넣었는지, 아니면 prefix+future 전체를 넣은 뒤 future 위치를 slice했는지도 명시해야 합니다. Temporal VAE라면 이 차이가 중요합니다.

2. “frozen representation probing”이라고 하기에는 probe가 충분히 강합니다

PSQ는 단순 linear readout이 아니라 trainable Transformer decoder입니다. 특히 observation prefix에서 future roll, future ZMP, final outcome을 예측하는 경우, 이 decoder 자체가 작은 dynamics predictor를 학습할 수 있습니다.

따라서 “backbone이 frozen이므로 representation에서 물리 상태를 단순히 읽어낸다”는 설명만으로는 부족합니다. Frozen backbone이라고 해서 trainable decoder가 dynamics를 학습하지 않는 것은 아닙니다.

이 문제를 막기 위해 논문은 Section 4.6에서 다음 baseline을 하겠다고 적고 있습니다.

globally averaged linear probe
mean-pooled MLP
temporal Transformer
speed-only control
direction–speed–steering control
train-mean predictor

하지만 현재 Table 1과 Table 2에는 이 결과들이 없습니다. Balanced accuracy와 Spearman correlation도 보고하겠다고 했지만 결과표에는 없습니다.

이 상태는 리뷰어에게 “실험이 아직 덜 끝났거나, 유리한 결과만 본문에 남긴 것 같다”는 인상을 줄 수 있습니다.

수정 방향

최소한 supplementary 또는 본문 표에 다음을 넣어야 합니다.

Train-mean baseline의 실제 물리 단위 MAE
Global linear / pooled MLP / temporal Transformer
Speed-only 및 steering-control의 tip AUROC·balanced accuracy
PSQ parameter count와 layer/head/query 수
Random or temporally shuffled feature control 하나

특히 V-JEPA의 tip AUROC 0.997은 speed shortcut이 강한 데이터에서 나온 값이므로, speed-only AUROC를 바로 옆에 보여줘야 의미가 생깁니다.

3. Page 9의 split 설명과 AUROC 사이에 치명적인 불일치가 있습니다

Page 9에는 test set이

“117 definitive tips and three right-censored conditions”

라고 되어 있습니다. 그러나 Page 10에는

“117 definitive outcomes and three right-censored cases”

라고 되어 있고, Table 1은 N=117에서 tip AUROC를 계산합니다.

정말 117개가 모두 tip이라면 negative class가 없으므로 AUROC는 정의될 수 없습니다. 단순히 “tips”가 들어간 오타일 가능성이 높지만, 제출 전에 반드시 코드와 split을 대조해야 합니다.

표나 본문에 다음을 직접 적는 것이 안전합니다.

Test: N
tip
	

=X, N
recovered
	

=Y, N
censored
	

=3

이것은 사소한 문장 교정이 아니라, 발견한 리뷰어에게는 전체 결과 신뢰도를 떨어뜨릴 수 있는 문제입니다.

4. 독립적인 test component가 4개뿐입니다

120개 test condition이 있어 보이지만, 독립성 단위로 주장하는 connected component는 4개뿐입니다. ZMP-risk ordering 20쌍은 3개 component에만 걸쳐 있습니다. 논문도 이를 작은 표본 diagnostic으로 표현하지만, Table 1의 비교적 좁은 일부 신뢰구간은 C=4와 함께 보면 질문을 받을 수 있습니다.

다음이 필요합니다.

component별 결과 4개를 supplementary에 직접 표시
각 component의 speed, turn direction, tip/recovery 구성
bootstrap의 정확한 hierarchy: component / condition / probe initialization / rollout seed 중 무엇을 어떻게 resample했는지
3개의 probe initialization과 4개의 rollout seed 변동성을 별도 표시
가능하다면 repeated group split 또는 component-wise cross-validation

현재는 rollout 4개를 먼저 평균내 condition-level prediction으로 만든 뒤 평가합니다. 이는 기대값 평가로는 타당하지만, single rollout deployment 성능과 생성 불확실성을 감춥니다. 평균 성능과 함께 per-seed 성능도 보고하는 편이 좋습니다.

5. “visually plausible”이라는 핵심 전제가 체계적으로 검증되지 않았습니다

논문의 이야기 구조는 사실상 다음입니다.

시각적으로는 그럴듯하지만, 물리 상태는 보존하지 못한다.

그런데 현재 “visually plausible/coherent”는 inspected samples에 대한 서술만 있고, 평가 프로토콜이나 수치가 없습니다. 특히 “simulator-tip 조건에서도 clear rollover가 관찰되지 않았다”는 매우 중요한 관찰인데, 몇 개를 누가 어떤 기준으로 보았는지 불분명합니다.

적어도 다음 중 하나가 필요합니다.

전체 또는 충분한 subset에 대한 blind human rating
rollover visible/not visible 및 local artifact의 명시적 annotation
perceptual metric과 PSQ error 사이의 상관관계
matched GT/generated qualitative figure와 CoM·ZMP·support-margin overlay

논문이 FVD와 VBench의 한계를 논하지만 실제 실험에서 perceptual metric과 PSQ가 얼마나 다르게 판단하는지는 보여주지 않습니다. 핵심 주장을 가장 직접적으로 강화할 실험입니다.

시간이 없으면 “visually plausible”을 다음처럼 낮추는 것이 안전합니다.

“qualitatively coherent in the inspected samples”

그리고 inspected sample의 정확한 수를 적으세요.

6. 현재 prompt가 rollover를 억제하는 방향일 수 있습니다

Generated future prompt가 다음을 요청합니다.

motion continuity
return to straight driving
scene preservation

“최종 outcome을 지정하지 않는다”고 쓰여 있지만, return to straight driving은 언어 모델 관점에서 정상 회복을 암시할 수 있습니다. 실제 물리가 rollover를 요구하더라도 prompt가 safe-looking continuation을 유도했을 가능성이 있습니다.

최소한 다음 neutral prompt와 비교하는 것이 좋습니다.

“Continue the video naturally while preserving the observed scene and motion.”

가장 좋은 것은 2–3개 prompt에서 핵심 gap이 유지되는지를 보여주는 것입니다. 전체 데이터가 어려우면 대표 subset에서라도 prompt sensitivity를 제시해야 합니다.

또한 LTX는 정밀한 future action sequence를 입력받는 action-conditioned world model이라기보다 video continuation model에 가깝습니다. 향후 steering command가 숫자나 action token으로 제공되지 않는다면, absolute future-roll 오차 중 일부는 물리 추론 실패가 아니라 control-conditioning 부족일 수 있습니다. 이 점은 limitation에 명시하는 것이 좋습니다.

방법론적 새로움에 대한 예상 리뷰

PSQ는 잘 구성된 프레임워크지만, 공격적인 리뷰어는 다음처럼 요약할 수 있습니다.

“Cross-attention probe에 여러 physical head를 붙이고, paired counterfactual accuracy를 계산한 것 아닌가?”

따라서 방법의 새로움을 “새로운 query architecture”에 너무 두기보다 다음 조합에 두는 편이 낫습니다.

GT-future feature에서만 학습하는 fixed-readout transfer
숨은 intervention property부터 continuous safety margin까지 이어지는 causal state chain
동일 context의 single-variable intervention pair
outcome shortcut과 physical sensitivity를 분리하는 평가

즉, architecture novelty보다 evaluation protocol novelty로 밀어야 합니다.

현재 contribution의

“ForkSafe-CF, the first PSQ testbed”

는 다소 순환적인 주장입니다. 다음처럼 바꾸는 편이 강합니다.

“a controlled counterfactual benchmark with single-variable interventions, leakage-resistant graph splits, and metric safety ground truth”

Related Work도 현재 17편으로 비교적 얇습니다. 특히 다음 범주의 문헌을 조금 더 연결하면 novelty positioning이 좋아집니다.

action-conditioned robotic video prediction/world models
physical-consistency evaluation of generated video
causal and counterfactual representation probing
safety verification and risk-sensitive planning
probe complexity/selectivity 및 OOD probing
결과 보고에서 추가로 고쳐야 할 부분
전체 target을 보여줘야 합니다

Framework는 pose, velocity, acceleration, CoM trajectory, contacts, contact locations, support geometry, ZMP trajectory 등을 예측한다고 설명하지만, 본문 결과는 mass, CoM, roll, minimum ZMP margin, tip만 보여줍니다.

이 정도만 보고할 예정이라면 방법 설명의 범위를 줄이거나, supplementary에 다음을 모두 넣어야 합니다.

state trajectory aggregate
contact presence/location
support geometry
CoM and ZMP trajectory
max roll/min contact count
outcome metrics

현재 상태에서는 “왜 이 다섯 지표만 선택했는가”라는 질문과 cherry-picking 우려가 생길 수 있습니다.

물리적 스케일을 더 설명하면 좋습니다
95.5 kg는 mass intervention step 120 kg에 근접
0.183 m는 최대 lateral shift 0.25 m에 비해 큼
16.16°는 20° tip threshold에 근접
0.111 m ZMP error가 실제 support width에서 어느 정도인지

이러한 상대적 의미를 Results에 한 문장씩 넣으면 숫자가 훨씬 강하게 읽힙니다.

“Gold ZMP-risk ordering”은 이름이 불명확합니다

표의 행 이름을 다음처럼 바꾸는 것이 낫습니다.

ZMP-risk ordering on outcome-discordant pairs

가능하면 20개 outcome-discordant pair뿐 아니라, 유효한 연속 ZMP 차이가 있는 전체 pair에서의 ordering도 supplementary에 보고하세요.

글과 용어에서 바로 고칠 부분
real future → ground-truth future 또는 simulator future
현재 데이터는 실제 세계가 아니라 simulator ground truth입니다. “real-to-generated”는 sim-to-real로 오해될 수 있습니다.
LTX real oracle → LTX GT-future representation control
본문 설명처럼 이것은 predictive oracle이라기보다 representation-space control입니다.
“full-grid analysis across three data shards” 정의 또는 삭제
현재 처음 등장하며 과학적 의미가 불분명합니다. 내부 처리 단위라면 논문에서 제거하세요.
V-JEPA 2.1, LTX-2.3의 정확한 checkpoint/version 설명
인용된 공개 논문의 모델명과 버전 번호가 바로 대응되지 않습니다. 모델 release tag, checkpoint, resolution, frame rate, latent scaling, sampler, steps, guidance를 명시해야 합니다.
Reference 10의 연도와 arXiv identifier 일관성 점검
전체 참고문헌의 연도·버전·모델명도 한 번 확인하는 것이 좋습니다.
Section 6.2를 절반 정도 줄이는 것을 권합니다
Guided generation 아이디어는 흥미롭지만 현재는 전적으로 future work입니다. 이 공간을 decode–re-encode control, baseline, qualitative evidence에 쓰는 편이 수락 가능성을 더 높입니다.
제출 전 48시간 우선순위
반드시 처리할 P0
Page 9의 “117 definitive tips” 문제를 코드와 대조하고 class count를 명시
Generated RGB decode–re-encode control 추가
Section 4.6에서 약속한 baseline 결과를 적어도 supplementary에 공개
Test component별 결과와 bootstrap hierarchy 명시
Prompt가 rollover를 억제하는지 neutral-prompt ablation 또는 주장 완화
그다음 P1
전체 physical target 결과 supplementary 추가
Qualitative matched-pair figure와 inspection protocol 추가
Per-rollout-seed 결과 및 seed variance 추가
probe architecture·hyperparameter·loss weight 상세화
dataset/code의 anonymous release 계획 명시

네 개 rollout seed가 이미 있으므로, seed variance가 실제 error와 연관되는지 보는 실험도 고효율입니다. 이것이 잘 나오면 단순 평균 보고를 넘어 uncertainty-aware physical auditing까지 연결되어 워크숍 적합성이 더 높아집니다.

제목에 대한 의견

현재 제목은 기억에 잘 남지만, 한 모델·한 시뮬레이터·고정 리드아웃 결과로 “what video world models forget”이라고 일반화해 다소 강합니다.

안전한 대안은 다음 둘입니다.

Physical Safety Queries Reveal Fixed-Readout Gaps in Video World Model Rollouts
A Counterfactual Fixed-Readout Audit of Safety-Critical Physical State in Video World Models

현재 제목을 유지하려면 decode–re-encode 및 generated-trained probe 중 적어도 하나가 있어야 방어가 쉬워집니다.

최종 판단

이 논문의 가장 좋은 자산은 “outcome ranking is not physical preservation”이라는 선명한 과학적 메시지와, 이를 검증하는 matched intervention benchmark입니다. 반면 가장 위험한 부분은 generated native latent와 VAE-encoded GT latent의 고정 probe 차이를 곧바로 물리 상태 손실로 해석하는 것입니다. 이 한 가지를 결정적인 control로 막고, 이미 본문에서 약속한 baseline을 실제로 보여주면 워크숍 기준 solid accept 수준인 7/10 전후까지 올라갈 수 있습니다. 현재 상태로도 제출 가치는 충분하지만, oral이나 award를 목표로 한다면 latent-pathway control은 사실상 필수입니다.