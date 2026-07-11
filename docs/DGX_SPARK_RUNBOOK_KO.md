# DGX Spark Qwen Runbook

## 목적

Qwen3.6-27B-FP8 direct-semantic arm을 DGX Spark에서 full-model load로 실행한다. RTX 5090에서
CPU offload를 사용하는 경로는 production에서 금지한다. DGX Spark는 ARM64 Grace Blackwell과
128GB unified memory를 사용하므로 CPU/GPU 사이에 model layer를 수동 분할하지 않는다.

- Hardware policy: `configs/hardware_profiles.json`
- Experiment config: `configs/experiments/qwen_3_6_direct.json`
- Server launcher: `scripts/start_qwen_vllm_dgx_spark.sh`
- NVIDIA Spark vLLM guide: <https://build.nvidia.com/spark/vllm>
- NVIDIA DGX Spark architecture: <https://docs.nvidia.com/dgx/dgx-spark/system-overview.html>
- NGC vLLM image: <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm/tags>

## DGX Spark 최초 준비

```bash
git clone https://github.com/juhwan0423/safety_probing.git
cd safety_probing

# Qwen download manager와 smoke client에 필요한 최소 환경이다.
uv sync --locked --extra model-prep
.venv/bin/safety-models download \
  --registry configs/model_registry.json --model qwen_3_6

docker pull nvcr.io/nvidia/vllm:26.06-py3
```

`docker ps`가 permission denied이면 NVIDIA guide에 따라 사용자를 `docker` group에 추가한 뒤 새 shell로
들어간다. 모델은 약 31GB이므로 container image와 checkpoint를 위한 여유 disk를 먼저 확인한다.

## Server 실행

```bash
cd ~/safety_probing
scripts/start_qwen_vllm.sh
```

Launcher는 다음을 강제한다.

- ARM64 host가 아니면 종료
- NVIDIA vLLM `26.06-py3` ARM64 image
- `gpu-memory-utilization=0.80`
- `cpu-offload-gb=0`
- Qwen context length 4096, video 1개, server port 8016

다른 image를 검증할 때만 `VLLM_IMAGE`를 명시적으로 override한다.

```bash
VLLM_IMAGE=nvcr.io/nvidia/vllm:26.06-py3 scripts/start_qwen_vllm.sh
```

## Workstation에서 호출

DGX Spark의 hostname 또는 LAN IP를 사용한다. 이 port는 인증이 없으므로 공용 인터넷에 노출하지 않는다.

```bash
export QWEN_VLLM_SERVER_URL=http://dgx-spark.local:8016
curl --fail "$QWEN_VLLM_SERVER_URL/health"

cd ~/isaac_ws/safety_probing
.venv/bin/safety-qwen-vllm-smoke --outcome confirmed_recovered
.venv/bin/safety-qwen-vllm-smoke --outcome observed_tip
```

Client는 MP4를 base64 data URI로 request에 포함한다. 따라서 DGX Spark에 RGB dataset을 복사하거나 두
host에서 동일한 absolute path를 만들 필요가 없다.

## 검증 기준

1. `/health`가 HTTP 200을 반환한다.
2. server log에서 CPU offload가 0이고 model load가 완료됐는지 확인한다.
3. 두 smoke request가 schema-valid JSON을 반환한다.
4. latency, peak unified-memory usage, prompt/completion token 수를 runtime report에 기록한다.
5. 성공 후 `docs/MODEL_RUNTIME_STATUS_KO.md`의 DGX 상태를 `verified`로 갱신한다.
