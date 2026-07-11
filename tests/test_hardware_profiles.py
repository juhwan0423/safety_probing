from __future__ import annotations

import json
from pathlib import Path
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load(relative_path: str) -> dict:
    return json.loads((PROJECT_ROOT / relative_path).read_text())


class HardwareProfileTest(unittest.TestCase):
    def test_production_profiles_disable_cpu_offload(self) -> None:
        hardware = _load("configs/hardware_profiles.json")
        self.assertFalse(hardware["policy"]["production_cpu_offload_allowed"])
        for profile in hardware["profiles"].values():
            self.assertEqual(profile["cpu_offload_gb"], 0)

    def test_qwen_is_assigned_to_dgx_spark(self) -> None:
        hardware = _load("configs/hardware_profiles.json")
        experiment = _load("configs/experiments/qwen_3_6_direct.json")
        registry = _load("configs/model_registry.json")

        self.assertEqual(hardware["model_assignment"]["qwen_3_6"], "dgx_spark")
        self.assertEqual(experiment["execution"]["hardware_profile"], "dgx_spark")
        self.assertFalse(experiment["execution"]["cpu_offload_allowed"])
        runtime = registry["models"]["qwen_3_6"]["runtime"]
        self.assertEqual(runtime["hardware_profile"], "dgx_spark")
        self.assertEqual(runtime["cpu_offload_gb"], 0)

    def test_5090_profile_excludes_qwen(self) -> None:
        hardware = _load("configs/hardware_profiles.json")
        local = hardware["profiles"]["rtx_5090_local"]
        self.assertIn("qwen_3_6", local["disallowed_models"])
        self.assertNotIn("qwen_3_6", local["assigned_models"])


if __name__ == "__main__":
    unittest.main()
