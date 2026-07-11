from __future__ import annotations

import base64
from pathlib import Path
import tempfile
import unittest

from safety_probing.adapters.qwen_direct import parse_prediction
from safety_probing.adapters.qwen_vllm import encode_video_url


class QwenPredictionParserTest(unittest.TestCase):
    def test_parses_json_after_thinking_block(self) -> None:
        result = parse_prediction(
            '<think>hidden reasoning</think>\n'
            '{"predicted_outcome":"observed_tip","tip_probability":0.75}'
        )
        self.assertEqual(result["predicted_outcome"], "observed_tip")
        self.assertEqual(result["tip_probability"], 0.75)

    def test_rejects_invalid_probability(self) -> None:
        with self.assertRaises(ValueError):
            parse_prediction(
                '{"predicted_outcome":"confirmed_recovered",'
                '"tip_probability":1.2}'
            )

    def test_encodes_video_for_remote_dgx_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "prefix.mp4"
            video.write_bytes(b"test-video")
            result = encode_video_url(video)
        prefix, payload = result.split(",", 1)
        self.assertEqual(prefix, "data:video/mp4;base64")
        self.assertEqual(base64.b64decode(payload), b"test-video")

    def test_rejects_unknown_media_transport(self) -> None:
        with self.assertRaises(ValueError):
            encode_video_url("prefix.mp4", "shared_memory_magic")


if __name__ == "__main__":
    unittest.main()
