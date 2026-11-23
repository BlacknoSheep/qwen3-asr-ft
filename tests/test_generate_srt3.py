"""前景过滤与空窗分段测试，不加载 ASR 模型。"""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts import generate_srt3 as srt


def tone(duration: float, dbfs: float) -> np.ndarray:
    samples = np.arange(round(duration * srt.SAMPLE_RATE))
    return (10 ** (dbfs / 20) * np.sin(2 * np.pi * 200 * samples / srt.SAMPLE_RATE)).astype(np.float32)


class ForegroundTests(unittest.TestCase):
    def detect(self, audio, padding=0, merge_gap=0):
        return srt.detect_foreground(audio, -32, padding, merge_gap)

    def test_background_singing_is_filtered(self):
        self.assertEqual(self.detect(tone(1, -34.7)), [])

    def test_quiet_speech_is_kept(self):
        audio = tone(1, -28.7)
        self.assertEqual(self.detect(audio), [(0, len(audio))])

    def test_empty_silence_and_short_audio(self):
        for length in (0, 1, 159, 799, 800, 16000):
            with self.subTest(length=length):
                self.assertEqual(self.detect(np.zeros(length, dtype=np.float32)), [])
        self.assertEqual(self.detect(tone(0.04, -20)), [])

    def test_isolated_impulse_does_not_trigger(self):
        for position in (0, 159, 160, 8000, 15999):
            audio = np.zeros(16000, dtype=np.float32)
            audio[position] = 1
            with self.subTest(position=position):
                self.assertEqual(self.detect(audio), [])

    def test_padding_and_input_unchanged(self):
        audio = np.zeros(16000, dtype=np.float32)
        audio[6400:9600] = tone(0.2, -28.7)
        original = audio.copy()
        start, end = self.detect(audio)[0]
        self.assertEqual(self.detect(audio, padding=2400), [(start - 2400, end + 2400)])
        np.testing.assert_array_equal(audio, original)

    def test_merge_gap(self):
        audio = np.zeros(16000, dtype=np.float32)
        audio[3200:4800] = tone(0.1, -28.7)
        audio[8000:9600] = tone(0.1, -28.7)
        separate = self.detect(audio)
        self.assertEqual(len(separate), 2)
        gap = separate[1][0] - separate[0][1]
        self.assertEqual(self.detect(audio, merge_gap=gap), [(separate[0][0], separate[1][1])])
        self.assertEqual(self.detect(audio, merge_gap=gap - 1), separate)

    def test_boundary_and_partial_tail(self):
        audio = tone(0.123, -28.7)
        self.assertEqual(self.detect(audio, padding=2400), [(0, len(audio))])
        self.assertEqual(self.detect(tone(0.05, -28.7)), [(0, 800)])

    def test_background_does_not_extend_foreground_indefinitely(self):
        audio = tone(2, -34.7)
        audio[8000:11200] = tone(0.2, -28.7)
        intervals = self.detect(audio, padding=2400, merge_gap=2400)
        self.assertEqual(len(intervals), 1)
        self.assertGreater(intervals[0][0], 0)
        self.assertLess(intervals[0][1], srt.SAMPLE_RATE)

    def test_window_intersection_is_half_open(self):
        windows = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 45)]
        self.assertEqual(srt.foreground_window_mask(windows, [(10, 20), (35, 40)]), [False, True, False, True, False])
        self.assertEqual(srt.foreground_window_mask(windows, []), [False] * 5)
        self.assertEqual(srt.foreground_window_mask([], [(0, 1)]), [])

    def test_pipeline_filter_and_disable_switch(self):
        audio = tone(1, -34.7)
        audio[8000:11200] = tone(0.2, -28.7)
        for disabled in (False, True):
            with self.subTest(disabled=disabled), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "test.srt"
                argv = ["generate_srt3", "--audio_path", "unused.wav", "--output_path", str(output), "--debug"]
                if disabled:
                    argv.append("--no_foreground_filter")

                def transcribe(audio, intervals, *args, **kwargs):
                    return ["测试"] * len(intervals)

                with (
                    patch("sys.argv", argv),
                    patch.object(srt, "load_audio", return_value=audio),
                    patch.object(srt, "load_model", return_value=(None, None)),
                    patch.object(srt, "transcribe", side_effect=transcribe) as mocked,
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    srt.main()
                debug = json.loads(output.with_suffix(".debug.json").read_text())
                self.assertEqual(len(mocked.call_args_list[0].args[1]), 4 if disabled else 3)
                self.assertEqual(debug["windows"][0]["filtered"], not disabled)
                self.assertEqual(debug["windows"][0]["foreground"], False)
                self.assertAlmostEqual(debug["windows"][0]["peak_dbfs"], -34.7, places=4)
                self.assertEqual(debug["windows"][0]["text"], "测试" if disabled else "")
                self.assertIn("测试", output.read_text())

    def test_all_background_skips_probe_intervals(self):
        audio = tone(1, -34.7)
        windows = [(0, 4000), (4000, 8000), (8000, 12000), (12000, 16000)]
        self.assertEqual(srt.foreground_window_mask(windows, self.detect(audio)), [False] * 4)


class SegmentTests(unittest.TestCase):
    def segments(self, texts, min_silence_s=0.5):
        step = srt.SAMPLE_RATE // 4
        records = [(i * step, (i + 1) * step, text) for i, text in enumerate(texts)]
        return srt.find_segments(records, round(min_silence_s * srt.SAMPLE_RATE))

    def test_short_gap_is_kept(self):
        self.assertEqual(self.segments(["甲", "", "乙"]), [(0, 12000)])

    def test_gap_at_threshold_splits(self):
        self.assertEqual(self.segments(["甲", "", "", "乙"]), [(0, 4000), (12000, 16000)])

    def test_long_gap_splits(self):
        self.assertEqual(self.segments(["甲", "", "", "", "乙"]), [(0, 4000), (16000, 20000)])

    def test_leading_and_trailing_empty_windows_are_trimmed(self):
        for trailing in (1, 2, 3):
            with self.subTest(trailing=trailing):
                self.assertEqual(self.segments(["", "", "甲"] + [""] * trailing), [(8000, 12000)])

    def test_empty_and_all_empty_records(self):
        self.assertEqual(self.segments([]), [])
        self.assertEqual(self.segments(["", "", ""]), [])

    def test_non_multiple_threshold_rounds_up_to_window_boundary(self):
        self.assertEqual(self.segments(["甲", "", "", "乙"], 0.6), [(0, 16000)])
        self.assertEqual(self.segments(["甲", "", "", "", "乙"], 0.6), [(0, 4000), (16000, 20000)])

    def test_threshold_at_or_below_resolution_restores_single_empty_split(self):
        for threshold in (0.1, 0.25):
            with self.subTest(threshold=threshold):
                self.assertEqual(self.segments(["甲", "", "乙"], threshold), [(0, 4000), (8000, 12000)])

    def test_empty_counter_resets_after_text(self):
        self.assertEqual(self.segments(["甲", "", "乙", "", "丙"]), [(0, 20000)])

    def test_partial_final_window_uses_actual_duration(self):
        records = [(0, 4000, "甲"), (4000, 8000, ""), (8000, 9000, "")]
        self.assertEqual(srt.find_segments(records, 8000), [(0, 4000)])

    def test_pipeline_uses_min_silence_argument(self):
        for threshold, expected in (
            (0.25, [[0, 4000], [8000, 12000], [20000, 24000]]),
            (0.5, [[0, 12000], [20000, 24000]]),
            (0.75, [[0, 24000]]),
        ):
            with self.subTest(threshold=threshold), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "test.srt"
                argv = [
                    "generate_srt3",
                    "--audio_path",
                    "unused.wav",
                    "--output_path",
                    str(output),
                    "--min_silence_s",
                    str(threshold),
                    "--debug",
                ]
                with (
                    patch("sys.argv", argv),
                    patch.object(srt, "load_audio", return_value=tone(1.5, -28.7)),
                    patch.object(srt, "load_model", return_value=(None, None)),
                    patch.object(
                        srt,
                        "transcribe",
                        side_effect=[
                            ["甲", "", "乙", "", "", "丙"],
                            ["测试"] * len(expected),
                        ],
                    ),
                    contextlib.redirect_stdout(io.StringIO()),
                ):
                    srt.main()
                debug = json.loads(output.with_suffix(".debug.json").read_text())
                self.assertEqual(debug["raw_segments"], expected)
                self.assertEqual(debug["min_silence_s"], threshold)

    def test_min_silence_argument(self):
        base = ["generate_srt3", "--audio_path", "unused.wav", "--output_path", "unused.srt"]
        with patch("sys.argv", base):
            self.assertEqual(srt.parse_args().min_silence_s, 0.5)
        with patch("sys.argv", base + ["--min_silence_s", "0.75"]):
            self.assertEqual(srt.parse_args().min_silence_s, 0.75)
        for value in ("0", "-0.1", "nan", "inf", "0.000001"):
            with (
                self.subTest(value=value),
                patch("sys.argv", base + [f"--min_silence_s={value}"]),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                srt.parse_args()


if __name__ == "__main__":
    unittest.main()
