"""Generation and evaluation contracts; fixtures do not measure translation quality."""

from __future__ import annotations

import subprocess
import sys

import pytest

from backend.app.evaluation.hybrid_eval import image_quality, language_quality, translation_quality
from backend.app.retrieval.hybrid import marian_onnx
from backend.tests import test_image_quality_protocol


def _translator(*, tokens=(7, 0), count=4):
    import numpy as np

    class Tokenizer:
        def __call__(self, text, **kwargs):
            assert kwargs == {"return_tensors": "np", "truncation": False}
            return {
                "input_ids": np.ones((1, count), dtype=np.int64),
                "attention_mask": np.ones((1, count), dtype=np.int64),
            }

        def decode(self, ids, **kwargs):
            assert ids in ([], [7]) and kwargs == {"skip_special_tokens": True}
            return "generated fixture" if ids else ""

    class Encoder:
        calls = 0

        def run(self, names, feeds):
            self.calls += 1
            assert names == ["last_hidden_state"] and set(feeds) == {"input_ids", "attention_mask"}
            return [np.zeros((1, count, 512))]

    class Decoder:
        seen = []

        def run(self, names, feeds):
            assert names == ["logits"]
            self.seen.append(feeds["input_ids"].tolist()[0])
            length = feeds["input_ids"].shape[1]
            scores = np.zeros((1, length, 65001))
            scores[0, -1, 65000] = 100  # Native forbidden pad must never win.
            scores[0, -1, tokens[min(length - 1, len(tokens) - 1)]] = 10
            return [scores]

    translator = marian_onnx.MarianOnnxTranslator.__new__(marian_onnx.MarianOnnxTranslator)
    translator.tokenizer, translator.encoder, translator.decoder = Tokenizer(), Encoder(), Decoder()
    translator.version, translator.max_tokens = "fixture-version", 3
    return translator


def test_greedy_honors_start_forbidden_pad_natural_eos_and_provenance():
    translator = _translator()
    result = translator.translate("原查询")
    assert result == marian_onnx.Translation("原查询", "generated fixture", "fixture-version", 4, 2)
    assert translator.decoder.seen == [[65000], [65000, 7]]


@pytest.mark.parametrize("value", [None, "", " ", "中" * 2049])
def test_invalid_query_is_rejected_before_model_calls(value):
    translator = _translator()
    with pytest.raises(ValueError, match="query"):
        translator.translate(value)
    assert translator.encoder.calls == 0


def test_input_budget_and_output_exhaustion_do_not_return_partial_text():
    translator = _translator(count=257)
    with pytest.raises(ValueError, match="input budget"):
        translator.translate("原查询")
    assert translator.encoder.calls == 0
    translator = _translator(tokens=(7,))
    with pytest.raises(ValueError, match="without EOS"):
        translator.translate("原查询")
    assert len(translator.decoder.seen) == 3
    with pytest.raises(ValueError, match="empty"):
        _translator(tokens=(0,)).translate("原查询")


def test_provider_failures_keep_queries_in_denominator_and_share_only_one_test_image_pass(tmp_path):
    import numpy as np

    source = test_image_quality_protocol.fixture_source(tmp_path)

    class Translator:
        version = "fixture"
        seen = []

        def translate(self, text):
            self.seen.append(text)
            if text == language_quality.CHINESE[4]:
                raise RuntimeError("explicit provider failure")
            return marian_onnx.Translation(text, "generated:" + text, self.version, 4, 3)

    class Encoder(test_image_quality_protocol.ScoreFixture):
        prompts = []

        def encode_texts(self, texts):
            self.prompts.extend(texts)

            def index(text):
                if text in image_quality.PROMPTS:
                    return image_quality.PROMPTS.index(text)
                return language_quality.CHINESE.index(text.removeprefix("generated:"))

            return np.eye(10)[[index(text) for text in texts]]

    model, translator = Encoder(source), Translator()
    result = translation_quality.evaluate(source, model, translator)
    assert translator.seen == list(language_quality.CHINESE)
    assert model.partitions_seen == ["test"] * 20
    assert len(model.prompts) == 29 and "" not in model.prompts
    assert result["failed_queries"] == 1
    assert result["rankings"]["machine_translation"]["deer"] == ()
    assert result["groups"]["machine_translation"]["metrics"]["positive"]["hit_at_10"] == pytest.approx(0.9)
    assert result["translations"][4]["error"]["type"] == "RuntimeError"
    assert result["groups"]["english_reference"]["metrics"]["positive"]["hit_at_10"] == 1


def test_provenance_mismatch_is_a_failed_translation_not_a_reference_fallback():
    class WrongProvider:
        version = "fixture"

        def translate(self, text):
            return marian_onnx.Translation("wrong input", "wrong output", "fixture", 1, 1)

    prompts, rows = translation_quality._translations(WrongProvider())
    assert prompts == [None] * 10 and all(row["status"] == "failed" for row in rows)


def test_nonfinite_model_scores_fail_instead_of_producing_a_plausible_ranking(tmp_path):
    import numpy as np

    source = test_image_quality_protocol.fixture_source(tmp_path)

    class BrokenEncoder:
        def encode_texts(self, texts):
            return np.full((len(texts), 2), np.nan)

    with pytest.raises(ValueError, match="invalid aligned scores"):
        translation_quality._group(
            source.partition("test"), np.ones((20, 2)), language_quality.CHINESE, BrokenEncoder()
        )


def test_optional_imports_and_help_do_not_require_model_dependencies():
    subprocess.run(
        [sys.executable, "-S", "-m", "backend.app.evaluation.translation_benchmark", "--help"],
        check=True,
        capture_output=True,
        timeout=15,
    )


def test_unknown_generation_rules_fail_closed():
    generation = {
        "decoder_start_token_id": 65000,
        "eos_token_id": 0,
        "pad_token_id": 65000,
        "bos_token_id": 0,
        "forced_eos_token_id": 0,
        "bad_words_ids": [[65000]],
        "num_beams": 6,
        "max_length": 512,
        "renormalize_logits": True,
    }
    config = {**generation, "model_type": "marian", "vocab_size": 65001, "d_model": 512}
    marian_onnx.MarianOnnxTranslator._check_profile(config, generation)
    with pytest.raises(ValueError, match="generation options"):
        marian_onnx.MarianOnnxTranslator._check_profile(config, {**generation, "forced_bos_token_id": 17})
