"""Fixed bilingual and paraphrase class-prompt coverage, without prompt selection."""

from __future__ import annotations

import time

from backend.app.evaluation import dataset as evaluation_dataset
from backend.app.evaluation.hybrid_eval import image_quality, mirror_images, reporting
from backend.app.retrieval.hybrid import vision_reranking

CHINESE = (
    "一张飞机的照片。",
    "一张小汽车的照片。",
    "一张鸟的照片。",
    "一张猫的照片。",
    "一张鹿的照片。",
    "一张狗的照片。",
    "一张青蛙的照片。",
    "一张马的照片。",
    "一张船的照片。",
    "一张卡车的照片。",
)
PARAPHRASES = (
    "an aircraft",
    "a passenger car",
    "a feathered bird",
    "a domestic feline",
    "an antlered deer",
    "a domestic canine",
    "a small frog amphibian",
    "an equine animal",
    "a large boat",
    "a cargo lorry",
)
PROMPT_GROUPS = {"english_original": image_quality.PROMPTS, "chinese": CHINESE, "english_paraphrase": PARAPHRASES}


def evaluate(source: mirror_images.ImageDataset, model: vision_reranking.ImageTextEncoder) -> reporting.JsonObject:
    """Actually encode200 test images and all30 frozen text prompts once.

    Fixed translations/paraphrases share dominant-class qrels, not exact scene
    descriptions or independent language relevance annotations. Every per-prompt
    result and ranking is retained; there is no best-prompt or best-language gate.
    """
    items = source.partition("test")
    ids = [item.identifier for item in items]
    started = time.perf_counter()
    vectors = model.encode_images([source.directory / item.filename for item in items])
    image_seconds = time.perf_counter() - started
    results, all_rankings = {}, {}
    for group, prompts in PROMPT_GROUPS.items():
        started = time.perf_counter()
        scores = vectors @ model.encode_texts(prompts).T
        seconds = time.perf_counter() - started
        cases, rankings, rows = [], {}, []
        for column, (label, prompt) in enumerate(zip(mirror_images.CLASSES, prompts)):
            case = evaluation_dataset.EvaluationCase(
                label, prompt, {item.identifier for item in items if item.label == label}, label
            )
            ranked = tuple(
                ids[index]
                for index in sorted(range(len(items)), key=lambda index: (-float(scores[index, column]), ids[index]))[
                    :10
                ]
            )
            cases.append(case)
            rankings[label] = ranked
            rows.append({"label": label, "prompt": prompt, "metrics": image_quality._metrics([case], {label: ranked})})
        results[group] = {
            "metrics": image_quality._metrics(cases, rankings),
            "text_seconds": seconds,
            "per_prompt": rows,
        }
        all_rankings[group] = rankings
    return {
        "schema": "lentica-vision-language-v1",
        "dataset_identity": source.identity,
        "encoder_id": model.encoder_id,
        "images": len(items),
        "image_encoding_seconds": image_seconds,
        "groups": results,
        "rankings": all_rankings,
        "ranking_sha256": reporting.digest(all_rankings),
        "scope": "Fixed30 known-class prompts on fresh image sources; class qrels only, no prompt tuning, arbitrary language/relations/open-world or generalization claim.",
    }
