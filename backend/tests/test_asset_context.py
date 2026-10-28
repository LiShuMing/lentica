from __future__ import annotations

import unittest

from backend.app.embedding.context_text_builder import build_context_vector_texts
from backend.app.models.asset_context import asset_context_from_payload
from backend.app.models.local_vision import normalize_vision_context


class AssetContextTests(unittest.TestCase):
    def test_v2_context_keeps_observations_and_inferences_separate(self) -> None:
        raw = """{
          "schema_version": "photo_context_v2",
          "observations": {
            "caption": "一个孩子在地铁站拿着玩具车",
            "short_caption": "地铁站里的孩子",
            "objects": ["玩具车"],
            "actions": ["拿着玩具车"],
            "scenes": ["地铁站"],
            "visible_text": ["Exit"],
            "relationships": [],
            "colors": [], "visual_attributes": [], "logos": [], "notable_regions": []
          },
          "inferences": {
            "possible_locations": [{"value": "杭州", "confidence": 0.42}],
            "possible_events": [{"value": "旅行", "confidence": 0.87}],
            "possible_time_periods": [], "possible_relationships": []
          },
          "raw_ocr_text": "Exit",
          "normalized_ocr_text": "Exit"
        }"""

        payload = normalize_vision_context(raw)
        parsed = asset_context_from_payload(payload)
        texts = build_context_vector_texts(parsed)

        self.assertEqual(payload["schema_version"], "photo_context_v2")
        self.assertEqual(parsed.observations.objects, ["玩具车"])
        self.assertEqual(parsed.inferences.possible_locations[0].confidence, 0.42)
        self.assertIn("旅行", texts.context_text)
        self.assertNotIn("杭州", texts.context_text)
        self.assertEqual(texts.ocr_text, "Exit")

    def test_legacy_context_is_projected_as_low_confidence_inference(self) -> None:
        parsed = asset_context_from_payload(
            {
                "caption": "海边日落",
                "objects": ["海", "沙滩"],
                "location": "海边",
                "ocr_text": "Sunset Cafe",
            }
        )
        texts = build_context_vector_texts(parsed)

        self.assertEqual(parsed.observations.short_caption, "海边日落")
        self.assertEqual(parsed.inferences.possible_locations[0].confidence, 0.4)
        self.assertNotIn("海边", texts.context_text)
        self.assertEqual(texts.ocr_text, "Sunset Cafe")


if __name__ == "__main__":
    unittest.main()
