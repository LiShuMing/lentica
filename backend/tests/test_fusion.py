from __future__ import annotations

import unittest

from backend.app.retrieval.fusion import RetrievalHit, weighted_rrf


class FusionTests(unittest.TestCase):
    def test_rrf_uses_rank_not_incompatible_raw_score_scale(self) -> None:
        fused = weighted_rrf(
            {
                "visual": [
                    RetrievalHit(asset_id=1, rank=1, score=0.31, source="visual"),
                    RetrievalHit(asset_id=2, rank=2, score=0.30, source="visual"),
                ],
                "context": [
                    RetrievalHit(asset_id=2, rank=1, score=0.92, source="context"),
                    RetrievalHit(asset_id=1, rank=2, score=0.11, source="context"),
                ],
            },
            weights={"visual": 1.0, "context": 1.0},
        )

        self.assertEqual(fused[0].asset_id, 1)
        self.assertEqual(set(fused[0].sources), {"visual", "context"})
        self.assertEqual(set(fused[1].sources), {"visual", "context"})

    def test_weight_can_prioritize_ocr_ranking_without_using_its_cosine_value(self) -> None:
        fused = weighted_rrf(
            {
                "visual": [RetrievalHit(asset_id=1, rank=1, score=0.99, source="visual")],
                "ocr_dense": [RetrievalHit(asset_id=2, rank=1, score=0.02, source="ocr_dense")],
            },
            weights={"visual": 0.4, "ocr_dense": 1.5},
        )

        self.assertEqual(fused[0].asset_id, 2)


if __name__ == "__main__":
    unittest.main()
