"""Capability routing, correlated evidence and multimodal input boundaries."""

from __future__ import annotations

import subprocess
import sys

import pytest

from backend.app.retrieval import hybrid
from backend.app.retrieval.hybrid import families, media, querying, routing, stages, units, vector


def test_family_cloning_cannot_add_votes_and_undeclared_routes_stay_independent():
    config = hybrid.SearchConfig(tuple(hybrid.Branch(name, 10) for name in ("caption", "tags", "native")))
    rankings = {"caption": [2, 1], "tags": [2, 1], "native": [1, 2]}
    grouped = families.FamilyRrfFusion({"caption": "generated", "tags": "generated"}).fuse(rankings, config)
    without_clone = families.FamilyRrfFusion({"caption": "generated"}).fuse(
        rankings, hybrid.SearchConfig(tuple(branch for branch in config.branches if branch.name != "tags"))
    )
    assert grouped == without_clone
    assert len(grouped[0].contributions) == 2
    assert grouped[0].asset_id == 1  # Equal independent evidence uses the ID tie.
    assert families.FamilyRrfFusion({}).fuse(rankings, config) == hybrid.RrfFusion().fuse(rankings, config)


def test_family_and_undeclared_branch_names_do_not_collide():
    config = hybrid.SearchConfig((hybrid.Branch("a"), hybrid.Branch("generated")))
    hit = families.FamilyRrfFusion({"a": "generated"}).fuse({"a": [1], "generated": [1]}, config)[0]
    assert len(hit.contributions) == 2
    with pytest.raises(ValueError):
        families.FamilyRrfFusion({"a": ""})


def test_planner_records_reasons_and_allocates_only_within_budget():
    routes = (
        querying.RouteSpec(hybrid.Branch("short", 1)),
        querying.RouteSpec(hybrid.Branch("long", 9)),
        querying.RouteSpec(hybrid.Branch("image"), kinds=frozenset({"image"})),
        querying.RouteSpec(hybrid.Branch("missing"), state="missing_modality"),
        querying.RouteSpec(hybrid.Branch("pending"), state="not_indexed"),
        querying.RouteSpec(hybrid.Branch("disabled", weight=0)),
    )
    plan = querying.plan(querying.MultimodalQuery("hello"), routes, querying.QueryBudget(total_candidates=5))
    assert [(item.window, item.status) for item in plan] == [
        (1, "planned"),
        (4, "planned"),
        (0, "unsupported_query"),
        (0, "missing_modality"),
        (0, "not_indexed"),
        (0, "disabled"),
    ]
    small = querying.plan(querying.MultimodalQuery("hello"), routes[:2], querying.QueryBudget(total_candidates=1))
    assert [item.status for item in small] == ["planned", "budget_disabled"]
    assert querying.classify(querying.MultimodalQuery("蓝色")) == "color"
    assert querying.classify(querying.MultimodalQuery("fix ERR_42")) == "identifier"
    assert querying.classify(querying.MultimodalQuery("red ERR_42")) == "identifier"


def test_actual_image_query_uses_pixels_filters_and_one_final_asset_fetch(tmp_path):
    from PIL import Image

    paths = {identifier: tmp_path / f"{identifier}.png" for identifier in (1, 2, 3)}
    for identifier, color in ((1, "red"), (2, "blue"), (3, "red")):
        Image.new("RGB", (20, 20), color).save(paths[identifier])
    catalog = units.UnitCatalog([units.RetrievalUnit(key, key + 100, "image") for key in paths])
    pixel = media.ColorHistogramRetriever(catalog, paths)

    class Store:
        calls = []

        def fetch(self, ids):
            self.calls.append(tuple(ids))
            return dict.fromkeys(ids, "asset")

    class NeverText:
        def retrieve(self, query, limit):
            pytest.fail("Image input must not invoke an unsupported text provider")

    store = Store()
    providers = {"pixels": media.ImageHistogramRoute(pixel), "text": NeverText()}
    specs = [
        querying.RouteSpec(hybrid.Branch("pixels"), kinds=frozenset({"image", "combined"}), family="visual"),
        querying.RouteSpec(hybrid.Branch("text"), kinds=frozenset({"text", "identifier"})),
    ]
    config = stages.StageConfig(hybrid.SearchConfig(tuple(spec.branch for spec in specs), top_k=1))
    request = querying.MultimodalQuery(image=paths[1], filters=hybrid.SearchFilter(allowed_ids=frozenset({2, 3})))
    result = routing.RoutedSearch(providers, catalog, routes=specs, store=store).search(request, config)
    assert result.kind == "image"
    assert result.result.hits[0].asset_id == 103 and store.calls == [(103,)]
    assert result.result.candidates[0].asset_id == 3
    assert pixel.retrieve_image(paths[1], hybrid.SearchFilter(allowed_ids=frozenset()), 10) == ()
    assert pixel.retrieve_image(paths[1], hybrid.SearchFilter(), 0) == ()
    with pytest.raises(ValueError, match="pixel budget"):
        querying.QueryBudget(max_image_pixels=399).validate(request)


def test_combined_request_reaches_provider_intact_without_adding_image_vectors(tmp_path):
    from PIL import Image

    path = tmp_path / "image.png"
    Image.new("RGB", (4, 4)).save(path)
    query = querying.MultimodalQuery("error ERR_3", path)
    calls = []

    class Provider:
        def retrieve(self, actual, limit):
            calls.append(actual)
            return (7,)

    spec = querying.RouteSpec(hybrid.Branch("combined"), kinds=frozenset({"combined"}))
    catalog = units.UnitCatalog([units.RetrievalUnit(7, 70, "image")])
    config = stages.StageConfig(hybrid.SearchConfig((spec.branch,), top_k=1))
    actual = routing.RoutedSearch({"combined": Provider()}, catalog, routes=[spec]).search(query, config)
    assert calls == [query] and calls[0] is query
    assert actual.result.hits[0].asset_id == 70


def test_no_capable_route_does_not_fetch_or_construct_fake_hits():
    spec = querying.RouteSpec(hybrid.Branch("ocr"), state="not_indexed")
    config = stages.StageConfig(hybrid.SearchConfig((spec.branch,)))
    search = routing.RoutedSearch({}, units.UnitCatalog([]), routes=[spec])
    result = search.search(querying.MultimodalQuery("hello"), config)
    assert result.result is None and result.plan[0].status == "not_indexed"
    with pytest.raises(ValueError, match="character budget"):
        search.search(querying.MultimodalQuery("hello"), config, budget=querying.QueryBudget(max_characters=4))
    with pytest.raises(ValueError):
        querying.MultimodalQuery()


def test_new_query_modules_import_without_optional_dependencies():
    subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            """
import sys
from backend.app.retrieval.hybrid import families, querying, routing
assert not any(name.startswith(('numpy', 'PIL', 'torch', 'sqlalchemy')) for name in sys.modules)
""",
        ],
        check=True,
        timeout=10,
    )


def test_image_vector_route_requires_space_identity_and_keeps_filters():
    index = vector.ExactVectorRetriever(
        [1, 2], [[1, 0], [0, 1]], lambda text: [1, 0], encoder_id="aligned-v1", query_encoder_id="aligned-v1"
    )
    route = routing.ImageVectorRoute(index, lambda path: [0, 1], image_encoder_id="aligned-v1")
    query = querying.MultimodalQuery(image="explicit.png", filters=hybrid.SearchFilter(allowed_ids=frozenset({1})))
    assert route.retrieve(query, 1) == (1,)
    assert route.retrieve(querying.MultimodalQuery("text only"), 1) == ()
    with pytest.raises(ValueError, match="same aligned space"):
        routing.ImageVectorRoute(index, lambda path: [0, 1], image_encoder_id="unrelated-v1")
