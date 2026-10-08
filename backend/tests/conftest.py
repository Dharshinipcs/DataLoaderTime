"""Shared fixtures: the real source bundle is loaded once per test session.

The tests read the REAL files under train/data (never modified) so that the
locked validation facts are continuously revalidated against reality.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolated_monitoring_store(tmp_path, monkeypatch):
    """Keep every test OUT of the real backend/store directory.

    The prediction path persists an event on every attempt (Task 11), and most
    test modules exercise predictions — without this fixture the real store
    would fill with test pollution. Each test gets its own empty store dir;
    monitoring tests additionally use it directly via the same monkeypatched
    path.
    """
    import dltime.store

    store_dir = tmp_path / "store"
    store_dir.mkdir()
    monkeypatch.setattr(dltime.store, "STORE_DIR", store_dir)
    return store_dir


from dltime.feature_assembly import build_training_frame
from dltime.pipeline import build_foundation
from dltime.sources import load_all_sources


@pytest.fixture(scope="session")
def bundle():
    return load_all_sources()


@pytest.fixture(scope="session")
def foundation(bundle):
    return build_foundation(bundle)


@pytest.fixture(scope="session")
def tf():
    return build_training_frame()


@pytest.fixture(scope="session")
def serving_artifact(tf):
    """Build, save and reload the serving artifact once per session (Task 04)."""
    from dltime.artifact import build_serving_artifact, load_serving_artifact, save_serving_artifact

    built = build_serving_artifact(tf)
    save_serving_artifact(built)
    return load_serving_artifact()


@pytest.fixture(scope="session")
def service(tf, serving_artifact, bundle, foundation):
    """Session-wide PredictionService over the real sources (Task 04)."""
    from dltime.inference import PredictionService
    from dltime.resolution import ConnectionResolver

    resolver = ConnectionResolver(
        bundle.loader_definitions_detail, bundle.loader_definitions_summary
    )
    return PredictionService(
        training_frame=tf,
        artifact=serving_artifact,
        foundation=foundation,
        resolver=resolver,
        definitions_detail=bundle.loader_definitions_detail,
        definitions_summary=bundle.loader_definitions_summary,
    )
