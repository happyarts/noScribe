"""Migrations only this fork needs, kept out of the upstream pull requests: its
users saved Voxtral builds under names that changed, and set Voxtral's options
in config.yml before the plugins had settings files of their own."""
from types import SimpleNamespace

import pytest

from noScribe.models import ModelDescriptor, ModelRef

m = pytest.importorskip("noScribe.main")


def test_a_renamed_build_finds_its_new_name():
    from noScribe.inference import LocalWorkerSettings
    from noScribe.plugins.factory import create_builtin_registry
    registry = create_builtin_registry({"precise": "models/precise"}, LocalWorkerSettings())
    plugin = registry.get("local-voxtral")
    plugin.list_models = lambda: [ModelDescriptor(ModelRef("local-voxtral", "voxtral-small-4bit"),
                                                  "voxtral-small-4bit", "voxtral", plugin.manifest.capabilities)]
    assert str(m._model_ref("voxtral-small-8bit", registry)) == "local-voxtral:voxtral-small-4bit"


@pytest.mark.parametrize("value, number", [(120, 120.0), ("90", 90.0), ("high", 0.0), (None, 0.0), (-3, 0.0)])
def test_voxtral_options_from_config_yml_are_read_as_numbers(monkeypatch, value, number):
    monkeypatch.setattr(m, "config", {"voxtral_chunk_sec": value})
    assert m._config_number("voxtral_chunk_sec") == number
