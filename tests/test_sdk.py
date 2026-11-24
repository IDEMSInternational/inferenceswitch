"""Which SDK each provider needs (ProviderSpec.sdk) — offline, nothing imported."""
from __future__ import annotations

import re
import tomllib
from pathlib import Path

from inferenceswitch import SDK, default_registry
from inferenceswitch.registry import SDK_BY_KIND

ROOT = Path(__file__).resolve().parents[1]


def test_every_provider_names_its_sdk():
    sdks = {spec.name: spec.sdk for spec in default_registry()}
    assert sdks["anthropic"] == SDK(extra="anthropic", modules=("anthropic",))
    assert sdks["gemini"] == SDK(extra="gemini", modules=("google.genai",))
    # OpenAI-compatible providers, hosted or local, all drive the openai SDK.
    for name in ("openai", "mistral", "deepseek", "groq", "ollama", "lmstudio"):
        assert sdks[name].extra == "openai"


def test_each_extra_exists_and_installs_its_sdk():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"][
        "optional-dependencies"
    ]
    distributions = {"openai": "openai", "anthropic": "anthropic", "google.genai": "google-genai"}
    for sdk in SDK_BY_KIND.values():
        names = [re.split(r"[<>=!~\[ ;]", r, maxsplit=1)[0] for r in extras[sdk.extra]]
        for module in sdk.modules:
            assert distributions[module] in names


def test_the_adapters_import_what_the_table_says():
    # Every require_sdk(module, extra) call imports a package (or a submodule of
    # one) that its extra's SDK lists, so a build collecting `modules` has it.
    modules_by_extra = {sdk.extra: sdk.modules for sdk in SDK_BY_KIND.values()}
    calls = []
    for source in (ROOT / "inferenceswitch" / "adapters").glob("*.py"):
        calls += re.findall(
            r'require_sdk\(\s*"([^"]+)",\s*"([^"]+)"\s*\)', source.read_text(encoding="utf-8")
        )
    assert calls
    for module, extra in calls:
        assert any(module == m or module.startswith(m + ".") for m in modules_by_extra[extra])
