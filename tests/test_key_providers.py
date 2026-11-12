from inferenceswitch import default_registry


def test_keyed_providers_have_a_label_and_env_var():
    keyed = default_registry().key_providers()
    assert [s.name for s in keyed][:3] == ["anthropic", "gemini", "openai"]
    assert all(s.api_key_env and s.title for s in keyed)
    assert all(s.takes_key for s in keyed)


def test_local_servers_take_no_key():
    reg = default_registry()
    assert not reg["ollama"].takes_key
    assert "ollama" not in [s.name for s in reg.key_providers()]
    assert reg["ollama"].title == "ollama"
