from azuero.config import load_config, resolve_resources


def test_profiles_have_same_keys():
    presets = load_config()["resources"]["presets"]
    assert set(presets["small"]) == set(presets["full"])


def test_override_precedence(monkeypatch):
    cfg = load_config()
    r = resolve_resources(cfg, "small")
    assert r.delete_raw_after_composite is True
    monkeypatch.setenv("AZUERO_RESOURCES__DELETE_RAW_AFTER_COMPOSITE", "false")
    r = resolve_resources(cfg, "small")
    assert r.delete_raw_after_composite is False
    r = resolve_resources(cfg, "small", {"delete_raw_after_composite": True})
    assert r.delete_raw_after_composite is True


def test_profile_env(monkeypatch):
    monkeypatch.setenv("AZUERO_PROFILE", "full")
    assert resolve_resources(load_config()).profile == "full"
