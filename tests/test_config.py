from ciblage import config


def test_paths_are_inside_project():
    assert config.DATA_DIR.parent == config.ROOT
    assert config.REPORTS_DIR.parent == config.ROOT
    assert (config.ROOT / "pyproject.toml").exists()
