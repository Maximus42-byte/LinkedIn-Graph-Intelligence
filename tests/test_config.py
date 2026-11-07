from lgi.config import load_config


def test_config_has_deterministic_seed() -> None:
    config = load_config()
    assert isinstance(config["seed"], int)
    assert config["perturbation_iterations"] > 0
