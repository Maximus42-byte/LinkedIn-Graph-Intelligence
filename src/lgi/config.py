"""Read pipeline parameters from a single YAML file."""

from pathlib import Path
import yaml


def load_config(path: str | Path = "config.yaml") -> dict:
    """Resolve relative paths from the config file's directory."""
    config_path = Path(path).resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["root"] = str(config_path.parent)
    return config
