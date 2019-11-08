from termcolor import colored
import argparse
import enum


def parse_cli_overides(config: dict):
    """
    Parse args from CLI and override config dictionary entries
    """
    parser = argparse.ArgumentParser()
    for key, value in config.items():
        parser.add_argument(f"--{key}")
    args = vars(parser.parse_args())

    def print_config_override(key, old_value, new_value, first_config_overide):
        if first_config_overide:
            print(colored("Config overrides:", "red"))
        print(f"     {key:25s} -> {new_value} (instead of {old_value})")

    def cast_argument(key, old_value, new_value):
        try:
            if new_value is None:
                return None
            if type(old_value) is int:
                return int(new_value)
            if type(old_value) is float:
                return float(new_value)
            if type(old_value) is str:
                return new_value
            if type(old_value) is bool:
                return new_value.lower() in ("yes", "true", "t", "1")
            if issubclass(old_value.__class__, enum.Enum):
                return old_value.__class__(new_value)
            if old_value is None:
                return new_value  # assume string
            raise ValueError()
        except Exception:
            raise ValueError(f"Unable to parse config key '{key}' with value '{new_value}'")

    first_config_overide = True
    for key, original_value in config.items():
        override_value = cast_argument(key, original_value, args[key])
        if override_value is not None and override_value != original_value:
            config[key] = override_value
            print_config_override(key, original_value, override_value, first_config_overide)
            first_config_overide = False

    return config
