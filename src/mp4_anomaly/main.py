"""Package entry point."""

from .app import run
from .cli import parse_config


def main() -> None:
    """Run the command-line application."""

    run(parse_config())


if __name__ == "__main__":
    main()
