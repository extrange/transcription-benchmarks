import logging


def setup_logging() -> None:
    """Set up basic logging output."""

    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
