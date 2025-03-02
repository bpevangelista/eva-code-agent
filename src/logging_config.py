import logging


def get_logger(name: str) -> logging.Logger:
    """Return a logger with the given name that uses our configuration."""
    return logging.getLogger(name)


def configure_logging(level=logging.INFO):
    # Remove existing handlers to ensure configuration is applied
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # Set colored level names
    logging.addLevelName(logging.DEBUG, "\x1b[38;20mDEBUG\x1b[0m")
    logging.addLevelName(logging.INFO, "\x1b[32;20mINFO \x1b[0m")
    logging.addLevelName(logging.WARNING, "\x1b[33;20mWARN \x1b[0m")
    logging.addLevelName(logging.ERROR, "\x1b[31;20mERROR\x1b[0m")

    logging.basicConfig(
        level=level, format="%(asctime)s %(levelname)s %(name)-12s %(message)s"
    )


configure_logging()
