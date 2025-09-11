import logging
import logging.config

COLORS = {
    "DEBUG": "\033[37m",   # white
    "INFO": "\033[32m",    # green
    "WARNING": "\033[33m", # yellow
    "ERROR": "\033[31m",   # red
    "CRITICAL": "\033[41m" # red background
}
RESET = "\033[0m"

class ColorFormatter(logging.Formatter):
    def format(self, record):
        levelname = record.levelname
        if levelname in COLORS:
            record.levelname_color = f"{COLORS[levelname]}{levelname}{RESET}"
        else:
            record.levelname_color = levelname
        return super().format(record)

class MaxLevelFilter(logging.Filter):
    def __init__(self, max_level):
        super().__init__()
        self.max_level = max_level
    def filter(self, record):
        return record.levelno <= self.max_level

class EffectiveRootLevelFilter(logging.Filter):
    """Pass only when the root logger's effective level equals `required_level`."""
    def __init__(self, required_level):
        super().__init__()
        self.required_level = required_level
    def filter(self, record):
        return logging.getLogger().getEffectiveLevel() == self.required_level

def configure_logging():
    logger_config = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "simple": {
                "()": ColorFormatter,
                "format": "\033[90m%(asctime)s\033[0m [%(levelname_color)s] \033[36m%(name)s\033[0m: %(message)s"
            },
            "verbose": {
                "()": ColorFormatter,
                "format": "\033[90m%(asctime)s\033[0m [%(levelname_color)s] \033[36m%(name)s\033[0m: \033[34m%(funcName)s\033[0m  %(message)s (\033[35m%(filename)s:%(lineno)d\033[0m)"
            }
        },
        "filters": {
            "max_debug": {
                "()": MaxLevelFilter,
                "max_level": logging.DEBUG,
            },
            "max_info": {
                "()": MaxLevelFilter,
                "max_level": logging.INFO,
            },
            # Toggle filters based on the root logger's current level
            "root_is_info": {
                "()": EffectiveRootLevelFilter,
                "required_level": logging.INFO,
            },
            "root_is_debug": {
                "()": EffectiveRootLevelFilter,
                "required_level": logging.DEBUG,
            },
        },
        "handlers": {
            # Active only when root level is DEBUG: shows DEBUG (and below) with verbose
            "debug_stdout": {
                "class": "logging.StreamHandler",
                "level": "DEBUG",
                "filters": ["max_debug", "root_is_debug"],
                "formatter": "verbose",
                "stream": "ext://sys.stdout",
            },
            # INFO → simple, only when root level is INFO (overview mode)
            "info_simple": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "filters": ["max_info", "root_is_info"],
                "formatter": "simple",
                "stream": "ext://sys.stdout",
            },
            # INFO → verbose, only when root level is DEBUG (debugging mode)
            "info_verbose": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "filters": ["max_info", "root_is_debug"],
                "formatter": "verbose",
                "stream": "ext://sys.stdout",
            },
            "stderr": {
                "class": "logging.StreamHandler",
                "level": "WARNING",
                "formatter": "verbose",
                "stream": "ext://sys.stderr",
            },
        },
        "loggers": {
            "": {
                "handlers": ["debug_stdout", "info_simple", "info_verbose", "stderr"],
                # Toggle this between logging.INFO and logging.DEBUG at runtime:
                "level": "DEBUG",
                "propagate": False,
            },
        },
    }

    logging.config.dictConfig(logger_config)
