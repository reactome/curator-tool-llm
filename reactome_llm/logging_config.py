import logging
import logging.config

def setup_logging():
    # Define a logging configuration
    LOGGING_CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "detailed": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "simple": {
                "format": "%(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "file": {
                "class": "logging.FileHandler",
                "filename": "reactome_llm.log",
                "formatter": "detailed",
                # "level": "INFO",
            },
            "console": {
                "class": "logging.StreamHandler",
                # "formatter": "simple",
                "formatter": "detailed",
                # "level": "INFO",
            },
        },
        "root": {
            # "handlers": ["file", "console"],
            "handlers": ["console"],
            "level": "INFO",
        },
    }

    # Apply logging configuration
    logging.config.dictConfig(LOGGING_CONFIG)
