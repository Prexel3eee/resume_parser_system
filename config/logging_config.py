import logging.config

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {  # root logger
            'handlers': ['default'],
            'level': 'INFO',
            'propagate': True
        },
        'pdfminer': {
            'level': 'WARNING',
            'propagate': False
        },
        'PIL': {
            'level': 'WARNING',
            'propagate': False
        },
        'src': {
            'level': 'INFO',
            'propagate': False
        }
    }
}

def setup_logging():
    """Configure logging for the application"""
    logging.config.dictConfig(LOGGING_CONFIG) 