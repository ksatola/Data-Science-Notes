# Module handler for logging
"""
Usage:
Import this module and use required levels like this:
import logger
logger.info("Started")
logger.error("Operation failed.")
logger.debug("Encountered debug case")
"""

import logging.handlers

logger = logging.getLogger(__name__)
formatter = logging.Formatter(
    fmt='%(filename)s | %(lineno)d | %(funcName)s | %(asctime)s | %(levelname)s: %(message)s',
    datefmt='%d-%b-%y %H:%M:%S'
)
logger.setLevel(logging.INFO)

logFilePath = "usage.log"
file_handler = logging.handlers.RotatingFileHandler(
    filename=logFilePath, maxBytes=1000000, backupCount=3
)
file_handler.setFormatter(formatter)
file_handler.setLevel(logging.DEBUG)
file_handler.addFilter(logging.Filter(name='root'))
logger.addHandler(file_handler)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(formatter)
logger.addHandler(consoleHandler)
