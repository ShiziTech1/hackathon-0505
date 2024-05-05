import json
import logging
from logging import Formatter


class JsonFormatter(Formatter):
    def __init__(self):
        super(JsonFormatter, self).__init__()


logger = logging.root
handler = logging.StreamHandler()
handler.setFormatter(JsonFormatter())
logger.handlers = [handler]
logger.setLevel(logging.DEBUG)
