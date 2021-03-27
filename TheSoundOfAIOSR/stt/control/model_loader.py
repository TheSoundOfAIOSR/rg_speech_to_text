import time
import logging

def load_stt_models(logger: logging.Logger):
    logger.debug("pre-loading STT models in cca 1 sec...")
    time.sleep(1)
    logger.debug("STT models has been loaded.")
