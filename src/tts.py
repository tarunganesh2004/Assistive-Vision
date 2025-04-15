import pyttsx3
from loguru import logger


class TTS:
    def __init__(self, rate=150, volume=0.9):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty("rate", rate)
            self.engine.setProperty("volume", volume)
            logger.info(f"TTS initialized: rate={rate}, volume={volume}")
        except Exception as e:
            logger.exception(f"TTS init error: {e}")
            raise

    def speak(self, text):
        try:
            if text:
                self.engine.stop()  # Clear queue to avoid overlap
                self.engine.say(text)
                self.engine.runAndWait()
                logger.debug(f"Spoke: {text}")
        except Exception as e:
            logger.exception(f"TTS error: {e}")
