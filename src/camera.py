import cv2
from loguru import logger


class Camera:
    def __init__(self, source=0, width=640, height=480):
        self.source = source
        self.width = width
        self.height = height
        self.cap = None
        logger.info(f"Initializing camera with source: {source}")

    def start(self):
        try:
            self.cap = cv2.VideoCapture(self.source)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                logger.error("Failed to open camera")
                raise RuntimeError("Camera not accessible")
            logger.info("Camera started")
            return self
        except Exception as e:
            logger.exception(f"Camera init failed: {e}")
            raise

    def read(self):
        try:
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame")
                return None
            return frame
        except Exception as e:
            logger.exception(f"Frame read error: {e}")
            return None

    def release(self):
        if self.cap:
            self.cap.release()
            logger.info("Camera released")
