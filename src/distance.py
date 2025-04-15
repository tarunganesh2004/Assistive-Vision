from loguru import logger
import numpy as np


class DistanceEstimator:
    def __init__(self, known_width=0.07, focal_length=1000):
        self.known_width = known_width  # e.g., bottle width in meters
        self.focal_length = focal_length
        logger.info(
            f"Distance estimator initialized: known_width={known_width}m, focal_length={focal_length}"
        )

    def estimate(self, pixel_width):
        try:
            if pixel_width <= 0:
                logger.warning("Invalid pixel width")
                return None
            distance = (self.known_width * self.focal_length) / pixel_width
            logger.debug(f"Estimated distance: {distance:.2f}m")
            return distance
        except Exception as e:
            logger.exception(f"Distance estimation error: {e}")
            return None

    def calibrate(self, known_distance, pixel_width):
        try:
            if pixel_width <= 0 or known_distance <= 0:
                raise ValueError("Invalid calibration parameters")
            self.focal_length = (pixel_width * known_distance) / self.known_width
            logger.info(f"Calibrated focal_length: {self.focal_length}")
        except Exception as e:
            logger.exception(f"Calibration error: {e}")
