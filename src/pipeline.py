from .camera import Camera
from .detector import Detector
from .ocr import OCR
from .distance import DistanceEstimator
from .tts import TTS
from .utils import load_config
from loguru import logger
import cv2
import threading


class Pipeline:
    def __init__(self, config):
        self.camera = Camera(
            source=config["camera"]["source"],
            width=config["camera"]["width"],
            height=config["camera"]["height"],
        )
        self.detector = Detector(conf_threshold=config["detector"]["conf_threshold"])
        self.ocr = OCR(lang=config["ocr"]["lang"])
        self.distance = DistanceEstimator(
            known_width=config["distance"]["known_width"],
            focal_length=config["distance"]["focal_length"],
        )
        self.tts = TTS(rate=config["tts"]["rate"], volume=config["tts"]["volume"])
        self.frame_skip_ocr = config["pipeline"]["frame_skip_ocr"]
        self.running = False
        logger.info("Pipeline initialized")

    def run(self):
        try:
            self.camera.start()
            self.running = True
            frame_count = 0

            def speak_async(text):
                threading.Thread(
                    target=self.tts.speak, args=(text,), daemon=True
                ).start()

            while self.running:
                frame = self.camera.read()
                if frame is None:
                    logger.warning("No frame, skipping")
                    continue

                # Object detection
                frame, detections = self.detector.detect(frame)
                narration = []
                for det in detections:
                    pixel_width = det["box"][2] - det["box"][0]
                    distance = self.distance.estimate(pixel_width)
                    if distance:
                        narration.append(f"{det['label']} at {distance:.1f} meters")

                # OCR (every N frames)
                if frame_count % self.frame_skip_ocr == 0:
                    frame, texts = self.ocr.read_text(frame)
                    for text in texts:
                        narration.append(f"Text: {text}")

                # Narrate
                if narration:
                    speak_async(". ".join(narration))

                # Display
                cv2.imshow("Assistive Vision", frame)
                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    logger.info("User requested exit")
                    break

        except Exception as e:
            logger.exception(f"Pipeline error: {e}")
        finally:
            self.stop()

    def stop(self):
        self.running = False
        self.camera.release()
        cv2.destroyAllWindows()
        logger.info("Pipeline stopped")
