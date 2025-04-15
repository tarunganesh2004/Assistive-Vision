from paddleocr import PaddleOCR
from loguru import logger
import cv2


class OCR:
    def __init__(self, lang="en"):
        self.ocr = PaddleOCR(use_angle_cls=True, lang=lang)
        logger.info("PaddleOCR initialized")

    def read_text(self, frame):
        try:
            results = self.ocr.ocr(frame, cls=True)
            texts = []
            if results and results[0]:
                for line in results[0]:
                    box, (text, score) = line[0], line[1]
                    if score > 0.5:
                        texts.append(text)
                        # Draw box
                        box = [(int(pt[0]), int(pt[1])) for pt in box]
                        cv2.polylines(frame, [np.array(box)], True, (255, 0, 0), 2)
                        cv2.putText(
                            frame,
                            text,
                            box[0],
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 0, 0),
                            2,
                        )
            logger.debug(f"OCR found {len(texts)} texts")
            return frame, texts
        except Exception as e:
            logger.exception(f"OCR error: {e}")
            return frame, []
