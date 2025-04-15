import paddle
from paddle.vision.models import rt_detr_l
from loguru import logger
import cv2
import numpy as np


class Detector:
    def __init__(self, model_path="rt_detr_l.pdparams", conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        self.model = rt_detr_l(pretrained=True)
        self.model.eval()
        self.classes = self._load_coco_classes()  # COCO 80 classes
        logger.info("RT-DETR model loaded")

    def _load_coco_classes(self):
        # Simplified COCO classes (full list in production)
        return {
            i: name
            for i, name in enumerate(
                [
                    "person",
                    "car",
                    "chair",
                    "book",
                    "bottle",  # Add more as needed
                ]
            )
        }

    def detect(self, frame):
        try:
            # Preprocess
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (640, 640))
            img = img.transpose((2, 0, 1)) / 255.0
            img = np.expand_dims(img, axis=0).astype(np.float32)

            # Inference
            with paddle.no_grad():
                outputs = self.model(paddle.to_tensor(img))

            # Post-process
            boxes, scores, labels = outputs
            detections = []
            for box, score, label in zip(boxes.numpy(), scores.numpy(), labels.numpy()):
                if score > self.conf_threshold:
                    x1, y1, x2, y2 = box.astype(int)
                    label = int(label)
                    if label in self.classes:
                        detections.append(
                            {
                                "label": self.classes[label],
                                "confidence": float(score),
                                "box": (x1, y1, x2, y2),
                            }
                        )
                        # Draw box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"{self.classes[label]} {score:.2f}",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2,
                        )
            logger.debug(f"Detected {len(detections)} objects")
            return frame, detections
        except Exception as e:
            logger.exception(f"Detection error: {e}")
            return frame, []
