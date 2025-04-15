import paddle
from ppdet.core.workspace import load_config, create
from ppdet.engine import Trainer
from loguru import logger
import cv2
import numpy as np


class Detector:
    def __init__(
        self, model_config="rt_detr_l.yml", weights="rt_detr_l_coco", conf_threshold=0.5
    ):
        self.conf_threshold = conf_threshold
        self.model = self._load_model(model_config, weights)
        self.classes = self._load_coco_classes()
        logger.info("RT-DETR model loaded")

    def _load_model(self, model_config, weights):
        try:
            # Load RT-DETR config
            cfg = load_config(f"ppdet/configs/rtdetr/{model_config}")
            cfg["weights"] = weights
            cfg["architecture"] = "RTDETR"
            # Create model
            model = create("model", cfg)
            model.eval()
            return model
        except Exception as e:
            logger.exception(f"Model load error: {e}")
            raise

    def _load_coco_classes(self):
        # COCO 80 classes (simplified for example)
        return {
            i: name
            for i, name in enumerate(
                [
                    "person",
                    "car",
                    "chair",
                    "book",
                    "bottle",  # Add more as needed
                    "cup",
                    "table",
                    "door",
                    "window",
                    "sign",
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
                outputs = self.model.predict([img])[0]
                boxes = outputs["bbox"]
                scores = outputs["bbox_score"]
                labels = outputs["bbox_label"]

            # Post-process
            detections = []
            for box, score, label in zip(boxes, scores, labels):
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
