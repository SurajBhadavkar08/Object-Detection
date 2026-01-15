import os
from typing import List, Tuple, Dict

import cv2
import numpy as np


class YoloV4Detector:
    """Lightweight wrapper around OpenCV DNN YOLOv4 for image detection."""

    def __init__(
        self,
        model_dir: str,
        cfg_filename: str = "yolov4.cfg",
        weights_filename: str = "yolov4.weights",
        labels_filename: str = "labels.txt",
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
    ) -> None:
        self.model_dir = model_dir
        self.cfg_path = os.path.join(model_dir, cfg_filename)
        self.weights_path = os.path.join(model_dir, weights_filename)
        self.labels_path = os.path.join(model_dir, labels_filename)

        if not os.path.exists(self.cfg_path):
            raise FileNotFoundError(self.cfg_path)
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(self.weights_path)
        if not os.path.exists(self.labels_path):
            raise FileNotFoundError(self.labels_path)

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold

        self.net = cv2.dnn.readNetFromDarknet(self.cfg_path, self.weights_path)
        layer_names = self.net.getLayerNames()
        self.output_layer_names = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]

        self.class_names: List[str] = []
        with open(self.labels_path, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name:
                    self.class_names.append(name)

    def set_thresholds(self, confidence_threshold: float, nms_threshold: float) -> None:
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)

    def detect(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Run detection on a BGR image and return results.

        Returns a dict with keys: boxes (Nx4), confidences (N,), class_ids (N,), indices (K,)
        """
        height, width = image_bgr.shape[:2]
        blob = cv2.dnn.blobFromImage(
            image_bgr, 1.0 / 255.0, (416, 416), (0, 0, 0), swapRB=True, crop=False
        )
        self.net.setInput(blob)
        layer_outputs = self.net.forward(self.output_layer_names)

        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for output in layer_outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                confidence = float(scores[class_id])
                if confidence >= self.confidence_threshold:
                    box = detection[0:4] * np.array([width, height, width, height])
                    (center_x, center_y, w, h) = box.astype("int")
                    x = int(center_x - (w / 2))
                    y = int(center_y - (h / 2))
                    boxes.append([x, y, int(w), int(h)])
                    confidences.append(confidence)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(
            boxes, confidences, self.confidence_threshold, self.nms_threshold
        )

        # Normalize indices to a flat list of ints
        flat_indices: List[int] = []
        if len(indices) > 0:
            flat_indices = [int(i) for i in np.array(indices).flatten().tolist()]

        return {
            "boxes": np.array(boxes, dtype=np.int32) if boxes else np.zeros((0, 4), dtype=np.int32),
            "confidences": np.array(confidences, dtype=np.float32) if confidences else np.zeros((0,), dtype=np.float32),
            "class_ids": np.array(class_ids, dtype=np.int32) if class_ids else np.zeros((0,), dtype=np.int32),
            "indices": np.array(flat_indices, dtype=np.int32),
        }

    def draw_detections(self, image_bgr: np.ndarray, detections: Dict[str, np.ndarray]) -> np.ndarray:
        """Return a copy of the image with detection rectangles and labels drawn."""
        output = image_bgr.copy()
        boxes = detections["boxes"]
        confidences = detections["confidences"]
        class_ids = detections["class_ids"]
        indices = detections["indices"]

        # Generate consistent colors per class id
        colors = {}
        for class_id in np.unique(class_ids):
            rng = np.random.default_rng(seed=int(class_id))
            colors[int(class_id)] = tuple(int(c) for c in rng.integers(0, 255, size=3))

        for i in indices:
            x, y, w, h = boxes[i].tolist()
            class_id = int(class_ids[i])
            confidence = float(confidences[i])
            color = colors.get(class_id, (0, 255, 0))

            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
            label = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else str(class_id)
            text = f"{label}: {confidence:.2f}"
            cv2.putText(output, text, (x, max(0, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return output


def get_default_detector() -> YoloV4Detector:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, "object  detection")
    return YoloV4Detector(model_dir=model_dir)




