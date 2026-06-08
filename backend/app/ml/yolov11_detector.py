"""
YOLOv11 Detector for PPE Detection

Uses a trained YOLO model for PPE detection with violation classes.
Supports two class profiles:
- construction: Ultralytics Construction-PPE classes
- lab: legacy lab PPE/action classes
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path
from ..core.config import settings

logger = logging.getLogger(__name__)


class YOLOv11Detector:
    """PPE Detector using trained YOLOv11 model."""

    LAB_CLASS_NAMES = {
        0: "Drinking",
        1: "Eating",
        2: "Gloves",
        3: "Googles",
        4: "Head Mask",
        5: "Lab Coat",
        6: "Mask",
        7: "No Gloves",
        8: "No Head Mask",
        9: "No Lab coat",
        10: "No Mask",
        11: "No googles",
    }

    LAB_CLASS_MAPPING = {
        2: "gloves",
        3: "safety goggles",
        4: "head mask",
        5: "lab coat",
        6: "face mask",
    }

    LAB_VIOLATION_CLASS_MAPPING = {
        7: "gloves",
        8: "head mask",
        9: "lab coat",
        10: "face mask",
        11: "safety goggles",
    }

    LAB_ACTION_VIOLATIONS = {0: "drinking", 1: "eating"}

    CONSTRUCTION_CLASS_NAMES = {
        0: "helmet",
        1: "gloves",
        2: "vest",
        3: "boots",
        4: "goggles",
        5: "none",
        6: "Person",
        7: "no_helmet",
        8: "no_goggle",
        9: "no_gloves",
        10: "no_boots",
    }

    CONSTRUCTION_CLASS_MAPPING = {
        0: "helmet",
        1: "gloves",
        2: "vest",
        3: "boots",
        4: "safety goggles",
    }

    CONSTRUCTION_VIOLATION_CLASS_MAPPING = {
        7: "helmet",
        8: "safety goggles",
        9: "gloves",
        10: "boots",
    }

    CONSTRUCTION_ACTION_VIOLATIONS = {}
    CONSTRUCTION_PERSON_CLASS_IDS = {6}

    def __init__(self):
        self.model = None
        self.model_type = None
        self.device = "cuda" if self._check_cuda() else "cpu"
        (
            self.class_names,
            self.class_mapping,
            self.violation_class_mapping,
            self.action_violations,
            self.person_class_ids,
        ) = self._load_class_profile(settings.PPE_CLASS_PROFILE)
        self.confidence_threshold = settings.DETECTION_CONFIDENCE_THRESHOLD
        self.violation_threshold = getattr(
            settings, "VIOLATION_CONFIDENCE_THRESHOLD", 0.3
        )
        self._initialized = False

        self.multi_scale_enabled = getattr(settings, "MULTI_SCALE_ENABLED", True)
        self.multi_scale_factors = getattr(
            settings, "MULTI_SCALE_FACTORS", [1.0, 1.5, 2.0]
        )
        self.multi_scale_nms_threshold = getattr(
            settings, "MULTI_SCALE_NMS_THRESHOLD", 0.5
        )

    def _load_class_profile(
        self, profile: str
    ) -> Tuple[
        Dict[int, str],
        Dict[int, str],
        Dict[int, str],
        Dict[int, str],
        Set[int],
    ]:
        normalized = (profile or "construction").strip().lower()
        if normalized in {"construction", "construction-ppe", "ppe"}:
            return (
                dict(self.CONSTRUCTION_CLASS_NAMES),
                dict(self.CONSTRUCTION_CLASS_MAPPING),
                dict(self.CONSTRUCTION_VIOLATION_CLASS_MAPPING),
                dict(self.CONSTRUCTION_ACTION_VIOLATIONS),
                set(self.CONSTRUCTION_PERSON_CLASS_IDS),
            )

        if normalized in {"lab", "laboratory", "legacy_lab"}:
            return (
                dict(self.LAB_CLASS_NAMES),
                dict(self.LAB_CLASS_MAPPING),
                dict(self.LAB_VIOLATION_CLASS_MAPPING),
                dict(self.LAB_ACTION_VIOLATIONS),
                set(),
            )

        raise ValueError(
            f"Unsupported PPE_CLASS_PROFILE={profile!r}. "
            "Use 'construction' or 'lab'."
        )

    def _check_cuda(self) -> bool:
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def initialize(self):
        """Initialize YOLOv11 model."""
        if self._initialized:
            return

        model_path = settings.YOLOV11_MODEL_PATH
        if not model_path:
            self._handle_unavailable_model("YOLOV11_MODEL_PATH is not set")
            return

        model_path = Path(model_path)
        if not model_path.exists():
            self._handle_unavailable_model(f"YOLOv11 model not found at {model_path}")
            return

        try:
            if model_path.suffix == ".onnx":
                self._load_onnx_model(model_path)
            else:
                self._load_pytorch_model(model_path)

            self._initialized = True
            logger.info(f"YOLOv11 loaded: {model_path.name} ({self.model_type})")
            if self.multi_scale_enabled:
                logger.info(
                    f"Multi-scale detection enabled: {self.multi_scale_factors}"
                )

        except Exception as e:
            logger.error(f"YOLOv11 loading failed: {e}")
            self._handle_unavailable_model(f"YOLOv11 loading failed: {e}", cause=e)

    def _handle_unavailable_model(self, message: str, cause: Exception | None = None):
        self._initialized = True
        if settings.USE_MOCK_DETECTOR:
            logger.warning("%s - using explicit mock detector", message)
            return

        if settings.FAIL_ON_MISSING_MODELS:
            raise RuntimeError(
                f"{message}. Place a trained PPE model at {settings.YOLOV11_MODEL_PATH} "
                "or set USE_MOCK_DETECTOR=true only for development."
            ) from cause

        logger.error("%s - PPE detection disabled", message)

    def _load_pytorch_model(self, model_path: Path):
        from ultralytics import YOLO

        self.model = YOLO(str(model_path))
        self.model_type = "pytorch"

        if hasattr(self.model, "names") and self.model.names:
            for class_id, class_name in self.model.names.items():
                self.class_names[int(class_id)] = str(class_name)
            self._configure_class_roles_from_model_names()

    def _configure_class_roles_from_model_names(self) -> None:
        """Derive detector roles from model.names when available."""
        ppe_aliases = {
            "helmet": "helmet",
            "hardhat": "helmet",
            "hard_hat": "helmet",
            "glove": "gloves",
            "gloves": "gloves",
            "vest": "vest",
            "safety_vest": "vest",
            "reflective_vest": "vest",
            "boot": "boots",
            "boots": "boots",
            "shoe": "boots",
            "shoes": "boots",
            "safety_shoes": "boots",
            "goggle": "safety goggles",
            "goggles": "safety goggles",
            "safety_goggle": "safety goggles",
            "safety_goggles": "safety goggles",
            "lab_coat": "lab coat",
            "labcoat": "lab coat",
            "mask": "face mask",
            "face_mask": "face mask",
            "head_mask": "head mask",
        }
        violation_aliases = {
            "no_helmet": "helmet",
            "no_hardhat": "helmet",
            "no_hard_hat": "helmet",
            "no_glove": "gloves",
            "no_gloves": "gloves",
            "no_goggle": "safety goggles",
            "no_goggles": "safety goggles",
            "no_safety_goggle": "safety goggles",
            "no_safety_goggles": "safety goggles",
            "no_boot": "boots",
            "no_boots": "boots",
            "no_shoe": "boots",
            "no_shoes": "boots",
            "no_lab_coat": "lab coat",
            "no_labcoat": "lab coat",
            "no_mask": "face mask",
            "no_face_mask": "face mask",
            "no_head_mask": "head mask",
        }
        action_aliases = {"drinking": "drinking", "eating": "eating"}

        class_mapping: Dict[int, str] = {}
        violation_mapping: Dict[int, str] = {}
        action_violations: Dict[int, str] = {}
        person_class_ids: Set[int] = set()

        for class_id, class_name in self.class_names.items():
            normalized = self._normalize_class_name(class_name)
            if normalized in {"person", "worker", "workers"}:
                person_class_ids.add(class_id)
            elif normalized in ppe_aliases:
                class_mapping[class_id] = ppe_aliases[normalized]
            elif normalized in violation_aliases:
                violation_mapping[class_id] = violation_aliases[normalized]
            elif normalized in action_aliases:
                action_violations[class_id] = action_aliases[normalized]

        if class_mapping:
            self.class_mapping = class_mapping
        if violation_mapping:
            self.violation_class_mapping = violation_mapping
        if action_violations:
            self.action_violations = action_violations
        if person_class_ids:
            self.person_class_ids = person_class_ids

        logger.info(
            "YOLO class roles: %d PPE, %d violations, %d actions, %d person classes",
            len(self.class_mapping),
            len(self.violation_class_mapping),
            len(self.action_violations),
            len(self.person_class_ids),
        )

    @staticmethod
    def _normalize_class_name(class_name: str) -> str:
        return (
            str(class_name)
            .strip()
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

    def _load_onnx_model(self, model_path: Path):
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        self.model = ort.InferenceSession(str(model_path), providers=providers)
        self.model_type = "onnx"

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Detect PPE items and violations in a frame."""
        if not self._initialized:
            self.initialize()

        results = {
            "persons": [],
            "ppe_detections": {},
            "violation_detections": {},
            "action_violations": [],
            "frame_shape": frame.shape[:2],
        }

        if self.model is None:
            if settings.USE_MOCK_DETECTOR:
                return self._mock_detect(frame)
            raise RuntimeError(
                "YOLOv11 PPE detector is unavailable. This runtime is configured "
                "to reject mock detections."
            )

        try:
            if self.model_type == "pytorch":
                detections = self._detect_pytorch(frame)
            else:
                detections = self._detect_onnx(frame)

            persons, ppe_detections, violation_detections, action_violations = (
                self._parse_detections(detections, frame.shape[:2])
            )

            results["persons"] = persons
            results["ppe_detections"] = ppe_detections
            results["violation_detections"] = violation_detections
            results["action_violations"] = action_violations

        except Exception as e:
            logger.error(f"YOLOv11 detection error: {e}")
            if settings.USE_MOCK_DETECTOR:
                return self._mock_detect(frame)
            raise

        return results

    def _detect_pytorch(self, frame: np.ndarray) -> List[Dict]:
        if self.multi_scale_enabled and len(self.multi_scale_factors) > 1:
            return self._detect_multiscale(frame)
        return self._detect_single_scale(frame, scale=1.0)

    def _detect_single_scale(self, frame: np.ndarray, scale: float = 1.0) -> List[Dict]:
        h, w = frame.shape[:2]

        if scale != 1.0:
            scaled_frame = cv2.resize(
                frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR
            )
        else:
            scaled_frame = frame

        min_threshold = min(self.confidence_threshold, self.violation_threshold)
        results = self.model(scaled_frame, conf=min_threshold, verbose=False, save=False)

        detections = []
        for result in results:
            for box in result.boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                if scale != 1.0:
                    xyxy = xyxy / scale

                detections.append(
                    {
                        "class_id": int(box.cls[0]),
                        "confidence": float(box.conf[0]),
                        "box": [
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ],
                    }
                )

        return detections

    def _detect_multiscale(self, frame: np.ndarray) -> List[Dict]:
        all_detections = []
        for scale in self.multi_scale_factors:
            all_detections.extend(self._detect_single_scale(frame, scale))

        if not all_detections:
            return []

        return self._apply_nms(all_detections)

    def _apply_nms(self, detections: List[Dict]) -> List[Dict]:
        if not detections:
            return []

        class_detections: Dict[int, List[Dict]] = {}
        for det in detections:
            class_id = det["class_id"]
            if class_id not in class_detections:
                class_detections[class_id] = []
            class_detections[class_id].append(det)

        merged = []
        for class_id, class_dets in class_detections.items():
            if len(class_dets) == 1:
                merged.append(class_dets[0])
                continue

            boxes = np.array([d["box"] for d in class_dets])
            scores = np.array([d["confidence"] for d in class_dets])
            keep_indices = self._nms(boxes, scores, self.multi_scale_nms_threshold)
            for idx in keep_indices:
                merged.append(class_dets[idx])

        return merged

    def _nms(
        self, boxes: np.ndarray, scores: np.ndarray, threshold: float
    ) -> List[int]:
        if len(boxes) == 0:
            return []

        order = scores.argsort()[::-1]
        keep = []

        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break

            remaining = order[1:]
            ious = self._compute_iou_batch(boxes[i], boxes[remaining])
            order = remaining[ious < threshold]

        return keep

    def _compute_iou_batch(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = box_area + boxes_area - intersection

        return intersection / np.maximum(union, 1e-6)

    def _detect_onnx(self, frame: np.ndarray) -> List[Dict]:
        input_name = self.model.get_inputs()[0].name
        input_shape = self.model.get_inputs()[0].shape
        img_size = input_shape[2]

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))
        img_array = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        outputs = self.model.run(None, {input_name: img_array})
        output = outputs[0]

        if len(output.shape) == 3:
            output = output[0]

        detections = []
        h, w = frame.shape[:2]
        scale_x, scale_y = w / img_size, h / img_size
        min_threshold = min(self.confidence_threshold, self.violation_threshold)

        for detection in output:
            if len(detection) < 6:
                continue

            x1, y1, x2, y2, conf = [float(d) for d in detection[:5]]

            if abs(x1) < 1.0 and abs(x2) < 1.0:
                x1, y1, x2, y2 = x1 * w, y1 * h, x2 * w, y2 * h
            else:
                x1, y1, x2, y2 = x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y

            if len(detection) == 6:
                class_id = int(detection[5])
                final_conf = conf
            else:
                class_scores = detection[5:]
                class_id = int(np.argmax(class_scores))
                final_conf = float(conf * class_scores[class_id])

            if (
                class_id in self.class_names
                and 0 <= final_conf <= 1
                and final_conf >= min_threshold
            ):
                detections.append(
                    {
                        "class_id": class_id,
                        "confidence": final_conf,
                        "box": [x1, y1, x2, y2],
                    }
                )

        return detections

    def _parse_detections(
        self, detections: List[Dict], frame_shape: Tuple[int, int]
    ) -> Tuple:
        persons = []
        ppe_detections = {ppe_type: [] for ppe_type in settings.PPE_PROMPTS}
        violation_detections = {}
        action_violations = []

        for det in detections:
            class_id = det["class_id"]
            confidence = det["confidence"]
            box = det["box"]

            if class_id in self.person_class_ids:
                if confidence >= self.confidence_threshold:
                    persons.append(
                        {
                            "id": len(persons),
                            "box": box,
                            "score": confidence,
                            "mask": None,
                            "frame_shape": frame_shape,
                        }
                    )

            elif class_id in self.class_mapping:
                ppe_type = self.class_mapping[class_id]
                if confidence >= self.confidence_threshold:
                    if ppe_type not in ppe_detections:
                        ppe_detections[ppe_type] = []
                    ppe_detections[ppe_type].append(
                        {"box": box, "score": confidence, "mask": None}
                    )

            elif class_id in self.violation_class_mapping:
                if confidence >= self.violation_threshold:
                    ppe_type = self.violation_class_mapping[class_id]
                    if ppe_type not in violation_detections:
                        violation_detections[ppe_type] = []
                    violation_detections[ppe_type].append(
                        {
                            "box": box,
                            "score": confidence,
                            "mask": None,
                            "class_name": self.class_names.get(class_id, ""),
                        }
                    )

            elif class_id in self.action_violations:
                action_violations.append(
                    {
                        "box": box,
                        "score": confidence,
                        "action": self.action_violations[class_id],
                        "class_name": self.class_names.get(class_id, ""),
                    }
                )

        if not persons:
            all_boxes = []
            for ppe_list in ppe_detections.values():
                all_boxes.extend([d["box"] for d in ppe_list])
            for viol_list in violation_detections.values():
                all_boxes.extend([d["box"] for d in viol_list])
            for action in action_violations:
                all_boxes.append(action["box"])

            if all_boxes:
                persons = self._create_person_boxes(all_boxes, frame_shape)

        return persons, ppe_detections, violation_detections, action_violations

    def _create_person_boxes(self, boxes: List, frame_shape: tuple) -> List[Dict]:
        if not boxes:
            return []

        x1_min = min(box[0] for box in boxes)
        y1_min = min(box[1] for box in boxes)
        x2_max = max(box[2] for box in boxes)
        y2_max = max(box[3] for box in boxes)

        h, w = frame_shape
        padding = 50
        person_box = [
            max(0, x1_min - padding),
            max(0, y1_min - padding),
            min(w, x2_max + padding),
            min(h, y2_max + padding),
        ]

        return [
            {
                "id": 0,
                "box": person_box,
                "score": 0.9,
                "mask": None,
                "frame_shape": frame_shape,
            }
        ]

    def _mock_detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Return an empty detector response when explicit mock mode is enabled."""
        h, w = frame.shape[:2]
        return {
            "persons": [],
            "ppe_detections": {ppe_type: [] for ppe_type in settings.PPE_PROMPTS},
            "violation_detections": {},
            "action_violations": [],
            "frame_shape": (h, w),
        }

    def associate_ppe_to_persons(
        self,
        persons: List[Dict],
        ppe_detections: Dict,
        violation_detections: Optional[Dict] = None,
        action_violations: Optional[List] = None,
    ) -> List[Dict]:
        """Associate PPE and violations with persons."""
        required_ppe = set(settings.REQUIRED_PPE)
        violation_detections = violation_detections or {}
        action_violations = action_violations or []

        for person in persons:
            person["detected_ppe"] = []
            person["missing_ppe"] = []
            person["action_violations"] = []
            person["detection_confidence"] = {}
            person["ppe_detections"] = []
            person_box = person["box"]

            for ppe_type, detections in ppe_detections.items():
                for detection in detections:
                    if self._boxes_overlap(person_box, detection["box"]):
                        if ppe_type not in person["detected_ppe"]:
                            person["detected_ppe"].append(ppe_type)
                            person["detection_confidence"][ppe_type] = detection.get(
                                "score", 0.0
                            )
                        person["ppe_detections"].append(
                            {
                                "label": ppe_type,
                                "display_name": ppe_type,
                                "box": detection["box"],
                                "score": detection.get("score", 0.0),
                                "is_violation": False,
                            }
                        )
                        break

            for ppe_type, detections in violation_detections.items():
                for detection in detections:
                    if self._boxes_overlap(person_box, detection["box"]):
                        if (
                            ppe_type not in person["missing_ppe"]
                            and ppe_type in required_ppe
                        ):
                            person["missing_ppe"].append(ppe_type)
                            person["detection_confidence"][f"no_{ppe_type}"] = (
                                detection.get("score", 0.0)
                            )
                        person["ppe_detections"].append(
                            {
                                "label": f"no_{ppe_type}",
                                "display_name": ppe_type,
                                "box": detection["box"],
                                "score": detection.get("score", 0.0),
                                "is_violation": True,
                            }
                        )
                        break

            if (
                getattr(settings, "INFER_MISSING_PPE_FROM_ABSENCE", True)
                and self._is_absence_inference_eligible(person)
            ):
                absence_confidence = getattr(
                    settings, "ABSENCE_INFERENCE_CONFIDENCE", 0.65
                )
                for ppe_type in settings.REQUIRED_PPE:
                    if (
                        ppe_type not in person["detected_ppe"]
                        and ppe_type not in person["missing_ppe"]
                    ):
                        person["missing_ppe"].append(ppe_type)
                        person["detection_confidence"][f"no_{ppe_type}"] = (
                            absence_confidence
                        )

            for action in action_violations:
                if self._boxes_overlap(person_box, action["box"]):
                    person["action_violations"].append(
                        {"action": action["action"], "score": action["score"]}
                    )

            person["is_violation"] = (
                len(person["missing_ppe"]) > 0 or len(person["action_violations"]) > 0
            )

        return persons

    def _is_absence_inference_eligible(self, person: Dict[str, Any]) -> bool:
        """Return True when the person crop is clear enough to infer missing PPE."""
        try:
            score = float(person.get("score", 0.0) or 0.0)
        except (TypeError, ValueError):
            score = 0.0

        if score < getattr(settings, "ABSENCE_MIN_PERSON_CONFIDENCE", 0.5):
            return False

        box = person.get("box") or []
        if len(box) != 4:
            return False

        x1, y1, x2, y2 = [float(coord) for coord in box]
        box_width = max(0.0, x2 - x1)
        box_height = max(0.0, y2 - y1)
        if box_width <= 0 or box_height <= 0:
            return False

        frame_shape = person.get("frame_shape")
        if not frame_shape:
            return True

        frame_height, frame_width = float(frame_shape[0]), float(frame_shape[1])
        if frame_width <= 0 or frame_height <= 0:
            return True

        area_ratio = (box_width * box_height) / (frame_width * frame_height)
        height_ratio = box_height / frame_height

        if area_ratio < getattr(settings, "ABSENCE_MIN_PERSON_BOX_AREA_RATIO", 0.004):
            return False
        if height_ratio < getattr(settings, "ABSENCE_MIN_PERSON_HEIGHT_RATIO", 0.1):
            return False

        edge_margin = getattr(settings, "ABSENCE_EDGE_MARGIN_RATIO", 0.01)
        if edge_margin > 0:
            left = frame_width * edge_margin
            top = frame_height * edge_margin
            right = frame_width * (1.0 - edge_margin)
            bottom = frame_height * (1.0 - edge_margin)
            if x1 <= left or y1 <= top or x2 >= right or y2 >= bottom:
                return False

        return True

    def _boxes_overlap(
        self, box1: List[float], box2: List[float], threshold: float = 0.3
    ) -> bool:
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return False

        intersection = (x2 - x1) * (y2 - y1)
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        return box2_area > 0 and intersection / box2_area >= threshold


# Singleton
_yolov11_detector = None


def get_yolov11_detector() -> YOLOv11Detector:
    global _yolov11_detector
    if _yolov11_detector is None:
        _yolov11_detector = YOLOv11Detector()
    return _yolov11_detector
