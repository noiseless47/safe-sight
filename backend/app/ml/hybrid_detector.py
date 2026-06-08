"""
Hybrid Detector

Combined Ultralytics person tracking + YOLO PPE detection + SAM3/SAM2 segmentation pipeline.

Flow:
- PersonDetector detects persons with tracking
- SAM3/SAM2 generates precise masks for persons
- YOLO detects PPE and violations
- Associate PPE with persons using spatial overlap
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Set

from .person_detector import get_person_detector, PersonDetector
from .yolov11_detector import get_yolov11_detector, YOLOv11Detector
from .mask_utils import calculate_box_containment, calculate_mask_containment
from ..core.config import settings

logger = logging.getLogger(__name__)


class HybridDetector:
    """
    Combined person tracking + PPE detection + SAM3/SAM2 pipeline.

    Flow:
    - PersonDetector detects person boxes with track_ids
    - SAM3/SAM2 generates person masks when available
    - YOLO detects PPE and violation boxes
    - Associate PPE with persons using spatial overlap
    """

    def __init__(self):
        self.person_detector: Optional[PersonDetector] = None
        self.ppe_detector: Optional[YOLOv11Detector] = None
        self.sam3_segmenter = None
        self.sam2_segmenter = None

        self._initialized = False
        self._use_sam3 = getattr(settings, "USE_SAM3", True)
        self._use_sam2 = getattr(settings, "USE_SAM2", True) and not self._use_sam3
        self._use_sam2_video = getattr(settings, "USE_SAM2_VIDEO_PROPAGATION", True)
        self._sam2_propagate_interval = getattr(settings, "SAM2_PROPAGATE_INTERVAL", 2)
        self._segment_ppe = getattr(settings, "SAM2_SEGMENT_PPE", True)

        self._frame_count = 0
        self._video_initialized = False
        self._last_track_ids: Set[int] = set()
        self._containment_threshold = getattr(
            settings, "MASK_CONTAINMENT_THRESHOLD", 0.3
        )
        self._violation_containment_threshold = 0.1

    def initialize(self) -> None:
        """Initialize all sub-detectors."""
        if self._initialized:
            return

        print("Initializing HybridDetector...")

        # Person detector
        self.person_detector = get_person_detector()
        self.person_detector.initialize()

        # PPE detector
        self.ppe_detector = get_yolov11_detector()
        self.ppe_detector.initialize()

        if self.ppe_detector.model is None:
            logger.error("PPE model failed to load - PPE detection disabled")

        if self._use_sam3:
            try:
                from .sam3_segmenter import get_sam3_segmenter

                self.sam3_segmenter = get_sam3_segmenter()
                self.sam3_segmenter.initialize()
                logger.info("SAM3 segmenter initialized")
            except Exception as e:
                logger.warning(f"SAM3 init failed, trying SAM2: {e}")
                self.sam3_segmenter = None
                self._use_sam3 = False
                self._use_sam2 = True

        if not self._use_sam3 and self._use_sam2:
            try:
                from .sam2_segmenter import get_sam2_segmenter

                self.sam2_segmenter = get_sam2_segmenter()
                self.sam2_segmenter.initialize()
                logger.info("SAM2 segmenter initialized")
            except Exception as e:
                logger.warning(f"SAM2 init failed: {e}")
                self.sam2_segmenter = None
                self._use_sam2 = False

        self._initialized = True
        segmenter = (
            "SAM3" if self.sam3_segmenter else "SAM2" if self.sam2_segmenter else "None"
        )
        print(f"HybridDetector initialized (Segmenter: {segmenter})")

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """Run full detection pipeline on a frame."""
        if not self._initialized:
            self.initialize()

        self._frame_count += 1
        frame_shape = frame.shape[:2]

        persons = self.person_detector.detect_with_tracking(frame)
        self._attach_frame_shape(persons, frame_shape)

        if self.sam3_segmenter and self._use_sam3:
            persons = self._add_masks_sam3(frame, persons)
        elif self.sam2_segmenter and self._use_sam2:
            persons = self._add_masks_sam2(frame, persons)

        ppe_result = self.ppe_detector.detect(frame)
        if not persons and ppe_result.get("persons"):
            persons = ppe_result["persons"]
            self._attach_frame_shape(persons, frame_shape)

        ppe_detections = ppe_result.get("ppe_detections", {})
        violation_detections = ppe_result.get("violation_detections", {})
        action_violations = ppe_result.get("action_violations", [])

        segmenter = self.sam3_segmenter or self.sam2_segmenter
        if segmenter and self._segment_ppe:
            ppe_detections = self._add_masks_to_ppe(frame, ppe_detections)
            violation_detections = self._add_masks_to_ppe(frame, violation_detections)

        return {
            "persons": persons,
            "ppe_detections": ppe_detections,
            "violation_detections": violation_detections,
            "action_violations": action_violations,
            "frame_shape": frame_shape,
        }

    @staticmethod
    def _attach_frame_shape(persons: List[Dict], frame_shape: tuple) -> None:
        for person in persons:
            person["frame_shape"] = frame_shape

    def _add_masks_sam3(self, frame: np.ndarray, persons: List[Dict]) -> List[Dict]:
        """Add SAM3 masks to persons."""
        if not persons:
            return persons

        logger.info(
            f"[HybridDetector] _add_masks_sam3 called with {len(persons)} persons"
        )

        try:
            masks = self.sam3_segmenter.process_frame(frame, persons)
            logger.info(f"[HybridDetector] SAM3 returned {len(masks)} masks")

            masks_assigned = 0
            for person in persons:
                track_id = person.get("track_id")
                if track_id is not None and track_id in masks:
                    mask = masks[track_id]
                    if mask.dtype != np.uint8:
                        mask = (mask > 0).astype(np.uint8)
                    if mask.ndim == 3:
                        mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
                    person["mask"] = mask
                    masks_assigned += 1
                    mask_pixels = int(np.sum(mask > 0))
                    logger.info(
                        f"[HybridDetector] Assigned mask to track {track_id}: {mask_pixels} pixels, shape={mask.shape}"
                    )
                else:
                    logger.debug(f"[HybridDetector] No mask for track {track_id}")

            logger.info(
                f"[HybridDetector] Assigned {masks_assigned}/{len(persons)} masks from SAM3"
            )

        except Exception as e:
            logger.warning(f"[HybridDetector] SAM3 mask generation failed: {e}")

        return persons

    def _add_masks_sam2(self, frame: np.ndarray, persons: List[Dict]) -> List[Dict]:
        """Add SAM2 masks to persons using video propagation."""
        if not persons:
            return persons

        current_track_ids = {
            p.get("track_id") for p in persons if p.get("track_id") is not None
        }

        # Initialize video tracking on first frame
        if not self._video_initialized:
            try:
                self.sam2_segmenter.init_video_tracking(frame, persons)
                self._video_initialized = True
                self._last_track_ids = current_track_ids
                masks = self.sam2_segmenter.propagate_masks(frame)
                self._assign_masks(persons, masks)
            except Exception as e:
                logger.debug(f"SAM2 video init failed: {e}")
                return self._segment_single_frame(frame, persons)
            return persons

        # Add new tracks
        new_tracks = current_track_ids - self._last_track_ids
        for person in persons:
            track_id = person.get("track_id")
            if track_id in new_tracks:
                box = person.get("box")
                if box:
                    try:
                        result = self.sam2_segmenter.add_new_object(
                            frame, box, track_id
                        )
                        if result and result.get("mask") is not None:
                            person["mask"] = self._normalize_mask(result["mask"])
                    except Exception:
                        pass

        # Remove lost tracks
        for track_id in self._last_track_ids - current_track_ids:
            try:
                self.sam2_segmenter.remove_object(track_id)
            except Exception:
                pass

        self._last_track_ids = current_track_ids

        # Propagate masks
        if self._frame_count % self._sam2_propagate_interval == 0:
            try:
                masks = self.sam2_segmenter.propagate_masks(frame)
                self._assign_masks(persons, masks)
            except Exception as e:
                logger.debug(f"SAM2 propagation failed: {e}")

        return persons

    def _assign_masks(self, persons: List[Dict], masks: Dict[int, np.ndarray]):
        """Assign masks to persons by track_id."""
        for person in persons:
            track_id = person.get("track_id")
            if track_id in masks:
                person["mask"] = self._normalize_mask(masks[track_id])

    def _normalize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Ensure mask is uint8 and 2D."""
        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8)
        if mask.ndim == 3:
            mask = mask[0] if mask.shape[0] == 1 else mask[:, :, 0]
        return mask

    def _segment_single_frame(
        self, frame: np.ndarray, persons: List[Dict]
    ) -> List[Dict]:
        """Fallback single-frame segmentation."""
        if not self.sam2_segmenter:
            return persons

        boxes = [p.get("box") for p in persons if p.get("box")]
        labels = [f"person_{p.get('track_id', i)}" for i, p in enumerate(persons)]

        if not boxes:
            return persons

        try:
            results = self.sam2_segmenter.segment_boxes_batch(
                frame, boxes, labels, use_multimask=True
            )
            for person, result in zip(persons, results):
                if result.get("valid") and result.get("mask") is not None:
                    person["mask"] = self._normalize_mask(result["mask"])
        except Exception:
            pass

        return persons

    def _add_masks_to_ppe(self, frame: np.ndarray, ppe_detections: Dict) -> Dict:
        """Add masks to PPE detections."""
        segmenter = self.sam3_segmenter or self.sam2_segmenter
        if not segmenter:
            return ppe_detections

        all_boxes, all_labels, box_to_key = [], [], []
        for ppe_type, detections in ppe_detections.items():
            for i, det in enumerate(detections):
                if det.get("box"):
                    all_boxes.append(det["box"])
                    all_labels.append(ppe_type)
                    box_to_key.append((ppe_type, i))

        if not all_boxes:
            return ppe_detections

        try:
            results = segmenter.segment_boxes_batch(
                frame, all_boxes, all_labels, use_multimask=True
            )
            for (ppe_type, idx), result in zip(box_to_key, results):
                if result.get("valid") and result.get("mask") is not None:
                    ppe_detections[ppe_type][idx]["mask"] = result["mask"]
        except Exception:
            pass

        return ppe_detections

    def associate_ppe_to_persons(
        self,
        persons: List[Dict],
        ppe_detections: Dict[str, List[Dict]],
        violation_detections: Optional[Dict[str, List[Dict]]] = None,
        action_violations: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """Associate PPE items and violations with persons using box/mask overlap."""
        class_map = getattr(settings, "PPE_CLASS_MAP", {})
        required_ppe = list(settings.REQUIRED_PPE)
        required_ppe_set = set(required_ppe)
        violation_detections = violation_detections or {}
        action_violations = action_violations or []
        infer_missing = getattr(settings, "INFER_MISSING_PPE_FROM_ABSENCE", True)
        absence_confidence = getattr(settings, "ABSENCE_INFERENCE_CONFIDENCE", 0.65)

        for person in persons:
            person_box = person.get("box", [0, 0, 0, 0])
            person_mask = person.get("mask")
            detected_ppe, missing_ppe, person_actions = [], [], []
            detection_confidence, ppe_dets = {}, []

            for ppe_class, ppe_list in ppe_detections.items():
                if ppe_class.startswith("No "):
                    continue
                for ppe in ppe_list:
                    ppe_box = ppe.get("box", [0, 0, 0, 0])
                    ppe_mask = ppe.get("mask")

                    if person_mask is not None and ppe_mask is not None:
                        containment = calculate_mask_containment(ppe_mask, person_mask)
                    else:
                        containment = calculate_box_containment(ppe_box, person_box)

                    if containment >= self._containment_threshold:
                        ppe_name = class_map.get(ppe_class, ppe_class)
                        if ppe_name not in detected_ppe:
                            detected_ppe.append(ppe_name)
                            detection_confidence[ppe_name] = ppe.get("score", 0.0)
                        ppe_dets.append(
                            {
                                "label": ppe_class,
                                "display_name": ppe_name,
                                "box": ppe_box,
                                "score": ppe.get("score", 0.0),
                                "is_violation": False,
                            }
                        )

            for viol_class, viol_list in violation_detections.items():
                for viol in viol_list:
                    viol_box = viol.get("box", [0, 0, 0, 0])
                    viol_mask = viol.get("mask")

                    viol_cx = (viol_box[0] + viol_box[2]) / 2
                    viol_cy = (viol_box[1] + viol_box[3]) / 2
                    center_inside = (
                        person_box[0] <= viol_cx <= person_box[2]
                        and person_box[1] <= viol_cy <= person_box[3]
                    )

                    if person_mask is not None and viol_mask is not None:
                        containment = calculate_mask_containment(viol_mask, person_mask)
                    else:
                        containment = max(
                            calculate_box_containment(viol_box, person_box),
                            calculate_box_containment(person_box, viol_box),
                        )

                    iou = self._calculate_iou(viol_box, person_box)

                    if (
                        center_inside
                        or containment >= self._violation_containment_threshold
                        or iou >= 0.1
                    ):
                        if (
                            viol_class in required_ppe_set
                            and viol_class not in missing_ppe
                        ):
                            missing_ppe.append(viol_class)
                            detection_confidence[f"no_{viol_class}"] = viol.get(
                                "score", 0.0
                            )
                        ppe_dets.append(
                            {
                                "label": viol_class,
                                "display_name": viol_class,
                                "box": viol_box,
                                "score": viol.get("score", 0.0),
                                "is_violation": True,
                            }
                        )

            for action in action_violations:
                action_box = action.get("box", [0, 0, 0, 0])
                containment = calculate_box_containment(action_box, person_box)
                if containment >= self._containment_threshold:
                    person_actions.append(
                        {
                            "action": action.get("class", action.get("action", "")),
                            "score": action.get("score", 0.0),
                        }
                    )

            if infer_missing and self._is_absence_inference_eligible(person):
                for ppe_name in required_ppe:
                    if ppe_name not in detected_ppe and ppe_name not in missing_ppe:
                        missing_ppe.append(ppe_name)
                        detection_confidence[f"no_{ppe_name}"] = absence_confidence

            person["detected_ppe"] = detected_ppe
            person["missing_ppe"] = missing_ppe
            person["action_violations"] = person_actions
            person["detection_confidence"] = detection_confidence
            person["ppe_detections"] = ppe_dets
            person["is_violation"] = len(missing_ppe) > 0 or len(person_actions) > 0

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

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def reset_video_state(self) -> None:
        """Reset video tracking state."""
        self._frame_count = 0
        self._video_initialized = False
        self._last_track_ids = set()

        if self.sam3_segmenter and hasattr(self.sam3_segmenter, "reset_video_state"):
            self.sam3_segmenter.reset_video_state()
        if self.sam2_segmenter and hasattr(self.sam2_segmenter, "reset_video_state"):
            self.sam2_segmenter.reset_video_state()

    def reset_sam_state(self) -> None:
        """Alias for reset_video_state."""
        self.reset_video_state()

    def reset_sam2_state(self) -> None:
        """Alias for reset_video_state."""
        self.reset_video_state()

    def __repr__(self) -> str:
        return f"HybridDetector(initialized={self._initialized})"


_hybrid_detector: Optional[HybridDetector] = None


def get_hybrid_detector() -> HybridDetector:
    """Get singleton HybridDetector instance."""
    global _hybrid_detector
    if _hybrid_detector is None:
        _hybrid_detector = HybridDetector()
    return _hybrid_detector
