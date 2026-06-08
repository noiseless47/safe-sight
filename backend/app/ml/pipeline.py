"""
Main Detection Pipeline

Orchestrates detection, tracking, and temporal filtering.
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Generator
from datetime import datetime
from uuid import uuid4

from .detector_factory import get_detector
from .temporal_filter import get_temporal_filter
from .mask_utils import draw_person_with_ppe, draw_frame_info
from ..core.config import settings

logger = logging.getLogger(__name__)


class DetectionPipeline:
    """
    Main pipeline orchestrating detection and tracking.

    Flow:
    1. Sample frames at target FPS
    2. Detect persons and PPE
    3. Associate PPE with persons
    4. Apply temporal filtering
    5. Generate compliance events
    """

    def __init__(self):
        self.detector = get_detector()
        self.temporal_filter = get_temporal_filter()

        self.target_fps = settings.FRAME_SAMPLE_RATE
        self.frame_count = 0
        self.current_video_source: Optional[str] = None

        self.show_masks = getattr(settings, "SHOW_MASKS", True)
        self.mask_alpha = getattr(settings, "MASK_ALPHA", 0.4)
        self.use_confidence_fusion = (
            getattr(settings, "TEMPORAL_FUSION_STRATEGY", "ema") != "none"
        )

    def initialize(self):
        """Initialize all ML models."""
        self.detector.initialize()

    def process_frame(
        self, frame: np.ndarray, video_source: str = "video"
    ) -> Dict[str, Any]:
        """Process a single frame through the pipeline."""
        # Reset state on video change
        if video_source != self.current_video_source:
            self.current_video_source = video_source
            if hasattr(self.detector, "reset_video_state"):
                self.detector.reset_video_state()
            self.temporal_filter.clear_all()
            self.frame_count = 0

        self.frame_count += 1
        timestamp = datetime.now()

        result = {
            "frame_number": self.frame_count,
            "timestamp": timestamp.isoformat(),
            "persons": [],
            "violations": [],
            "events": [],
            "tracks": [],
            "annotated_frame": None,
        }

        # Detect persons and PPE
        detections = self.detector.detect(frame)
        persons = detections.get("persons", [])
        ppe_detections = detections.get("ppe_detections", {})
        violation_detections = detections.get("violation_detections", {})
        action_violations = detections.get("action_violations", [])

        # Associate PPE with persons
        persons = self.detector.associate_ppe_to_persons(
            persons, ppe_detections, violation_detections, action_violations
        )

        # Process each person
        for person in persons:
            track_id = person.get("track_id")
            person_id = (
                f"person_{track_id}"
                if track_id is not None
                else f"track_{person.get('id', 0)}"
            )
            if track_id is None:
                track_id = person.get("id", 0)

            person_result = {
                "person_id": person_id,
                "track_id": track_id,
                "box": person.get("box", [0, 0, 0, 0]),
                "mask": person.get("mask"),
                "detected_ppe": person.get("detected_ppe", []),
                "missing_ppe": person.get("missing_ppe", []),
                "action_violations": person.get("action_violations", []),
                "detection_confidence": person.get("detection_confidence", {}),
                "ppe_detections": person.get("ppe_detections", []),
                "is_violation": person.get("is_violation", False),
            }

            # Apply temporal filtering
            detection_confidence = person_result.get("detection_confidence", {})
            if self.use_confidence_fusion and detection_confidence:
                filter_result = self.temporal_filter.update_with_confidence(
                    person_id, detection_confidence
                )
                person_result["fused_confidence"] = filter_result.get(
                    "fused_confidence", {}
                )
            else:
                filter_result = self.temporal_filter.update(
                    person_id, person_result.get("missing_ppe", [])
                )

            person_result["stable_violation"] = filter_result["is_violation"]
            person_result["stable_missing_ppe"] = filter_result["stable_missing_ppe"]

            action_viols = person_result.get("action_violations", [])
            has_action_violation = len(action_viols) > 0

            # Generate event if violation detected
            if filter_result["is_violation"] or has_action_violation:
                all_violations = list(filter_result["stable_missing_ppe"])
                for av in action_viols:
                    all_violations.append(f"{av['action']} in lab")

                event = {
                    "id": str(uuid4()),
                    "person_id": person_id,
                    "track_id": track_id,
                    "timestamp": timestamp.isoformat(),
                    "video_source": video_source,
                    "frame_number": self.frame_count,
                    "detected_ppe": person_result.get("detected_ppe", []),
                    "missing_ppe": filter_result["stable_missing_ppe"],
                    "action_violations": [av["action"] for av in action_viols],
                    "is_violation": True,
                    "detection_confidence": person_result.get(
                        "detection_confidence", {}
                    ),
                    "fused_confidence": person_result.get("fused_confidence", {}),
                }
                result["events"].append(event)
                result["violations"].append(
                    {
                        "person_id": person_id,
                        "track_id": track_id,
                        "missing_ppe": filter_result["stable_missing_ppe"],
                        "action_violations": [av["action"] for av in action_viols],
                        "box": person_result.get("box"),
                    }
                )

            result["persons"].append(person_result)
            result["tracks"].append(
                {
                    "track_id": track_id,
                    "person_id": person_id,
                    "box": person_result.get("box"),
                }
            )

        # Annotate frame
        result["annotated_frame"] = self._annotate_frame(
            frame, result["persons"], violation_detections, action_violations
        )

        return result

    def _annotate_frame(
        self,
        frame: np.ndarray,
        persons: List[Dict],
        violation_detections: Optional[Dict] = None,
        action_violations: Optional[List] = None,
    ) -> np.ndarray:
        """Draw annotations on frame."""
        annotated = frame.copy()
        num_violations = sum(1 for p in persons if p.get("stable_violation", False))

        # Log mask status for each person
        persons_with_masks = sum(1 for p in persons if p.get("mask") is not None)
        logger.info(
            f"[Pipeline] _annotate_frame: {len(persons)} persons, {persons_with_masks} with masks, show_masks={self.show_masks}"
        )

        for person in persons:
            track_id = person.get("track_id", "?")
            has_mask = person.get("mask") is not None
            if has_mask:
                mask_pixels = int(np.sum(person["mask"] > 0))
                logger.info(
                    f"[Pipeline] Person track {track_id}: mask={mask_pixels} pixels"
                )
            else:
                logger.debug(f"[Pipeline] Person track {track_id}: no mask")

            ppe_detections = person.get("ppe_detections", [])
            annotated = draw_person_with_ppe(
                annotated,
                person,
                ppe_detections,
                show_masks=self.show_masks,
                mask_alpha=self.mask_alpha,
            )

        for person in persons:
            ppe_detections = person.get("ppe_detections", [])
            annotated = draw_person_with_ppe(
                annotated,
                person,
                ppe_detections,
                show_masks=self.show_masks,
                mask_alpha=self.mask_alpha,
            )

        # Draw violation boxes
        if violation_detections:
            for ppe_type, viol_list in violation_detections.items():
                for viol in viol_list:
                    box = viol.get("box", [0, 0, 0, 0])
                    if box != [0, 0, 0, 0]:
                        x1, y1, x2, y2 = [int(c) for c in box]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(
                            annotated,
                            ppe_type,
                            (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1,
                        )

        # Draw action violations
        if action_violations:
            for action in action_violations:
                box = action.get("box", [0, 0, 0, 0])
                if box != [0, 0, 0, 0]:
                    x1, y1, x2, y2 = [int(c) for c in box]
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    label = f"Action: {action.get('action', 'violation')}"
                    cv2.putText(
                        annotated,
                        label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 0, 255),
                        1,
                    )

        annotated = draw_frame_info(
            annotated, self.frame_count, len(persons), num_violations
        )
        return annotated

    def process_video(self, video_path: str) -> Generator[Dict[str, Any], None, None]:
        """Process a video file frame by frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(video_fps / self.target_fps))
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_skip == 0:
                    result = self.process_frame(frame, video_source=video_path)
                    result["video_frame_idx"] = frame_idx
                    yield result

                frame_idx += 1
        finally:
            cap.release()

    def load_known_persons(self, persons: List):
        """Load known persons (legacy, not used with YOLOv8 tracking)."""
        pass

    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            "frame_count": self.frame_count,
            "current_video": self.current_video_source,
        }

    def reset(self):
        """Reset pipeline state."""
        self.temporal_filter.clear_all()
        self.frame_count = 0
        self.current_video_source = None
        if hasattr(self.detector, "reset_video_state"):
            self.detector.reset_video_state()


# Singleton
_pipeline = None


def get_pipeline() -> DetectionPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = DetectionPipeline()
    return _pipeline
