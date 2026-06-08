"""
Mask Utilities

Utility functions for mask operations and visualization.
Used by the hybrid YOLO + SAM 2/3 pipeline.
"""

import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Any

logger = logging.getLogger(__name__)


# Color scheme (BGR format for OpenCV)
COLORS = {
    "person": (255, 0, 0),  # Blue
    "helmet": (0, 215, 255),  # Gold
    "vest": (0, 255, 128),  # Green
    "boots": (42, 42, 165),  # Brown
    "goggles": (0, 255, 0),  # Green
    "safety goggles": (0, 255, 0),  # Green
    "face mask": (0, 255, 255),  # Yellow
    "lab coat": (255, 255, 0),  # Cyan
    "gloves": (255, 0, 255),  # Magenta
    "safety shoes": (0, 165, 255),  # Orange
    "violation": (0, 0, 255),  # Red
    "tentative": (0, 255, 255),  # Yellow
    "default": (128, 128, 128),  # Gray
}


def get_color(label: str, is_violation: bool = False) -> Tuple[int, int, int]:
    """
    Get color for a given label.

    Args:
        label: Detection label (e.g., "person", "safety goggles")
        is_violation: Whether this is a violation

    Returns:
        BGR color tuple
    """
    if is_violation:
        return COLORS["violation"]

    label_lower = label.lower()
    return COLORS.get(label_lower, COLORS["default"])


def calculate_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """
    Calculate Intersection over Union between two binary masks.

    Args:
        mask1: First binary mask (H, W) with values 0 or 1
        mask2: Second binary mask (H, W) with values 0 or 1

    Returns:
        IoU value between 0 and 1
    """
    if mask1.shape != mask2.shape:
        return 0.0

    # Ensure binary
    mask1 = (mask1 > 0).astype(np.uint8)
    mask2 = (mask2 > 0).astype(np.uint8)

    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()

    if union == 0:
        return 0.0

    return float(intersection) / float(union)


def calculate_mask_containment(inner_mask: np.ndarray, outer_mask: np.ndarray) -> float:
    """
    Calculate how much of the inner mask is contained within the outer mask.

    Args:
        inner_mask: The mask to check containment for (e.g., PPE mask)
        outer_mask: The containing mask (e.g., person mask)

    Returns:
        Containment ratio between 0 and 1
    """
    if inner_mask.shape != outer_mask.shape:
        return 0.0

    # Ensure binary
    inner = (inner_mask > 0).astype(np.uint8)
    outer = (outer_mask > 0).astype(np.uint8)

    inner_area = inner.sum()
    if inner_area == 0:
        return 0.0

    intersection = np.logical_and(inner, outer).sum()
    return float(intersection) / float(inner_area)


def calculate_box_containment(inner_box: List[float], outer_box: List[float]) -> float:
    """
    Calculate how much of the inner box is contained within the outer box.
    Fallback when masks are not available.

    Args:
        inner_box: [x1, y1, x2, y2] of the inner box
        outer_box: [x1, y1, x2, y2] of the outer box

    Returns:
        Containment ratio between 0 and 1
    """
    # Calculate intersection
    x1 = max(inner_box[0], outer_box[0])
    y1 = max(inner_box[1], outer_box[1])
    x2 = min(inner_box[2], outer_box[2])
    y2 = min(inner_box[3], outer_box[3])

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)
    inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])

    if inner_area <= 0:
        return 0.0

    return float(intersection) / float(inner_area)


def mask_to_polygon(mask: np.ndarray) -> List[List[int]]:
    """
    Convert a binary mask to polygon coordinates.

    Args:
        mask: Binary mask (H, W)

    Returns:
        List of polygon points [[x1, y1], [x2, y2], ...]
    """
    # Ensure uint8
    mask_uint8 = (mask > 0).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return []

    # Get the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # Simplify contour
    epsilon = 0.01 * cv2.arcLength(largest_contour, True)
    simplified = cv2.approxPolyDP(largest_contour, epsilon, True)

    # Convert to list of points
    points = simplified.reshape(-1, 2).tolist()
    return points


def mask_to_box(mask: np.ndarray) -> List[float]:
    """
    Convert a binary mask to a bounding box.

    Args:
        mask: Binary mask (H, W)

    Returns:
        Bounding box [x1, y1, x2, y2]
    """
    # Find non-zero pixels
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)

    if not rows.any() or not cols.any():
        return [0.0, 0.0, 0.0, 0.0]

    y_indices = np.where(rows)[0]
    x_indices = np.where(cols)[0]

    y1, y2 = float(y_indices[0]), float(y_indices[-1])
    x1, x2 = float(x_indices[0]), float(x_indices[-1])

    return [x1, y1, x2, y2]


def draw_mask_overlay(
    frame: np.ndarray,
    mask: np.ndarray,
    color: Tuple[int, int, int],
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw a semi-transparent mask overlay on a frame.

    Args:
        frame: BGR image (H, W, 3)
        mask: Binary mask (H, W)
        color: BGR color tuple
        alpha: Transparency (0 = transparent, 1 = opaque)

    Returns:
        Frame with mask overlay
    """
    if mask is None or mask.size == 0:
        logger.debug("[MaskUtils] draw_mask_overlay: mask is None or empty")
        return frame

    mask_pixels = int(np.sum(mask > 0))
    logger.info(
        f"[MaskUtils] draw_mask_overlay: mask_pixels={mask_pixels}, shape={mask.shape}, color={color}, alpha={alpha}"
    )

    # Ensure mask matches frame size
    if mask.shape[:2] != frame.shape[:2]:
        logger.info(
            f"[MaskUtils] Resizing mask from {mask.shape[:2]} to {frame.shape[:2]}"
        )
        mask = cv2.resize(mask.astype(np.uint8), (frame.shape[1], frame.shape[0]))

    # Create colored overlay
    overlay = frame.copy()
    mask_bool = mask > 0

    # Apply color to masked region
    overlay[mask_bool] = color

    # Blend with original frame
    result = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

    # Keep non-masked regions unchanged
    result[~mask_bool] = frame[~mask_bool]

    return result


def draw_detection(
    frame: np.ndarray,
    box: List[float],
    mask: Optional[np.ndarray],
    label: str,
    color: Tuple[int, int, int],
    is_violation: bool = False,
    show_mask: bool = True,
    mask_alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw a detection with box, mask, and label.

    Args:
        frame: BGR image
        box: Bounding box [x1, y1, x2, y2]
        mask: Optional binary mask
        label: Detection label
        color: BGR color (overridden if is_violation)
        is_violation: Whether this is a violation
        show_mask: Whether to draw the mask overlay
        mask_alpha: Mask transparency

    Returns:
        Annotated frame
    """
    result = frame.copy()

    # Override color for violations
    if is_violation:
        color = COLORS["violation"]

    # Draw mask overlay first (so box is on top)
    if show_mask and mask is not None:
        result = draw_mask_overlay(result, mask, color, mask_alpha)

    # Draw bounding box
    x1, y1, x2, y2 = [int(c) for c in box]
    thickness = 3 if is_violation else 2
    cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

    # Draw label background
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(result, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), color, -1)

    # Draw label text
    cv2.putText(
        result,
        label,
        (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return result


def draw_violation_indicator(
    frame: np.ndarray,
    person_box: List[float],
    missing_ppe: List[str],
) -> np.ndarray:
    """
    Draw a violation warning indicator below a person box.

    Args:
        frame: BGR image
        person_box: Person bounding box [x1, y1, x2, y2]
        missing_ppe: List of missing PPE items

    Returns:
        Annotated frame
    """
    if not missing_ppe:
        return frame

    result = frame.copy()
    x1, y1, x2, y2 = [int(c) for c in person_box]

    # Shorten PPE names
    short_names = []
    for ppe in missing_ppe:
        short = ppe.replace("safety ", "").replace("protective ", "")
        short_names.append(short)

    warning_text = "MISSING: " + ", ".join(short_names)

    # Draw warning below the box
    (text_w, text_h), _ = cv2.getTextSize(
        warning_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )

    # Warning background
    bg_y1 = y2 + 5
    bg_y2 = y2 + text_h + 15
    cv2.rectangle(
        result, (x1, bg_y1), (x1 + text_w + 8, bg_y2), COLORS["violation"], -1
    )

    # Warning text
    cv2.putText(
        result,
        warning_text,
        (x1 + 4, bg_y2 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    return result


def draw_person_with_ppe(
    frame: np.ndarray,
    person: Dict[str, Any],
    ppe_detections: List[Dict[str, Any]],
    show_masks: bool = True,
    mask_alpha: float = 0.4,
) -> np.ndarray:
    """
    Draw a person with their associated PPE detections.

    Args:
        frame: BGR image
        person: Person detection dict with box, mask, track_id, etc.
        ppe_detections: List of PPE detections associated with this person
        show_masks: Whether to draw mask overlays
        mask_alpha: Mask transparency

    Returns:
        Annotated frame
    """
    result = frame.copy()

    # Determine if person has violations
    is_violation = person.get("stable_violation", False) or person.get(
        "is_violation", False
    )
    track_state = person.get("track_state", "confirmed")
    track_id = person.get("track_id", "?")
    person_id = person.get("person_id", "")

    # Choose base color
    if track_state == "tentative":
        base_color = COLORS["tentative"]
    elif is_violation:
        base_color = COLORS["violation"]
    else:
        base_color = COLORS["person"]

    # Draw person mask
    person_mask = person.get("mask")
    if show_masks and person_mask is not None:
        mask_pixels = int(np.sum(person_mask > 0))
        logger.info(
            f"[MaskUtils] Drawing person mask for track {track_id}: {mask_pixels} pixels, shape={person_mask.shape}"
        )
        result = draw_mask_overlay(result, person_mask, base_color, mask_alpha)
    else:
        if show_masks:
            logger.debug(
                f"[MaskUtils] No mask for track {track_id} (show_masks={show_masks}, mask={person_mask is not None})"
            )

    # Draw person box
    person_box = person.get("box", [0, 0, 100, 100])
    x1, y1, x2, y2 = [int(c) for c in person_box]
    thickness = 3 if is_violation else 2
    cv2.rectangle(result, (x1, y1), (x2, y2), base_color, thickness)

    # Draw PPE detections
    for ppe in ppe_detections:
        ppe_label = ppe.get("label", "ppe")
        ppe_box = ppe.get("box", [0, 0, 0, 0])
        ppe_mask = ppe.get("mask")
        is_violation_ppe = ppe.get("is_violation", False)

        # Use red color for violation PPE boxes
        if is_violation_ppe:
            ppe_color = (0, 0, 255)  # Red for violations
        else:
            ppe_color = get_color(ppe_label)

        # Draw PPE mask
        if show_masks and ppe_mask is not None:
            result = draw_mask_overlay(result, ppe_mask, ppe_color, mask_alpha * 0.8)

        # Draw PPE box (thicker for violations)
        px1, py1, px2, py2 = [int(c) for c in ppe_box]
        thickness = 2 if is_violation_ppe else 1
        cv2.rectangle(result, (px1, py1), (px2, py2), ppe_color, thickness)

        # Add label for violations
        if is_violation_ppe:
            display_name = ppe.get("display_name", ppe_label)
            label = f"Missing: {display_name}"
            cv2.putText(
                result,
                label,
                (px1, py1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (0, 0, 255),
                1,
            )

    # Build label
    label = f"T{track_id}"
    if person_id and not person_id.startswith("track_"):
        label = f"{label}:{person_id}"

    if track_state == "tentative":
        label = f"{label} (detecting...)"

    # Draw label
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(result, (x1, y1 - text_h - 10), (x1 + text_w + 4, y1), base_color, -1)
    cv2.putText(
        result,
        label,
        (x1 + 2, y1 - 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    # Draw violation indicator
    if is_violation:
        missing_ppe = person.get("stable_missing_ppe", []) or person.get(
            "missing_ppe", []
        )
        result = draw_violation_indicator(result, person_box, missing_ppe)

    return result


def draw_frame_info(
    frame: np.ndarray,
    frame_number: int,
    num_tracks: int,
    num_violations: int = 0,
) -> np.ndarray:
    """
    Draw frame information overlay.

    Args:
        frame: BGR image
        frame_number: Current frame number
        num_tracks: Number of active tracks
        num_violations: Number of current violations

    Returns:
        Annotated frame
    """
    result = frame.copy()

    info_text = f"Frame: {frame_number} | Tracks: {num_tracks}"
    if num_violations > 0:
        info_text += f" | Violations: {num_violations}"

    cv2.putText(
        result,
        info_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
    )

    return result
