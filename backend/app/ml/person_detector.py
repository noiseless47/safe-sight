"""
Person Detector

Person detector using a pretrained Ultralytics YOLO model with native tracking.
Auto-downloads the model on first run.
Only detects COCO class 0 (person).
Supports native Ultralytics tracking via model.track().
"""

import numpy as np
from typing import List, Dict, Any, Optional
import os
import shutil
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

try:
    import requests
    from tqdm import tqdm
except ImportError:
    requests = None
    tqdm = None

from ..core.config import settings


class PersonDetector:
    """
    Person detector using a pretrained Ultralytics model with native tracking.

    Uses a COCO-pretrained model which detects 80 classes,
    but we only extract class 0 (person).
    Supports native Ultralytics tracking for consistent track_ids across frames.
    """

    PERSON_CLASS_ID = 0  # COCO class ID for person

    def __init__(self):
        self.model: Optional[Any] = None
        self._initialized = False
        self.confidence_threshold = settings.DETECTION_CONFIDENCE_THRESHOLD
        self.device = "cuda" if self._cuda_available() else "cpu"

    def _cuda_available(self) -> bool:
        """Check if CUDA is available."""
        try:
            import torch

            return torch.cuda.is_available()
        except ImportError:
            return False

    def initialize(self) -> None:
        """
        Initialize the configured person detector model.
        Auto-downloads if not present.
        """
        if self._initialized:
            return

        if YOLO is None:
            raise ImportError(
                "ultralytics is required for PersonDetector. "
                "Install with: pip install ultralytics"
            )

        model_name = getattr(settings, "PERSON_MODEL_NAME", "yolo11m.pt")
        model_path = settings.WEIGHTS_DIR / "person_detector" / model_name
        model_path.parent.mkdir(parents=True, exist_ok=True)

        if model_path.exists():
            print(f"Loading person detector from: {model_path}")
            try:
                self.model = YOLO(str(model_path))
            except Exception as e:
                print(
                    f"Warning: Failed to load local model ({e}), attempting download..."
                )
                # Delete corrupted file
                model_path.unlink(missing_ok=True)
                # Fall through to download logic below
        else:
            # Auto-download from ultralytics hub
            print(f"Downloading {model_name} for person detection...")
            max_retries = 1
            last_error = None

            # First try ultralytics auto-download
            for attempt in range(max_retries + 1):
                try:
                    self.model = YOLO(model_name)
                    downloaded_path = Path(model_name)
                    if downloaded_path.exists() and not model_path.exists():
                        shutil.move(str(downloaded_path), str(model_path))
                        self.model = YOLO(str(model_path))
                    break  # Success, exit retry loop
                except Exception as e:
                    last_error = e
                    if attempt < max_retries:
                        print(f"Ultralytics download attempt {attempt + 1} failed: {e}")
                        print("Clearing cache and retrying...")
                        # Try clearing ultralytics cache
                        cache_dir = os.path.expanduser("~/.cache/ultralytics")
                        if os.path.exists(cache_dir):
                            try:
                                shutil.rmtree(cache_dir)
                                print("Cache cleared, retrying download...")
                            except Exception as e2:
                                print(f"Warning: Could not clear cache: {e2}")
                        # Also try clearing any partial downloads in weights dir
                        weights_cache = settings.WEIGHTS_DIR / "person_detector"
                        if weights_cache.exists():
                            for file in weights_cache.glob("*.pt"):
                                try:
                                    file.unlink()
                                    print(f"Removed potentially corrupted file: {file}")
                                except Exception:
                                    pass

            # If ultralytics download failed, try manual download
            if self.model is None:
                print("Ultralytics auto-download failed, attempting manual download...")
                model_url = (
                    "https://github.com/ultralytics/assets/releases/latest/download/"
                    f"{model_name}"
                )

                if requests is None:
                    raise RuntimeError(
                        f"Failed to download {model_name}. "
                        f"Ultralytics error: {last_error}. "
                        f"Manual download requires 'requests' package. "
                        f"Install with: pip install requests tqdm"
                    ) from last_error

                try:
                    print(f"Downloading from: {model_url}")
                    response = requests.get(model_url, stream=True, timeout=30)
                    response.raise_for_status()

                    total_size = int(response.headers.get("content-length", 0))
                    model_path.parent.mkdir(parents=True, exist_ok=True)

                    # Download with progress bar if tqdm available
                    if tqdm is not None:
                        with (
                            open(model_path, "wb") as f,
                            tqdm(
                                desc=f"Downloading {model_name}",
                                total=total_size,
                                unit="B",
                                unit_scale=True,
                                unit_divisor=1024,
                            ) as bar,
                        ):
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    bar.update(len(chunk))
                    else:
                        # Simple download without progress bar
                        with open(model_path, "wb") as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)

                    print(f"Model downloaded successfully to: {model_path}")
                    # Now load the manually downloaded model
                    self.model = YOLO(str(model_path))

                except Exception as e:
                    # Clean up partial download
                    if model_path.exists():
                        try:
                            model_path.unlink()
                        except Exception:
                            pass

                    raise RuntimeError(
                        f"Failed to download {model_name}. "
                        f"Ultralytics error: {last_error}. "
                        f"Manual download error: {e}. "
                        f"Please check your internet connection and try again. "
                        f"Or manually download from: {model_url} "
                        f"and place it at: {model_path}"
                    ) from e

        # Move to appropriate device
        self.model.to(self.device)

        self._initialized = True
        print(f"PersonDetector initialized on {self.device}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect persons in a frame.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            List of person detections, each with:
                - id: Detection index
                - box: [x1, y1, x2, y2]
                - score: Confidence score
                - label: "person"
                - mask: None (SAM2 will add masks later)
        """
        if not self._initialized:
            self.initialize()

        if self.model is None:
            return []

        # Run inference
        results = self.model(
            frame,
            conf=self.confidence_threshold,
            classes=[self.PERSON_CLASS_ID],  # Only detect persons
            verbose=False,
            save=False,
        )

        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for i, box in enumerate(boxes):
                # Extract coordinates and convert to native Python types
                xyxy = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())
                cls = int(box.cls[0].cpu().numpy())

                # Only include person class (should already be filtered, but double-check)
                if cls != self.PERSON_CLASS_ID:
                    continue

                persons.append(
                    {
                        "id": i,
                        "box": [
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ],
                        "score": conf,
                        "label": "person",
                        "mask": None,  # Will be filled by SAM2
                    }
                )

        return persons

    def detect_with_tracking(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect persons with YOLOv8 native tracking.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            List of person detections, each with:
                - id: Detection index
                - track_id: Consistent track ID across frames (from YOLOv8)
                - box: [x1, y1, x2, y2]
                - score: Confidence score
                - label: "person"
        """
        if not self._initialized:
            self.initialize()

        if self.model is None:
            return []

        if frame is None or frame.size == 0:
            print("Warning: PersonDetector received empty frame")
            return []

        try:
            # Use Ultralytics native tracking
            results = self.model.track(
                frame,
                persist=True,  # Maintain track_ids across frames
                conf=self.confidence_threshold,
                classes=[self.PERSON_CLASS_ID],  # Only detect persons
                verbose=False,
                save=False,
            )
        except Exception as e:
            print(f"Error in PersonDetector.detect_with_tracking: {e}")
            import traceback

            traceback.print_exc()
            return []

        persons = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            # Get track_ids (None if tracking not available)
            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()
            else:
                track_ids = [None] * len(boxes)

            for i, (box, track_id) in enumerate(zip(boxes.xyxy, track_ids)):
                # Extract coordinates and convert to native Python types
                xyxy = box.cpu().numpy()
                conf = float(boxes.conf[i].cpu().numpy())
                cls = int(boxes.cls[i].cpu().numpy())

                # Only include person class (should already be filtered, but double-check)
                if cls != self.PERSON_CLASS_ID:
                    continue

                persons.append(
                    {
                        "id": i,
                        "track_id": int(track_id) if track_id is not None else None,
                        "box": [
                            float(xyxy[0]),
                            float(xyxy[1]),
                            float(xyxy[2]),
                            float(xyxy[3]),
                        ],
                        "score": conf,
                        "label": "person",
                    }
                )

        return persons

    def __repr__(self) -> str:
        return f"PersonDetector(initialized={self._initialized}, device={self.device})"


# Singleton instance
_person_detector: Optional[PersonDetector] = None


def get_person_detector() -> PersonDetector:
    """Get singleton PersonDetector instance."""
    global _person_detector
    if _person_detector is None:
        _person_detector = PersonDetector()
    return _person_detector
