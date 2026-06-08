"""Application settings with environment variable support."""

from pydantic_settings import BaseSettings
from pydantic import Field, model_validator
from pathlib import Path
from typing import List, Optional, Dict


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    # App settings
    APP_NAME: str = "SafeSight PPE Safety"
    DEBUG: bool = True

    # API Settings
    API_V1_PREFIX: str = "/api/v1"
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://127.0.0.1:3000"]
    )

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./safesight.db"
    SQLALCHEMY_ECHO: bool = Field(default=False)

    # Detector settings
    DETECTOR_TYPE: str = Field(default="hybrid")
    PERSON_MODEL_NAME: str = Field(default="yolo11m.pt")
    PPE_CLASS_PROFILE: str = Field(default="construction")
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.5
    VIOLATION_CONFIDENCE_THRESHOLD: float = 0.3
    FACE_RECOGNITION_THRESHOLD: float = 0.6

    # SAM3 settings
    SAM3_MODEL: str = "facebook/sam3"
    SAM3_MODEL_ID: str = (
        "facebook/sam2.1-hiera-tiny"  # HuggingFace model for transformers
    )
    SAM3_MODEL_PATH: Optional[Path] = None
    USE_SAM3: bool = Field(default=False)

    # SAM2 settings
    SAM2_MODEL_TYPE: str = Field(default="sam2.1_hiera_base_plus")
    SAM2_MODEL_PATH: Optional[Path] = None
    USE_SAM2: bool = Field(default=True)
    USE_SAM2_VIDEO_PROPAGATION: bool = Field(default=True)
    SAM2_PROPAGATE_INTERVAL: int = Field(default=2)
    SAM2_SEGMENT_PPE: bool = Field(default=True)

    # Mask settings
    MASK_DENSITY_THRESHOLD: float = Field(default=0.1)
    MASK_CONTAINMENT_THRESHOLD: float = Field(default=0.5)
    SHOW_MASKS: bool = Field(default=True)
    MASK_ALPHA: float = Field(default=0.4)
    ALLOW_BOX_MASK_FALLBACK: bool = Field(default=False)

    # Mock mode
    USE_MOCK_DETECTOR: bool = False
    USE_MOCK_FACE: bool = False
    FAIL_ON_MISSING_MODELS: bool = True

    # Temporal filtering
    FRAME_SAMPLE_RATE: int = 10
    TEMPORAL_BUFFER_SIZE: int = 3
    TEMPORAL_VIOLATION_MIN_FRAMES: int = 2
    TEMPORAL_VIOLATION_MIN_FRAMES_CLEAR: int = 3  # Frames without violation to clear (hysteresis)
    TEMPORAL_FUSION_STRATEGY: str = Field(default="ema")
    TEMPORAL_EMA_ALPHA: float = Field(default=0.7)
    TEMPORAL_CONFIDENCE_THRESHOLD: float = Field(default=0.4)
    EVENT_DEDUP_CLEAR_GAP_FRAMES: int = Field(default=12)
    INFER_MISSING_PPE_FROM_ABSENCE: bool = Field(default=True)
    ABSENCE_INFERENCE_CONFIDENCE: float = Field(default=0.65)
    ABSENCE_MIN_PERSON_CONFIDENCE: float = Field(default=0.5)
    ABSENCE_MIN_PERSON_BOX_AREA_RATIO: float = Field(default=0.004)
    ABSENCE_MIN_PERSON_HEIGHT_RATIO: float = Field(default=0.1)
    ABSENCE_EDGE_MARGIN_RATIO: float = Field(default=0.01)

    # Live stream settings
    LIVE_STREAM_DISPLAY_FPS: int = Field(default=30)  # Display frame rate
    LIVE_STREAM_PROCESS_FPS: int = Field(default=4)   # ML processing rate (lower = faster)
    LIVE_STREAM_QUEUE_SIZE: int = Field(default=1)    # Keep newest frame under GPU load
    LIVE_STREAM_INTERPOLATE: bool = Field(default=False)  # Smooth bbox movement (optional)

    # Multi-scale detection
    MULTI_SCALE_ENABLED: bool = Field(default=True)
    MULTI_SCALE_FACTORS: List[float] = Field(default=[1.0, 1.5, 2.0])
    MULTI_SCALE_NMS_THRESHOLD: float = Field(default=0.5)

    # PPE configuration. Defaults are aligned with Ultralytics Construction-PPE.
    PPE_PROMPTS: List[str] = Field(
        default=["helmet", "vest", "gloves", "boots", "safety goggles"]
    )
    REQUIRED_PPE: List[str] = Field(default=["helmet", "vest"])

    PPE_CLASS_MAP: Dict[str, str] = Field(
        default={
            "helmet": "helmet",
            "gloves": "gloves",
            "vest": "vest",
            "boots": "boots",
            "goggles": "safety goggles",
            "no_helmet": "helmet",
            "no_goggle": "safety goggles",
            "no_gloves": "gloves",
            "no_boots": "boots",
        }
    )

    VIOLATION_CLASSES: List[str] = Field(
        default=["no_helmet", "no_goggle", "no_gloves", "no_boots"]
    )

    ACTION_VIOLATION_CLASSES: List[str] = Field(default=[])

    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    WEIGHTS_DIR: Optional[Path] = None
    DATA_DIR: Optional[Path] = None
    VIDEOS_DIR: Optional[Path] = None
    SNAPSHOTS_DIR: Optional[Path] = None
    PROCESSED_DIR: Optional[Path] = None
    ENABLE_SNAPSHOT_CAPTURE: bool = True
    YOLOV11_MODEL_PATH: Optional[Path] = None

    @model_validator(mode="after")
    def set_derived_paths(self):
        """Set derived paths after initialization."""
        if self.WEIGHTS_DIR is None:
            object.__setattr__(self, "WEIGHTS_DIR", self.BASE_DIR / "weights")

        if self.DATA_DIR is None:
            object.__setattr__(self, "DATA_DIR", self.BASE_DIR.parent / "data")

        if self.VIDEOS_DIR is None:
            object.__setattr__(self, "VIDEOS_DIR", self.DATA_DIR / "videos")

        if self.SNAPSHOTS_DIR is None:
            object.__setattr__(self, "SNAPSHOTS_DIR", self.DATA_DIR / "snapshots")

        if self.PROCESSED_DIR is None:
            object.__setattr__(self, "PROCESSED_DIR", self.DATA_DIR / "processed")

        if self.SAM2_MODEL_PATH is None:
            object.__setattr__(
                self,
                "SAM2_MODEL_PATH",
                self.WEIGHTS_DIR / "sam2" / "sam2.1_hiera_base_plus.pt",
            )

        if self.SAM3_MODEL_PATH is None:
            object.__setattr__(
                self, "SAM3_MODEL_PATH", self.WEIGHTS_DIR / "sam3" / "sam3.pt"
            )

        # Set YOLOv11 model path (prefer .pt over .onnx)
        if self.YOLOV11_MODEL_PATH is None:
            pt_path = self.WEIGHTS_DIR / "ppe_detector" / "best.pt"
            onnx_path = self.WEIGHTS_DIR / "ppe_detector" / "best.onnx"

            if pt_path.exists():
                object.__setattr__(self, "YOLOV11_MODEL_PATH", pt_path)
            elif onnx_path.exists():
                object.__setattr__(self, "YOLOV11_MODEL_PATH", onnx_path)
            else:
                object.__setattr__(self, "YOLOV11_MODEL_PATH", pt_path)

        return self

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
