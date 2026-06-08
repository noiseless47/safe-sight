from sqlalchemy import Column, String, DateTime, Integer
from sqlalchemy.sql import func
from uuid import uuid4
from ..core.database import Base


def generate_uuid():
    return str(uuid4())


class VideoSource(Base):
    """
    Tracks video sources processed by the system.

    This model stores metadata about uploaded videos, webcam sessions,
    or RTSP streams that have been processed for PPE detection.
    """

    __tablename__ = "video_sources"

    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)  # Display name (e.g., "Lab Camera 1")
    source_type = Column(String, nullable=False)  # "uploaded", "webcam", "rtsp"
    path = Column(String, nullable=True)  # File path or stream URL
    created_at = Column(DateTime, server_default=func.now())

    # Processing stats
    total_frames = Column(Integer, default=0)
    processed_frames = Column(Integer, default=0)
    total_violations = Column(Integer, default=0)
    total_persons_detected = Column(Integer, default=0)

    # Processing status
    status = Column(
        String, default="pending"
    )  # "pending", "processing", "completed", "failed"
