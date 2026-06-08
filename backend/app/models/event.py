from sqlalchemy import (
    Column,
    String,
    DateTime,
    Boolean,
    Float,
    ForeignKey,
    JSON,
    Integer,
)
from sqlalchemy.sql import func
from uuid import uuid4
from ..core.database import Base


def generate_uuid():
    return str(uuid4())


class ComplianceEvent(Base):
    __tablename__ = "compliance_events"

    id = Column(String, primary_key=True, default=generate_uuid)
    person_id = Column(String, ForeignKey("persons.id"), nullable=True)
    track_id = Column(Integer, nullable=True)
    timestamp = Column(DateTime, server_default=func.now())
    video_source = Column(String, nullable=True)
    frame_number = Column(Integer, default=0)

    # Detection results
    detected_ppe = Column(JSON, default=list)  # ["goggles", "lab_coat"]
    missing_ppe = Column(JSON, default=list)  # ["mask"]
    action_violations = Column(JSON, default=list)  # ["drinking", "eating"]

    # Violation info
    is_violation = Column(Boolean, default=False)

    # Confidence scores
    detection_confidence = Column(JSON, default=dict)

    # Optional snapshot
    snapshot_path = Column(String, nullable=True)

    # Event deduplication fields - track violation duration
    start_frame = Column(Integer, nullable=True)  # Frame when violation started
    end_frame = Column(Integer, nullable=True)  # Frame when violation ended (null if ongoing)
    end_timestamp = Column(DateTime, nullable=True)  # When violation ended
    duration_frames = Column(Integer, default=1)  # Total frames violation lasted
    is_ongoing = Column(Boolean, default=True)  # Whether violation is still active
