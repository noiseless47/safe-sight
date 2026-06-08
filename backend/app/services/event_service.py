from typing import Any, List, Optional
from datetime import datetime

import cv2
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.event import ComplianceEvent


class EventService:
    """Service for persisting compliance events."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def create_event(
        self,
        person_id: Optional[str],
        track_id: Optional[int],
        timestamp: datetime,
        video_source: str,
        frame_number: int,
        detected_ppe: List[str],
        missing_ppe: List[str],
        is_violation: bool,
        detection_confidence: Optional[dict] = None,
        snapshot_path: Optional[str] = None,
        start_frame: Optional[int] = None,
        action_violations: Optional[List[str]] = None,
        event_id: Optional[str] = None,
    ) -> ComplianceEvent:
        """Create and persist a compliance event."""
        event = ComplianceEvent(
            id=event_id,
            person_id=person_id,
            track_id=track_id,
            timestamp=timestamp,
            video_source=video_source,
            frame_number=frame_number,
            detected_ppe=detected_ppe,
            missing_ppe=missing_ppe,
            action_violations=action_violations or [],
            is_violation=is_violation,
            detection_confidence=detection_confidence or {},
            snapshot_path=snapshot_path,
            start_frame=start_frame or frame_number,
            is_ongoing=True,
            duration_frames=1,
        )
        self.session.add(event)
        return event

    async def close_event(
        self,
        event_id: str,
        end_frame: int,
        end_timestamp: datetime,
        final_missing_ppe: Optional[List[str] | dict] = None,
    ) -> Optional[ComplianceEvent]:
        """
        Close an ongoing event when the violation ends.

        Updates the event with end_frame, end_timestamp, duration,
        and optionally updates the missing_ppe list with the accumulated union.
        """
        result = await self.session.execute(
            select(ComplianceEvent).where(ComplianceEvent.id == event_id)
        )
        event = result.scalar_one_or_none()

        if event:
            event.end_frame = end_frame
            event.end_timestamp = end_timestamp
            event.is_ongoing = False

            # Update missing_ppe with the full accumulated set if provided
            if final_missing_ppe:
                event.missing_ppe = self._normalize_missing_ppe(final_missing_ppe)

            if event.start_frame is not None:
                event.duration_frames = end_frame - event.start_frame + 1
            else:
                event.duration_frames = end_frame - event.frame_number + 1

        return event

    @staticmethod
    def _normalize_missing_ppe(value: Any) -> List[str]:
        if isinstance(value, dict):
            items = value.get("ppe", [])
        else:
            items = value

        if not isinstance(items, list):
            return []

        return [str(item) for item in items]

    async def get_event(self, event_id: str) -> Optional[ComplianceEvent]:
        """Get an event by ID."""
        result = await self.session.execute(
            select(ComplianceEvent).where(ComplianceEvent.id == event_id)
        )
        return result.scalar_one_or_none()

    async def save_snapshot(
        self, frame, snapshots_dir, filename: str, quality: int = 85
    ) -> str:
        """Save a snapshot image and return its path."""
        snapshots_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshots_dir / filename
        cv2.imwrite(str(snapshot_path), frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return str(snapshot_path)
