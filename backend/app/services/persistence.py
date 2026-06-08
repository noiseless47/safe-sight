from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from uuid import uuid4

import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import settings
from ..core.websocket import manager as ws_manager
from ..services.event_service import EventService
from ..services.person_service import PersonService
from ..services.deduplication import get_deduplication_manager, DeduplicationManager


def to_python_type(value):
    """Convert numpy types to native Python types."""
    if isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, np.bool_):
        return bool(value)
    return value


class PersistenceManager:
    """Coordinates persistence of events and person records with deduplication."""

    def __init__(self, session: AsyncSession):
        self.session = session
        self.event_service = EventService(session)
        self.person_service = PersonService(session)
        self.dedup_manager = get_deduplication_manager()

    async def persist_frame_results(
        self, result: Dict[str, Any], snapshot_frame
    ) -> Dict[str, Any]:
        """
        Persist events and update person records from a frame result.

        Uses deduplication to prevent creating duplicate events for ongoing violations.
        Only creates new events when:
        - A new violation starts
        - The violation type changes (different missing PPE)

        Returns:
            Dict with 'created_events' and 'closed_events' counts
        """
        persons = result.get("persons", [])
        frame_number = to_python_type(result.get("frame_number", 0))
        timestamp_str = result.get("timestamp")
        timestamp = (
            datetime.fromisoformat(timestamp_str) if timestamp_str else datetime.now()
        )

        created_events = 0
        closed_events = 0

        # Process each person in the frame
        for person in persons:
            person_id = person.get("person_id")
            if not person_id:
                continue

            # Convert track_id to native Python int
            track_id = to_python_type(person.get("track_id"))

            # Skip track-only IDs for face-recognized persons check
            if not person_id.startswith("track_"):
                face_embedding = person.get("face_embedding")
                await self.person_service.get_or_create_person(
                    person_id, face_embedding
                )

            # Get violation info from temporal filter results
            is_stable_violation = person.get("stable_violation", False)
            stable_missing_ppe = person.get("stable_missing_ppe", [])
            action_violations = person.get("action_violations", [])
            video_source = result.get("video_source", "unknown")

            # Action violations (Drinking/Eating) are immediate violations
            has_action_violation = len(action_violations) > 0

            # Combine missing PPE with action violations for deduplication
            all_violations = list(stable_missing_ppe) if is_stable_violation else []
            for av in action_violations:
                all_violations.append(f"action:{av.get('action', 'unknown')}")

            # Use deduplication to determine if we should create an event
            should_create, ended_event_id, reason, final_ppe = (
                self.dedup_manager.should_create_event(
                    person_id=person_id,
                    video_source=video_source,
                    missing_ppe=all_violations,
                    frame_number=frame_number,
                )
            )

            # Close ended event if any
            if ended_event_id:
                # Extract PPE and actions from final violations dict
                final_ppe_list = []
                final_actions_list = []

                if isinstance(final_ppe, dict):
                    final_ppe_list = final_ppe.get("ppe", [])
                    final_actions_list = final_ppe.get("actions", [])
                elif isinstance(final_ppe, list):
                    # Backward compatibility: if it's still a list, separate them
                    for item in final_ppe:
                        if item.startswith("action:"):
                            final_actions_list.append(item.replace("action:", ""))
                        else:
                            final_ppe_list.append(item)

                await self.event_service.close_event(
                    event_id=ended_event_id,
                    end_frame=frame_number - 1,  # Ended on previous frame
                    end_timestamp=timestamp,
                    final_missing_ppe=final_ppe_list,
                )
                closed_events += 1

            # Create new event if needed
            if should_create and (is_stable_violation or has_action_violation):
                # Generate event ID
                event_id = str(uuid4())

                # Save snapshot for new violations
                snapshot_path = None
                if settings.ENABLE_SNAPSHOT_CAPTURE and snapshot_frame is not None:
                    filename = f"{person_id}_{frame_number}.jpg"
                    snapshot_path = await self.event_service.save_snapshot(
                        snapshot_frame, settings.SNAPSHOTS_DIR, filename
                    )

                # Prepare action violations for storage
                action_violation_names = [
                    av.get("action", "unknown") for av in action_violations
                ]

                # Create the event
                event = await self.event_service.create_event(
                    person_id=person_id,
                    track_id=track_id,
                    timestamp=timestamp,
                    video_source=video_source,
                    frame_number=frame_number,
                    detected_ppe=person.get("detected_ppe", []),
                    missing_ppe=stable_missing_ppe,
                    action_violations=action_violation_names,
                    is_violation=True,
                    detection_confidence=person.get("detection_confidence"),
                    snapshot_path=snapshot_path,
                    start_frame=frame_number,
                    event_id=event_id,
                )

                # Register with deduplication manager
                self.dedup_manager.register_event(
                    event_id=event_id,
                    person_id=person_id,
                    video_source=video_source,
                    missing_ppe=all_violations,
                    frame_number=frame_number,
                    timestamp=timestamp,
                )

                # Update person stats
                if not person_id.startswith("track_"):
                    await self.person_service.increment_event_counts(person_id, True)

                # Broadcast event via WebSocket
                await ws_manager.broadcast(
                    {
                        "type": "violation",
                        "title": "New Safety Violation",
                        "message": f"Person {person_id} missing {', '.join(all_violations)}",
                        "timestamp": timestamp.isoformat(),
                        "severity": "warning",
                        "event_id": event.id,
                        "person_id": person_id,
                        "missing_ppe": all_violations,
                    }
                )

                created_events += 1

        await self.session.commit()

        return {
            "created_events": created_events,
            "closed_events": closed_events,
        }

    async def finalize_video_processing(self, video_source: str) -> int:
        """
        Finalize all active violations when video processing completes.

        Closes any ongoing events that weren't explicitly ended.
        Returns the number of events closed.
        """
        events_to_close = self.dedup_manager.finalize_video(video_source)

        for event_id, last_frame, final_ppe in events_to_close:
            if isinstance(final_ppe, dict):
                final_missing_ppe = final_ppe.get("ppe", [])
            else:
                final_missing_ppe = final_ppe

            await self.event_service.close_event(
                event_id=event_id,
                end_frame=last_frame,
                end_timestamp=datetime.now(),
                final_missing_ppe=final_missing_ppe,
            )

        await self.session.commit()
        return len(events_to_close)
