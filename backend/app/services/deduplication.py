"""
Event Deduplication Service

Prevents creating duplicate events for ongoing violations.
Only creates new events when:
- A new violation starts (person was compliant, now has violations)
- The violation type changes significantly (different PPE items or actions)
- A violation ends (person becomes compliant)

Updates existing events when violations end with end_frame and duration.

Improved logic:
- Tracks PPE and action violations separately
- Merges them into a single event per person
- Handles transitions smoothly without creating duplicates
"""

from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime

from ..core.config import settings


@dataclass
class ActiveViolation:
    """Tracks an active ongoing violation."""

    event_id: str
    person_id: str
    missing_ppe: Set[str]  # PPE items missing
    actions: Set[str]  # Action violations (eating, drinking)
    start_frame: int
    start_timestamp: datetime
    last_frame: int
    video_source: str


class DeduplicationManager:
    """
    Manages event deduplication to prevent flooding the database
    with duplicate events for the same ongoing violation.
    """

    def __init__(self):
        # Track active violations: (person_id, video_source) -> ActiveViolation
        self.active_violations: Dict[Tuple[str, str], ActiveViolation] = {}

    def should_create_event(
        self,
        person_id: str,
        video_source: str,
        missing_ppe: List[str],
        frame_number: int,
    ) -> Tuple[bool, Optional[str], Optional[str], Optional[Dict[str, List[str]]]]:
        """
        Determine if a new event should be created.

        Args:
            person_id: Unique person identifier
            video_source: Video file or camera source
            missing_ppe: Combined list of violations (PPE + actions with 'action:' prefix)
            frame_number: Current frame number

        Returns:
            (should_create, ended_event_id, reason, final_violations)
            where final_violations is {"ppe": [...], "actions": [...]}
        """
        key = (person_id, video_source)

        # Separate PPE violations from action violations
        current_ppe = set()
        current_actions = set()

        for item in missing_ppe:
            if item.startswith("action:"):
                current_actions.add(item.replace("action:", ""))
            else:
                current_ppe.add(item)

        active = self.active_violations.get(key)
        has_current_violation = bool(current_ppe or current_actions)

        # Case 1: No active violation and no current violation
        if not active and not has_current_violation:
            return False, None, None, None

        # Case 2: No active violation but new violation detected
        if not active and has_current_violation:
            return True, None, "new", None

        # Case 3: Active violation but person is now compliant
        if active and not has_current_violation:
            clear_gap = getattr(settings, "EVENT_DEDUP_CLEAR_GAP_FRAMES", 12)
            if frame_number - active.last_frame <= clear_gap:
                return False, None, "gap_tolerated", None

            ended_event_id = active.event_id
            final_violations = {
                "ppe": list(active.missing_ppe),
                "actions": list(active.actions)
            }
            del self.active_violations[key]
            return False, ended_event_id, "ended", final_violations

        # Case 4: Active violation and still has violations
        if active and has_current_violation:
            # Check if violations have changed significantly
            ppe_changed = not self._is_similar_violation(current_ppe, active.missing_ppe)
            actions_changed = current_actions != active.actions

            # Use more lenient logic: only create new event if BOTH types change completely
            # or if one was empty and now has violations (significant change)
            significant_change = False

            # Significant change scenarios:
            # 1. Had no PPE violations, now has PPE violations (AND it's not a subset)
            # 2. Had PPE violations, now has completely different PPE violations
            # 3. Action violations changed (these are instant, so any change is significant)

            if ppe_changed and current_ppe and active.missing_ppe:
                # Both had PPE, but they're completely different
                significant_change = True

            if significant_change:
                # End old event, create new one
                ended_event_id = active.event_id
                final_violations = {
                    "ppe": list(active.missing_ppe),
                    "actions": list(active.actions)
                }
                del self.active_violations[key]
                return True, ended_event_id, "changed", final_violations
            else:
                # Continue same event, update the violations
                # Use union to track all violations that occurred
                active.missing_ppe = active.missing_ppe.union(current_ppe)
                active.actions = current_actions  # Actions are instant, use current
                active.last_frame = frame_number
                return False, None, "continuing", None

        return False, None, None, None

    def _is_similar_violation(self, set1: Set[str], set2: Set[str]) -> bool:
        """
        Check if two PPE violation sets are similar enough to be the same event.

        Similar if:
        - They're equal
        - One is a subset of the other (handles occlusions)
        - They overlap significantly (at least 50%)
        """
        if not set1 and not set2:
            return True
        if not set1 or not set2:
            return False
        if set1 == set2:
            return True
        if set1.issubset(set2) or set1.issuperset(set2):
            return True

        # Check overlap percentage
        overlap = len(set1.intersection(set2))
        smaller = min(len(set1), len(set2))
        overlap_ratio = overlap / smaller if smaller > 0 else 0

        return overlap_ratio >= 0.5

    def register_event(
        self,
        event_id: str,
        person_id: str,
        video_source: str,
        missing_ppe: List[str],
        frame_number: int,
        timestamp: datetime,
    ) -> None:
        """Register a newly created event as an active violation."""
        key = (person_id, video_source)

        # Separate PPE from actions
        ppe_items = set()
        actions = set()

        for item in missing_ppe:
            if item.startswith("action:"):
                actions.add(item.replace("action:", ""))
            else:
                ppe_items.add(item)

        self.active_violations[key] = ActiveViolation(
            event_id=event_id,
            person_id=person_id,
            missing_ppe=ppe_items,
            actions=actions,
            start_frame=frame_number,
            start_timestamp=timestamp,
            last_frame=frame_number,
            video_source=video_source,
        )

    def get_active_violation(
        self, person_id: str, video_source: str
    ) -> Optional[ActiveViolation]:
        """Get the active violation for a person if any."""
        return self.active_violations.get((person_id, video_source))

    def get_violation_duration(
        self, person_id: str, video_source: str, current_frame: int
    ) -> int:
        """Get the duration in frames for an active violation."""
        active = self.get_active_violation(person_id, video_source)
        if active:
            return current_frame - active.start_frame + 1
        return 0

    def finalize_video(self, video_source: str) -> List[Tuple[str, int, Dict[str, List[str]]]]:
        """
        Finalize all active violations for a video (when processing ends).

        Returns list of (event_id, last_frame, final_violations) for events that need to be closed.
        """
        to_close = []
        keys_to_remove = []

        for key, violation in self.active_violations.items():
            if violation.video_source == video_source:
                final_violations = {
                    "ppe": list(violation.missing_ppe),
                    "actions": list(violation.actions)
                }
                to_close.append(
                    (
                        violation.event_id,
                        violation.last_frame,
                        final_violations,
                    )
                )
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.active_violations[key]

        return to_close

    def clear(self) -> None:
        """Clear all tracked violations."""
        self.active_violations.clear()

    def get_stats(self) -> Dict:
        """Get statistics about active violations."""
        return {
            "active_violations": len(self.active_violations),
            "by_video": {
                video: len(
                    [
                        v
                        for v in self.active_violations.values()
                        if v.video_source == video
                    ]
                )
                for video in set(
                    v.video_source for v in self.active_violations.values()
                )
            },
        }


# Singleton instance
_deduplication_manager: Optional[DeduplicationManager] = None


def get_deduplication_manager() -> DeduplicationManager:
    """Get the singleton deduplication manager instance."""
    global _deduplication_manager
    if _deduplication_manager is None:
        _deduplication_manager = DeduplicationManager()
    return _deduplication_manager
