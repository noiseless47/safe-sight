"""
Person Re-Identification Gallery

Stores person feature galleries (face embeddings + appearance features) for
re-identifying individuals across track deletions and re-entries.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np
from datetime import datetime


@dataclass
class PersonRecord:
    """Stores features for a single person across multiple detections."""

    person_id: str
    face_embeddings: List[np.ndarray] = field(default_factory=list)
    appearance_features: List[np.ndarray] = field(default_factory=list)
    last_seen_frame: int = 0
    first_seen_frame: int = 0
    total_detections: int = 0
    last_seen_time: datetime = field(default_factory=datetime.now)
    name: Optional[str] = None


class PersonGallery:
    """
    Maintains a gallery of known persons for re-identification.

    Matching priority:
    1. Face embedding match
    2. Appearance feature match
    3. No match: create new person_id
    """

    def __init__(
        self,
        face_threshold: float = 0.6,
        appearance_threshold: float = 0.5,
        max_features_per_person: int = 50,
    ):
        self.face_threshold = face_threshold
        self.appearance_threshold = appearance_threshold
        self.max_features_per_person = max_features_per_person

        self.persons: Dict[str, PersonRecord] = {}
        self.next_person_id = 1

    def _get_next_person_id(self) -> str:
        """Generate next sequential person ID."""
        pid = f"person_{self.next_person_id}"
        self.next_person_id += 1
        return pid

    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

    def _mean_embedding(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Compute mean of embedding list."""
        if not embeddings:
            return np.zeros(512)
        return np.mean(embeddings, axis=0)

    def add_person(
        self,
        person_id: Optional[str] = None,
        face_embedding: Optional[np.ndarray] = None,
        appearance_feature: Optional[np.ndarray] = None,
        frame_number: int = 0,
    ) -> str:
        """
        Add a new person or update existing person.

        Args:
            person_id: Optional specific ID (for loading from DB)
            face_embedding: Face embedding if available
            appearance_feature: Appearance feature if available
            frame_number: Current frame number

        Returns:
            The person_id assigned/used
        """
        if person_id is None:
            person_id = self._get_next_person_id()

        if person_id not in self.persons:
            self.persons[person_id] = PersonRecord(
                person_id=person_id,
                first_seen_frame=frame_number,
            )

        record = self.persons[person_id]

        if face_embedding is not None:
            if len(record.face_embeddings) < self.max_features_per_person:
                record.face_embeddings.append(face_embedding)

        if appearance_feature is not None:
            if len(record.appearance_features) < self.max_features_per_person:
                record.appearance_features.append(appearance_feature)

        record.last_seen_frame = frame_number
        record.total_detections += 1
        record.last_seen_time = datetime.now()

        return person_id

    def update_person(
        self,
        person_id: str,
        face_embedding: Optional[np.ndarray] = None,
        appearance_feature: Optional[np.ndarray] = None,
        frame_number: int = 0,
    ) -> bool:
        """
        Update an existing person's features.

        Args:
            person_id: The person to update
            face_embedding: New face embedding if available
            appearance_feature: New appearance feature if available
            frame_number: Current frame number

        Returns:
            True if person found and updated, False otherwise
        """
        if person_id not in self.persons:
            return False

        record = self.persons[person_id]

        if face_embedding is not None:
            if len(record.face_embeddings) < self.max_features_per_person:
                record.face_embeddings.append(face_embedding)

        if appearance_feature is not None:
            if len(record.appearance_features) < self.max_features_per_person:
                record.appearance_features.append(appearance_feature)

        record.last_seen_frame = frame_number
        record.total_detections += 1
        record.last_seen_time = datetime.now()

        return True

    def find_match(
        self,
        face_embedding: Optional[np.ndarray] = None,
        appearance_feature: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[str], str]:
        """
        Find best matching person in gallery.

        Args:
            face_embedding: Face embedding if available
            appearance_feature: Appearance feature if available

        Returns:
            Tuple of (person_id or None, match_type: "face" | "appearance" | "none")
        """
        best_match_id: Optional[str] = None
        best_score = 0.0
        match_type = "none"

        # Priority 1: Try face matching
        if face_embedding is not None and len(face_embedding) > 0:
            for person_id, record in self.persons.items():
                if record.face_embeddings:
                    mean_face = self._mean_embedding(record.face_embeddings)
                    score = self._cosine_similarity(face_embedding, mean_face)
                    if score > best_score and score >= self.face_threshold:
                        best_match_id = person_id
                        best_score = score
                        match_type = "face"

        # Priority 2: Try appearance matching (if no face match found)
        if best_match_id is None and appearance_feature is not None:
            for person_id, record in self.persons.items():
                if record.appearance_features:
                    mean_app = self._mean_embedding(record.appearance_features)
                    score = self._cosine_similarity(appearance_feature, mean_app)
                    if score > best_score and score >= self.appearance_threshold:
                        best_match_id = person_id
                        best_score = score
                        match_type = "appearance"

        return best_match_id, match_type

    def get_person_count(self) -> int:
        """Get total number of unique persons in gallery."""
        return len(self.persons)

    def get_person(self, person_id: str) -> Optional[PersonRecord]:
        """Get a specific person record."""
        return self.persons.get(person_id)

    def get_all_persons(self) -> Dict[str, PersonRecord]:
        """Get all person records."""
        return self.persons

    def get_stats(self) -> Dict:
        """Get gallery statistics."""
        total_embeddings = sum(
            len(p.face_embeddings) + len(p.appearance_features)
            for p in self.persons.values()
        )
        return {
            "total_persons": len(self.persons),
            "total_embeddings": total_embeddings,
            "next_person_id": self.next_person_id,
        }

    def clear(self):
        """Clear all persons from gallery."""
        self.persons.clear()
        self.next_person_id = 1

    def export_for_db(self) -> Dict[str, Dict]:
        """Export gallery data for database storage."""
        return {
            pid: {
                "face_embeddings": [
                    e.tolist() if isinstance(e, np.ndarray) else list(e)
                    for e in record.face_embeddings
                ],
                "appearance_features": [
                    e.tolist() if isinstance(e, np.ndarray) else list(e)
                    for e in record.appearance_features
                ],
                "first_seen_frame": record.first_seen_frame,
                "last_seen_frame": record.last_seen_frame,
                "total_detections": record.total_detections,
                "name": record.name,
            }
            for pid, record in self.persons.items()
        }
