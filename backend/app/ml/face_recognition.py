"""
Face Recognition using InsightFace

Provides face detection and embedding extraction for person re-identification.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import pickle

from ..core.config import settings


class FaceRecognizer:
    """
    Face detection and recognition using InsightFace.

    Uses RetinaFace for detection and ArcFace for embeddings.
    """

    def __init__(self, threshold: float = 0.6):
        self.app = None
        self.threshold = threshold
        self._initialized = False

    def initialize(self):
        """Lazy initialization of InsightFace."""
        if self._initialized:
            return

        try:
            from insightface.app import FaceAnalysis

            print("Loading InsightFace model...")
            self.app = FaceAnalysis(
                name="buffalo_l",
                providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
            )
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            self._initialized = True
            print("InsightFace loaded successfully!")
        except Exception as e:
            print(f"InsightFace not available: {e}")
            self._initialized = True

    def detect_faces(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect faces in frame and extract embeddings.

        Args:
            frame: BGR numpy array from OpenCV

        Returns:
            List of face detections with boxes and embeddings
        """
        if not self._initialized:
            self.initialize()

        if self.app is None:
            return []

        try:
            faces = self.app.get(frame)

            results = []
            for face in faces:
                results.append(
                    {
                        "box": face.bbox.tolist(),
                        "embedding": face.embedding,
                        "score": float(face.det_score),
                        "landmarks": face.landmark_2d_106.tolist()
                        if face.landmark_2d_106 is not None
                        else None,
                    }
                )

            return results
        except Exception as e:
            print(f"Face detection error: {e}")
            return []

    def _mock_detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Return no faces when InsightFace is unavailable."""
        return []

    def compare_embeddings(
        self, embedding1: np.ndarray, embedding2: np.ndarray
    ) -> float:
        """
        Compare two face embeddings using cosine similarity.

        Returns:
            Similarity score between 0 and 1
        """
        if embedding1 is None or embedding2 is None:
            return 0.0

        # Normalize embeddings
        e1 = embedding1 / np.linalg.norm(embedding1)
        e2 = embedding2 / np.linalg.norm(embedding2)

        # Cosine similarity
        similarity = np.dot(e1, e2)

        return float(similarity)

    def find_matching_person(
        self, embedding: np.ndarray, known_embeddings: List[Tuple[str, np.ndarray]]
    ) -> Optional[Tuple[str, float]]:
        """
        Find the best matching person from known embeddings.

        Args:
            embedding: Face embedding to match
            known_embeddings: List of (person_id, embedding) tuples

        Returns:
            Tuple of (person_id, similarity) if match found, else None
        """
        if embedding is None or not known_embeddings:
            return None

        best_match = None
        best_score = 0.0

        for person_id, known_embedding in known_embeddings:
            similarity = self.compare_embeddings(embedding, known_embedding)

            if similarity > best_score and similarity >= self.threshold:
                best_score = similarity
                best_match = person_id

        if best_match:
            return (best_match, best_score)

        return None

    @staticmethod
    def serialize_embedding(embedding: np.ndarray) -> bytes:
        """Serialize embedding for database storage."""
        return pickle.dumps(embedding)

    @staticmethod
    def deserialize_embedding(data: bytes) -> np.ndarray:
        """Deserialize embedding from database."""
        return pickle.loads(data)


# Singleton instance
_recognizer = None


def get_face_recognizer() -> FaceRecognizer:
    global _recognizer
    if _recognizer is None:
        _recognizer = FaceRecognizer()
    return _recognizer
