from typing import Optional, Tuple, List
from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.person import Person
from ..ml.face_recognition import FaceRecognizer


class PersonService:
    """Service for creating and updating person records."""

    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_all_embeddings(self) -> List[Tuple[str, bytes]]:
        """Fetch all stored person embeddings."""
        result = await self.session.execute(
            select(Person.id, Person.face_embedding).where(
                Person.face_embedding.isnot(None)
            )
        )
        return result.all()

    async def get_person(self, person_id: str) -> Optional[Person]:
        """Get a person by ID."""
        return await self.session.get(Person, person_id)

    async def get_or_create_person(
        self,
        person_id: str,
        embedding: Optional[object],
        name: Optional[str] = None,
        thumbnail: Optional[bytes] = None,
    ) -> Person:
        """Get or create a person record, updating embedding if provided."""
        person = await self.get_person(person_id)

        embedding_bytes = None
        if embedding is not None:
            embedding_bytes = FaceRecognizer.serialize_embedding(embedding)

        if person is None:
            person = Person(
                id=person_id,
                name=name,
                face_embedding=embedding_bytes,
                thumbnail=thumbnail,
                first_seen=datetime.now(),
                last_seen=datetime.now(),
                total_events=0,
                violation_count=0,
            )
            self.session.add(person)
        else:
            update_data = {"last_seen": datetime.now()}
            if name:
                update_data["name"] = name
            if embedding_bytes is not None:
                update_data["face_embedding"] = embedding_bytes
            if thumbnail is not None:
                update_data["thumbnail"] = thumbnail

            await self.session.execute(
                update(Person).where(Person.id == person_id).values(**update_data)
            )

        return person

    async def increment_event_counts(self, person_id: str, is_violation: bool):
        """Increment total events and violations for a person."""
        person = await self.get_person(person_id)
        if person is None:
            return

        total_events = (person.total_events or 0) + 1
        violation_count = (person.violation_count or 0) + (1 if is_violation else 0)

        await self.session.execute(
            update(Person)
            .where(Person.id == person_id)
            .values(
                total_events=total_events,
                violation_count=violation_count,
                last_seen=datetime.now(),
            )
        )
