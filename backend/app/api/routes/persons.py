from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import List, Optional
from datetime import datetime
from pydantic import BaseModel

from ..deps import get_database
from ...models.person import Person


class PersonResponse(BaseModel):
    id: str
    name: Optional[str]
    first_seen: datetime
    last_seen: datetime
    total_events: float
    violation_count: float
    compliance_rate: float

    class Config:
        from_attributes = True


class PersonUpdate(BaseModel):
    """Request model for updating person details."""

    name: Optional[str] = None


class PersonListResponse(BaseModel):
    persons: List[PersonResponse]
    total: int


router = APIRouter(prefix="/persons", tags=["persons"])


@router.get("", response_model=PersonListResponse)
async def get_persons(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_database),
):
    """Get paginated list of tracked persons."""
    # Get total count
    count_query = select(func.count(Person.id))
    total = await db.scalar(count_query) or 0

    # Get persons
    query = (
        select(Person)
        .order_by(desc(Person.last_seen))
        .offset((page - 1) * page_size)
        .limit(page_size)
    )

    result = await db.execute(query)
    persons = result.scalars().all()

    # Convert to response model
    person_responses = []
    for p in persons:
        person_responses.append(
            PersonResponse(
                id=p.id,
                name=p.name,
                first_seen=p.first_seen,
                last_seen=p.last_seen,
                total_events=p.total_events,
                violation_count=p.violation_count,
                compliance_rate=p.compliance_rate,
            )
        )

    return PersonListResponse(persons=person_responses, total=total)


@router.get("/{person_id}", response_model=PersonResponse)
async def get_person(person_id: str, db: AsyncSession = Depends(get_database)):
    """Get a single person by ID."""
    query = select(Person).where(Person.id == person_id)
    result = await db.execute(query)
    person = result.scalar_one_or_none()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    return PersonResponse(
        id=person.id,
        name=person.name,
        first_seen=person.first_seen,
        last_seen=person.last_seen,
        total_events=person.total_events,
        violation_count=person.violation_count,
        compliance_rate=person.compliance_rate,
    )


@router.get("/top/violators")
async def get_top_violators(
    limit: int = Query(5, ge=1, le=20), db: AsyncSession = Depends(get_database)
):
    """Get persons with most violations."""
    query = (
        select(Person)
        .where(Person.violation_count > 0)
        .order_by(desc(Person.violation_count))
        .limit(limit)
    )

    result = await db.execute(query)
    persons = result.scalars().all()

    return [
        {
            "id": p.id,
            "name": p.name,
            "violation_count": p.violation_count,
            "compliance_rate": p.compliance_rate,
        }
        for p in persons
    ]


@router.patch("/{person_id}", response_model=PersonResponse)
async def update_person(
    person_id: str,
    update: PersonUpdate,
    db: AsyncSession = Depends(get_database),
):
    """
    Update person details (e.g., assign a name to an identified person).

    This is useful for admins to label automatically detected persons
    with their actual names.
    """
    query = select(Person).where(Person.id == person_id)
    result = await db.execute(query)
    person = result.scalar_one_or_none()

    if not person:
        raise HTTPException(status_code=404, detail="Person not found")

    # Update fields if provided
    if update.name is not None:
        person.name = update.name

    await db.commit()
    await db.refresh(person)

    return PersonResponse(
        id=person.id,
        name=person.name,
        first_seen=person.first_seen,
        last_seen=person.last_seen,
        total_events=person.total_events,
        violation_count=person.violation_count,
        compliance_rate=person.compliance_rate,
    )
