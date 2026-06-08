from fastapi import APIRouter, Depends, Query, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc
from typing import List, Optional
from datetime import datetime, timedelta
from pydantic import BaseModel

from ..deps import get_database
from ...models.event import ComplianceEvent


class EventResponse(BaseModel):
    id: str
    person_id: Optional[str]
    timestamp: datetime
    video_source: Optional[str]
    frame_number: int = 0
    detected_ppe: List[str]
    missing_ppe: List[str]
    action_violations: List[str] = []  # Drinking/Eating violations
    is_violation: bool
    # Event deduplication fields
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None
    end_timestamp: Optional[datetime] = None
    duration_frames: int = 1
    is_ongoing: bool = True

    class Config:
        from_attributes = True


class EventsListResponse(BaseModel):
    events: List[EventResponse]
    total: int
    page: int
    page_size: int


router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=EventsListResponse)
async def get_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    person_id: Optional[str] = None,
    violations_only: bool = False,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    db: AsyncSession = Depends(get_database),
):
    """Get paginated list of compliance events."""
    query = select(ComplianceEvent)

    # Apply filters
    if person_id:
        query = query.where(ComplianceEvent.person_id == person_id)
    if violations_only:
        query = query.where(ComplianceEvent.is_violation == True)
    if start_date:
        query = query.where(ComplianceEvent.timestamp >= start_date)
    if end_date:
        query = query.where(ComplianceEvent.timestamp <= end_date)

    # Get total count
    count_query = select(func.count()).select_from(query.subquery())
    total = await db.scalar(count_query) or 0

    # Apply pagination and ordering
    query = query.order_by(desc(ComplianceEvent.timestamp))
    query = query.offset((page - 1) * page_size).limit(page_size)

    result = await db.execute(query)
    events = result.scalars().all()

    return EventsListResponse(
        events=[EventResponse.model_validate(e) for e in events],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(event_id: str, db: AsyncSession = Depends(get_database)):
    """Get a single event by ID."""
    query = select(ComplianceEvent).where(ComplianceEvent.id == event_id)
    result = await db.execute(query)
    event = result.scalar_one_or_none()

    if not event:
        raise HTTPException(status_code=404, detail="Event not found")

    return EventResponse.model_validate(event)


@router.get("/recent/violations")
async def get_recent_violations(
    limit: int = Query(10, ge=1, le=50), db: AsyncSession = Depends(get_database)
):
    """Get recent violation events."""
    query = (
        select(ComplianceEvent)
        .where(ComplianceEvent.is_violation == True)
        .order_by(desc(ComplianceEvent.timestamp))
        .limit(limit)
    )

    result = await db.execute(query)
    events = result.scalars().all()

    return [EventResponse.model_validate(e) for e in events]
