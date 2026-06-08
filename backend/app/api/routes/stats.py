from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from datetime import datetime, timedelta
from typing import List

from ..deps import get_database
from ...models.event import ComplianceEvent
from ...models.person import Person


router = APIRouter(prefix="/stats", tags=["statistics"])


@router.get("/summary")
async def get_summary_stats(db: AsyncSession = Depends(get_database)):
    """
    Get summary statistics for dashboard.

    Compliance Rate Calculation:
    - Measures the percentage of PEOPLE who are fully compliant
    - A person is compliant if they have ZERO violations
    - Formula: (persons_with_zero_violations / total_persons) * 100

    Example:
        - 10 people tracked, 8 have no violations → 80% compliance rate
        - More meaningful than event-based calculation for safety monitoring
    """
    # Total events today
    today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    total_events_query = select(func.count(ComplianceEvent.id))
    total_events = await db.scalar(total_events_query) or 0

    today_events_query = select(func.count(ComplianceEvent.id)).where(
        ComplianceEvent.timestamp >= today_start
    )
    today_events = await db.scalar(today_events_query) or 0

    # Violations
    total_violations_query = select(func.count(ComplianceEvent.id)).where(
        ComplianceEvent.is_violation == True
    )
    total_violations = await db.scalar(total_violations_query) or 0

    today_violations_query = select(func.count(ComplianceEvent.id)).where(
        ComplianceEvent.is_violation == True, ComplianceEvent.timestamp >= today_start
    )
    today_violations = await db.scalar(today_violations_query) or 0

    # Unique persons
    total_persons_query = select(func.count(Person.id))
    total_persons = await db.scalar(total_persons_query) or 0

    # Compliance rate - based on persons, not events
    # A person is compliant if they have zero violations
    compliance_rate = 100.0
    if total_persons > 0:
        # Count persons with at least one violation
        persons_with_violations_query = select(func.count(Person.id)).where(
            Person.violation_count > 0
        )
        persons_with_violations = await db.scalar(persons_with_violations_query) or 0

        # Compliance rate = % of persons who are fully compliant
        compliant_persons = total_persons - persons_with_violations
        compliance_rate = (compliant_persons / total_persons) * 100

    return {
        "total_events": total_events,
        "today_events": today_events,
        "total_violations": total_violations,
        "today_violations": today_violations,
        "total_persons": total_persons,
        "compliance_rate": round(compliance_rate, 1),
        "last_updated": datetime.now().isoformat(),
    }


@router.get("/timeline")
async def get_violation_timeline(
    days: int = 7, db: AsyncSession = Depends(get_database)
):
    """Get violations over time for charting."""
    start_date = datetime.now() - timedelta(days=days)

    # Group violations by day
    query = (
        select(
            func.date(ComplianceEvent.timestamp).label("date"),
            func.count(ComplianceEvent.id).label("count"),
        )
        .where(
            ComplianceEvent.is_violation == True,
            ComplianceEvent.timestamp >= start_date,
        )
        .group_by(func.date(ComplianceEvent.timestamp))
        .order_by(func.date(ComplianceEvent.timestamp))
    )

    result = await db.execute(query)
    rows = result.all()

    return [{"date": str(row.date), "violations": row.count} for row in rows]


@router.get("/by-ppe")
async def get_violations_by_ppe(db: AsyncSession = Depends(get_database)):
    """Get violation breakdown by PPE type."""
    # This would require JSON querying which varies by database
    # For SQLite, we'll do it in Python
    query = select(ComplianceEvent.missing_ppe).where(
        ComplianceEvent.is_violation == True
    )

    result = await db.execute(query)
    events = result.scalars().all()

    # Count missing PPE types
    ppe_counts = {}
    for missing_ppe in events:
        if missing_ppe:
            for ppe in missing_ppe:
                ppe_counts[ppe] = ppe_counts.get(ppe, 0) + 1

    return [
        {"ppe_type": ppe, "count": count}
        for ppe, count in sorted(ppe_counts.items(), key=lambda x: -x[1])
    ]
