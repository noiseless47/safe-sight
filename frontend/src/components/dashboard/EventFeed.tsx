"use client";

import { useCallback, useEffect, useMemo, useState } from 'react';
import type { CSSProperties } from 'react';
import { ComplianceEvent, fetchRecentEvents } from '@/lib/api';
import styles from './EventFeed.module.css';

interface EventFeedProps {
  isConnected: boolean;
  newEvent: ComplianceEvent | null;
  panelHeight?: number | null;
}

type FeedFilter = 'attention' | 'open' | 'all';
type PriorityTone = 'critical' | 'high' | 'medium' | 'safe';

interface PrioritizedEvent extends ComplianceEvent {
  confidenceScore: number | null;
  priorityLabel: string;
  priorityScore: number;
  priorityTone: PriorityTone;
  sourceLabel: string;
  workerLabel: string;
  actionText: string;
}

const PPE_LABELS: Record<string, string> = {
  helmet: 'Helmet',
  hardhat: 'Hard Hat',
  hard_hat: 'Hard Hat',
  vest: 'Vest',
  gloves: 'Gloves',
  glove: 'Gloves',
  goggles: 'Goggles',
  safety_goggles: 'Safety Goggles',
  mask: 'Mask',
  boots: 'Boots',
  shoe: 'Safety Shoes',
  shoes: 'Safety Shoes',
  person: 'Person',
};

function mergeEvents(currentEvents: ComplianceEvent[], incomingEvents: ComplianceEvent[]) {
  const byId = new Map<string, ComplianceEvent>();

  [...currentEvents, ...incomingEvents].forEach((event) => {
    byId.set(event.id, { ...byId.get(event.id), ...event });
  });

  return [...byId.values()]
    .sort((first, second) => new Date(second.timestamp).getTime() - new Date(first.timestamp).getTime())
    .slice(0, 60);
}

function humanizeToken(value: string) {
  const normalized = value.replace(/^action:/, '').replace(/[_-]+/g, ' ').trim().toLowerCase();
  if (!normalized) return 'Unknown';

  return (
    PPE_LABELS[normalized] ??
    normalized
      .split(' ')
      .filter(Boolean)
      .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
      .join(' ')
  );
}

function sourceLabel(source?: string) {
  if (!source) return 'Unknown source';

  const filename = source.split(/[\\/]/).pop() ?? source;
  return filename
    .replace(/\.[^.]+$/, '')
    .replace(/^\d+[_\-\s]*/, '')
    .split(/[_\-\s]+/)
    .filter(Boolean)
    .map((token) => {
      const upperToken = token.toUpperCase();
      if (upperToken === 'PPE' || upperToken === 'CCTV') return upperToken;
      return token.charAt(0).toUpperCase() + token.slice(1);
    })
    .join(' ');
}

function workerLabel(event: ComplianceEvent) {
  if (typeof event.track_id === 'number') return `Track ${event.track_id}`;
  if (event.person_id) return event.person_id.replace(/^person_/, 'Person ').replace(/^track_/, 'Track ');
  return 'Unassigned worker';
}

function relativeTime(timestamp: string, now: number) {
  const diffSeconds = Math.max(0, Math.floor((now - new Date(timestamp).getTime()) / 1000));
  if (diffSeconds < 5) return 'Now';
  if (diffSeconds < 60) return `${diffSeconds}s ago`;
  const minutes = Math.floor(diffSeconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  return `${Math.floor(hours / 24)}d ago`;
}

function exactTime(timestamp: string) {
  return new Date(timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function collectConfidenceValues(value: unknown, values: number[]) {
  if (typeof value === 'number' && Number.isFinite(value)) {
    values.push(value > 1 ? value / 100 : value);
    return;
  }

  if (Array.isArray(value)) {
    value.forEach((item) => collectConfidenceValues(item, values));
    return;
  }

  if (value && typeof value === 'object') {
    Object.values(value).forEach((item) => collectConfidenceValues(item, values));
  }
}

function confidenceScore(event: ComplianceEvent) {
  const values: number[] = [];
  collectConfidenceValues(event.detection_confidence, values);
  if (values.length === 0) return null;

  const average = values.reduce((sum, value) => sum + value, 0) / values.length;
  return Math.round(Math.max(0, Math.min(1, average)) * 100);
}

function priorityFor(event: ComplianceEvent): Pick<
  PrioritizedEvent,
  'priorityLabel' | 'priorityScore' | 'priorityTone'
> {
  if (!event.is_violation) {
    return { priorityLabel: 'Clear', priorityScore: 0, priorityTone: 'safe' };
  }

  const missing = event.missing_ppe.map((item) => item.toLowerCase());
  const actionCount = event.action_violations.length;
  const protectionGaps = missing.filter((item) =>
    ['helmet', 'hardhat', 'hard_hat', 'mask', 'respirator', 'goggles', 'safety_goggles'].includes(item),
  ).length;

  const score = Math.min(
    100,
    35 +
      missing.length * 15 +
      protectionGaps * 15 +
      actionCount * 30 +
      (event.is_ongoing === false ? 0 : 8),
  );

  if (score >= 85 || actionCount > 0) {
    return { priorityLabel: 'Critical', priorityScore: score, priorityTone: 'critical' };
  }
  if (score >= 65) return { priorityLabel: 'High', priorityScore: score, priorityTone: 'high' };
  return { priorityLabel: 'Medium', priorityScore: score, priorityTone: 'medium' };
}

function actionText(event: ComplianceEvent) {
  const missing = event.missing_ppe.map((item) => item.toLowerCase());

  if (!event.is_violation) return 'No intervention needed';
  if (event.action_violations.length > 0) return 'Stop task and address unsafe behavior';
  if (missing.some((item) => ['helmet', 'hardhat', 'hard_hat'].includes(item))) {
    return 'Stop work until head protection is corrected';
  }
  if (missing.length >= 2) {
    return `Full PPE check: ${event.missing_ppe.map(humanizeToken).join(', ')}`;
  }
  if (missing.length === 1) return `Correct missing ${humanizeToken(event.missing_ppe[0])}`;
  return 'Review worker PPE before continuing';
}

function buildTitle(event: PrioritizedEvent) {
  const issues = [...event.missing_ppe, ...event.action_violations].map(humanizeToken);
  if (!event.is_violation) return `${event.workerLabel} compliant`;
  if (issues.length === 0) return `${event.workerLabel} safety violation`;
  return `${event.workerLabel}: ${issues.join(', ')}`;
}

function enrichEvent(event: ComplianceEvent): PrioritizedEvent {
  const priority = priorityFor(event);
  return {
    ...event,
    ...priority,
    confidenceScore: confidenceScore(event),
    sourceLabel: sourceLabel(event.video_source),
    workerLabel: workerLabel(event),
    actionText: actionText(event),
  };
}

function durationLabel(event: ComplianceEvent) {
  if (event.is_ongoing !== false) return 'Ongoing';
  const duration = event.duration_frames ?? 1;
  return `${duration} frame${duration === 1 ? '' : 's'}`;
}

export default function EventFeed({ isConnected, newEvent, panelHeight }: EventFeedProps) {
  const [events, setEvents] = useState<ComplianceEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [lastSyncAt, setLastSyncAt] = useState<Date | null>(null);
  const [filter, setFilter] = useState<FeedFilter>('attention');
  const [highlightedId, setHighlightedId] = useState('');
  const [now, setNow] = useState(Date.now());

  const refreshEvents = useCallback(async (showLoading = false) => {
    if (showLoading) setLoading(true);

    try {
      const data = await fetchRecentEvents(50);
      setEvents((current) => mergeEvents(current, data.events));
      setError('');
      setLastSyncAt(new Date());
    } catch (eventError) {
      console.error('Failed to load events', eventError);
      setError('Event history unavailable');
    } finally {
      if (showLoading) setLoading(false);
    }
  }, []);

  useEffect(() => {
    refreshEvents(true);
    const refreshInterval = window.setInterval(() => refreshEvents(false), 8000);
    const clockInterval = window.setInterval(() => setNow(Date.now()), 15000);

    const refreshWhenVisible = () => {
      if (document.visibilityState === 'visible') void refreshEvents(false);
    };

    document.addEventListener('visibilitychange', refreshWhenVisible);
    return () => {
      window.clearInterval(refreshInterval);
      window.clearInterval(clockInterval);
      document.removeEventListener('visibilitychange', refreshWhenVisible);
    };
  }, [refreshEvents]);

  useEffect(() => {
    if (!newEvent) return;

    setEvents((current) => mergeEvents(current, [newEvent]));
    setHighlightedId(newEvent.id);
    setLastSyncAt(new Date());

    const highlightTimer = window.setTimeout(() => setHighlightedId(''), 3200);
    return () => window.clearTimeout(highlightTimer);
  }, [newEvent]);

  const enrichedEvents = useMemo(() => events.map(enrichEvent), [events]);

  const attentionEvents = useMemo(
    () => enrichedEvents.filter((event) => event.is_violation),
    [enrichedEvents],
  );
  const openEvents = useMemo(
    () => enrichedEvents.filter((event) => event.is_violation && event.is_ongoing !== false),
    [enrichedEvents],
  );
  const criticalEvents = useMemo(
    () => enrichedEvents.filter((event) => ['critical', 'high'].includes(event.priorityTone)),
    [enrichedEvents],
  );

  const visibleEvents = useMemo(() => {
    if (filter === 'all') return enrichedEvents;
    if (filter === 'open') return openEvents;
    return attentionEvents;
  }, [attentionEvents, enrichedEvents, filter, openEvents]);

  const latestEvent = enrichedEvents[0];
  const lastSyncLabel = lastSyncAt
    ? `Synced ${relativeTime(lastSyncAt.toISOString(), now)}`
    : 'Sync pending';
  const panelStyle = panelHeight ? ({ height: `${panelHeight}px` } satisfies CSSProperties) : undefined;

  return (
    <section className={`glass-panel ${styles.container}`} style={panelStyle} aria-label="Live event stream">
      <div className={styles.header}>
        <div>
          <div className={styles.kicker}>
            <span className={`${styles.liveDot} ${isConnected ? styles.connected : styles.syncing}`} />
            {isConnected ? 'Live' : 'Syncing'}
          </div>
          <h3>Event Stream</h3>
        </div>
        <span className={styles.syncBadge}>{lastSyncLabel}</span>
      </div>

      <div className={styles.summaryGrid}>
        <div className={styles.summaryCard}>
          <span>Open</span>
          <strong>{openEvents.length}</strong>
        </div>
        <div className={styles.summaryCard}>
          <span>Priority</span>
          <strong>{criticalEvents.length}</strong>
        </div>
        <div className={styles.summaryCard}>
          <span>Latest</span>
          <strong>{latestEvent ? relativeTime(latestEvent.timestamp, now) : 'None'}</strong>
        </div>
      </div>

      <div className={styles.filters} aria-label="Event filters">
        <button
          type="button"
          className={filter === 'attention' ? styles.activeFilter : ''}
          onClick={() => setFilter('attention')}
        >
          Needs Attention
          <span>{attentionEvents.length}</span>
        </button>
        <button
          type="button"
          className={filter === 'open' ? styles.activeFilter : ''}
          onClick={() => setFilter('open')}
        >
          Open
          <span>{openEvents.length}</span>
        </button>
        <button
          type="button"
          className={filter === 'all' ? styles.activeFilter : ''}
          onClick={() => setFilter('all')}
        >
          All
          <span>{enrichedEvents.length}</span>
        </button>
      </div>

      <div className={styles.list}>
        {loading ? (
          <div className={styles.emptyState}>Loading event stream...</div>
        ) : error ? (
          <div className={styles.emptyState}>{error}</div>
        ) : visibleEvents.length === 0 ? (
          <div className={styles.emptyState}>
            {filter === 'all' ? 'No events recorded yet' : 'No active safety events'}
          </div>
        ) : (
          visibleEvents.map((event) => (
            <article
              key={event.id}
              className={`${styles.eventCard} ${styles[event.priorityTone]} ${
                highlightedId === event.id ? styles.newEvent : ''
              }`}
            >
              <div className={styles.eventTop}>
                <div className={styles.eventTitleGroup}>
                  <span className={styles.timeAgo}>{relativeTime(event.timestamp, now)}</span>
                  <h4>{buildTitle(event)}</h4>
                </div>
                <div className={`${styles.priorityPill} ${styles[event.priorityTone]}`}>
                  {event.priorityLabel}
                </div>
              </div>

              <div className={styles.actionRow}>
                <span>Action</span>
                <strong>{event.actionText}</strong>
              </div>

              <div className={styles.evidenceGrid}>
                <div>
                  <span>Source</span>
                  <strong>{event.sourceLabel}</strong>
                </div>
                <div>
                  <span>Frame</span>
                  <strong>{event.frame_number || event.start_frame || 0}</strong>
                </div>
                <div>
                  <span>State</span>
                  <strong>{durationLabel(event)}</strong>
                </div>
                <div>
                  <span>Confidence</span>
                  <strong>{event.confidenceScore === null ? 'N/A' : `${event.confidenceScore}%`}</strong>
                </div>
              </div>

              <div className={styles.chipBlock}>
                {event.detected_ppe.length > 0 ? (
                  <div className={styles.detailRow}>
                    <span>Detected</span>
                    <div>
                      {event.detected_ppe.map((ppe) => (
                        <em key={`${event.id}-detected-${ppe}`} className={styles.detectedChip}>
                          {humanizeToken(ppe)}
                        </em>
                      ))}
                    </div>
                  </div>
                ) : null}

                {event.missing_ppe.length > 0 || event.action_violations.length > 0 ? (
                  <div className={styles.detailRow}>
                    <span>Issue</span>
                    <div>
                      {[...event.missing_ppe, ...event.action_violations].map((issue) => (
                        <em key={`${event.id}-issue-${issue}`} className={styles.missingChip}>
                          {humanizeToken(issue)}
                        </em>
                      ))}
                    </div>
                  </div>
                ) : null}
              </div>

              <div className={styles.eventFooter}>
                <span>{exactTime(event.timestamp)}</span>
                <span>{event.workerLabel}</span>
              </div>
            </article>
          ))
        )}
      </div>
    </section>
  );
}
