"use client";

import { useEffect, useState } from 'react';
import { ComplianceEvent, fetchRecentEvents } from '@/lib/api';
import styles from './EventFeed.module.css';

interface EventFeedProps {
  newEvent: ComplianceEvent | null;
}

export default function EventFeed({ newEvent }: EventFeedProps) {
  const [events, setEvents] = useState<ComplianceEvent[]>([]);
  const [loading, setLoading] = useState(true);

  // Initial load
  useEffect(() => {
    const loadEvents = async () => {
      try {
        const data = await fetchRecentEvents(20);
        setEvents(data.events);
      } catch (e) {
        console.error('Failed to load events', e);
      } finally {
        setLoading(false);
      }
    };
    loadEvents();
  }, []);

  // Handle new incoming websocket events
  useEffect(() => {
    if (newEvent) {
      setEvents(prev => {
        // Prevent duplicates based on ID
        if (prev.some(e => e.id === newEvent.id)) return prev;
        // Add new event at top, keep max 50
        return [newEvent, ...prev].slice(0, 50);
      });
    }
  }, [newEvent]);

  return (
    <div className={`glass-panel ${styles.container}`}>
      <div className={styles.header}>
        <h3>Live Event Stream</h3>
        <span className={styles.badge}>{events.length} Recent</span>
      </div>

      <div className={styles.list}>
        {loading ? (
          <div className={styles.emptyState}>Loading events...</div>
        ) : events.length === 0 ? (
          <div className={styles.emptyState}>
            <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.5 }}>
              <path d="M12 22c5.523 0 10-4.477 10-10S17.523 2 12 2 2 6.477 2 12s4.477 10 10 10z"></path>
              <path d="M12 8v4"></path>
              <path d="M12 16h.01"></path>
            </svg>
            No events recorded yet
          </div>
        ) : (
          events.map(event => (
            <div key={event.id} className={`${styles.eventCard} ${event.is_violation ? styles.violation : ''}`}>
              <div className={styles.eventHeader}>
                <span className={styles.timestamp}>
                  {new Date(event.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
                </span>
                <span className={`${styles.statusTag} ${event.is_violation ? styles.danger : styles.safe}`}>
                  {event.is_violation ? 'Violation' : 'Compliant'}
                </span>
              </div>
              
              <div className={styles.details}>
                {event.detected_ppe && event.detected_ppe.length > 0 && (
                  <div className={styles.detailRow}>
                    <span className={styles.label}>Detected</span>
                    <div className={styles.values}>
                      {event.detected_ppe.map((ppe, i) => (
                        <span key={i} className={styles.ppeItem}>{ppe}</span>
                      ))}
                    </div>
                  </div>
                )}
                
                {event.missing_ppe && event.missing_ppe.length > 0 && (
                  <div className={styles.detailRow}>
                    <span className={styles.label}>Missing</span>
                    <div className={styles.values}>
                      {event.missing_ppe.map((ppe, i) => (
                        <span key={i} className={`${styles.ppeItem} ${styles.missing}`}>{ppe}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
}
