"use client";

import { useEffect, useState, useRef, useCallback } from 'react';
import { WS_BASE_URL, ComplianceEvent } from '@/lib/api';

function normalizeEventPayload(payload: unknown): ComplianceEvent | null {
  if (!payload || typeof payload !== 'object') return null;

  const message = payload as Record<string, unknown>;
  const candidate =
    message.data && typeof message.data === 'object'
      ? (message.data as Record<string, unknown>)
      : message.event && typeof message.event === 'object'
        ? (message.event as Record<string, unknown>)
        : message;

  const eventId = candidate.id ?? candidate.event_id ?? message.event_id;
  const timestamp = candidate.timestamp ?? message.timestamp;
  const isViolation = candidate.is_violation ?? message.type === 'violation';

  if (!eventId || !timestamp || typeof isViolation !== 'boolean') return null;

  return {
    id: String(eventId),
    person_id: candidate.person_id ? String(candidate.person_id) : undefined,
    track_id: typeof candidate.track_id === 'number' ? candidate.track_id : undefined,
    timestamp: String(timestamp),
    video_source: candidate.video_source ? String(candidate.video_source) : undefined,
    frame_number: typeof candidate.frame_number === 'number' ? candidate.frame_number : 0,
    detected_ppe: Array.isArray(candidate.detected_ppe) ? candidate.detected_ppe.map(String) : [],
    missing_ppe: Array.isArray(candidate.missing_ppe) ? candidate.missing_ppe.map(String) : [],
    action_violations: Array.isArray(candidate.action_violations)
      ? candidate.action_violations.map(String)
      : [],
    is_violation: isViolation,
    detection_confidence:
      candidate.detection_confidence && typeof candidate.detection_confidence === 'object'
        ? (candidate.detection_confidence as ComplianceEvent['detection_confidence'])
        : {},
    snapshot_path: candidate.snapshot_path ? String(candidate.snapshot_path) : undefined,
    start_frame: typeof candidate.start_frame === 'number' ? candidate.start_frame : undefined,
    end_frame: typeof candidate.end_frame === 'number' ? candidate.end_frame : undefined,
    end_timestamp: candidate.end_timestamp ? String(candidate.end_timestamp) : undefined,
    duration_frames:
      typeof candidate.duration_frames === 'number' ? candidate.duration_frames : undefined,
    is_ongoing: typeof candidate.is_ongoing === 'boolean' ? candidate.is_ongoing : undefined,
  };
}

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<ComplianceEvent | null>(null);
  const [lastMessageAt, setLastMessageAt] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<number | null>(null);
  const heartbeatTimerRef = useRef<number | null>(null);
  const shouldReconnectRef = useRef(true);

  const connect = useCallback(() => {
    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    ) {
      return;
    }

    const ws = new WebSocket(WS_BASE_URL);

    ws.onopen = () => {
      setIsConnected(true);
      setLastMessageAt(new Date().toISOString());
      heartbeatTimerRef.current = window.setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send('ping');
        }
      }, 25000);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const nextEvent = normalizeEventPayload(data);
        setLastMessageAt(new Date().toISOString());
        if (nextEvent) setLastEvent(nextEvent);
      } catch (e) {
        console.error('Failed to parse WS message:', e);
      }
    };

    ws.onclose = () => {
      setIsConnected(false);
      if (heartbeatTimerRef.current) {
        window.clearInterval(heartbeatTimerRef.current);
        heartbeatTimerRef.current = null;
      }
      wsRef.current = null;

      if (shouldReconnectRef.current) {
        reconnectTimerRef.current = window.setTimeout(connect, 3000);
      }
    };

    ws.onerror = (err) => {
      console.error('WebSocket Error:', err);
      ws.close();
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    shouldReconnectRef.current = true;
    connect();
    return () => {
      shouldReconnectRef.current = false;
      if (reconnectTimerRef.current) {
        window.clearTimeout(reconnectTimerRef.current);
        reconnectTimerRef.current = null;
      }
      if (heartbeatTimerRef.current) {
        window.clearInterval(heartbeatTimerRef.current);
        heartbeatTimerRef.current = null;
      }
      wsRef.current?.close();
    };
  }, [connect]);

  return { isConnected, lastEvent, lastMessageAt };
}
