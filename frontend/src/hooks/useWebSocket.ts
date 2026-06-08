"use client";

import { useEffect, useState, useRef, useCallback } from 'react';
import { WS_BASE_URL, ComplianceEvent } from '@/lib/api';

export function useWebSocket() {
  const [isConnected, setIsConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<ComplianceEvent | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_BASE_URL);

    ws.onopen = () => {
      console.log('WebSocket Connected');
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        // Assuming backend sends {"type": "event", "data": ComplianceEvent} or just the event
        if (data.type === 'event' || data.is_violation !== undefined) {
          setLastEvent(data.data || data);
        }
      } catch (e) {
        console.error('Failed to parse WS message:', e);
      }
    };

    ws.onclose = () => {
      console.log('WebSocket Disconnected. Reconnecting...');
      setIsConnected(false);
      setTimeout(connect, 3000);
    };

    ws.onerror = (err) => {
      console.error('WebSocket Error:', err);
      ws.close();
    };

    wsRef.current = ws;
  }, []);

  useEffect(() => {
    connect();
    return () => {
      wsRef.current?.close();
    };
  }, [connect]);

  return { isConnected, lastEvent };
}
