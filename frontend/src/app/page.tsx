"use client";

import { useEffect, useRef, useState } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import Navbar from '@/components/layout/Navbar';
import SummaryStats from '@/components/dashboard/SummaryStats';
import LiveVideo from '@/components/dashboard/LiveVideo';
import EventFeed from '@/components/dashboard/EventFeed';
import styles from './page.module.css';

export default function Dashboard() {
  const { isConnected, lastEvent } = useWebSocket();
  const updateTrigger = lastEvent ? new Date(lastEvent.timestamp).getTime() : 0;
  const videoWrapperRef = useRef<HTMLDivElement | null>(null);
  const [eventPanelHeight, setEventPanelHeight] = useState<number | null>(null);

  useEffect(() => {
    const updatePanelHeight = () => {
      const videoWrapper = videoWrapperRef.current;
      if (!videoWrapper || window.innerWidth <= 1180) {
        setEventPanelHeight(null);
        return;
      }

      const nextHeight = Math.round(videoWrapper.getBoundingClientRect().height);
      if (nextHeight > 0) setEventPanelHeight(nextHeight);
    };

    updatePanelHeight();

    const resizeObserver = new ResizeObserver(updatePanelHeight);
    if (videoWrapperRef.current) resizeObserver.observe(videoWrapperRef.current);

    window.addEventListener('resize', updatePanelHeight);
    return () => {
      resizeObserver.disconnect();
      window.removeEventListener('resize', updatePanelHeight);
    };
  }, []);

  return (
    <div className={styles.layout}>
      <Navbar isConnected={isConnected} />

      <main className={styles.main}>
        <header className={styles.header}>
          <div>
            <div className={styles.eyebrow}>
              <span className={styles.eyebrowDot} />
              AI PPE Monitoring
            </div>
            <h1 className={styles.title}>
              Live Monitoring <span>Center</span>
            </h1>
            <p className={styles.subtitle}>
              Real-time construction zone safety intelligence with live camera analysis,
              validation videos, and event-level PPE compliance tracking.
            </p>
          </div>

          <div className={styles.cameraPill}>
            <span className={styles.cameraDot} />
            Cam 01 - Main Entrance
          </div>
        </header>

        <div className={styles.topRow}>
          <SummaryStats updateTrigger={updateTrigger} />
        </div>

        <div className={styles.mainGrid}>
          <div className={styles.leftColumn}>
            <div className={styles.videoWrapper} ref={videoWrapperRef}>
              <LiveVideo />
            </div>
          </div>

          <div className={styles.rightColumn}>
            <EventFeed
              isConnected={isConnected}
              newEvent={lastEvent}
              panelHeight={eventPanelHeight}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
