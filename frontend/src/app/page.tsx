"use client";

import { useWebSocket } from '@/hooks/useWebSocket';
import Sidebar from '@/components/layout/Sidebar';
import SummaryStats from '@/components/dashboard/SummaryStats';
import LiveVideo from '@/components/dashboard/LiveVideo';
import EventFeed from '@/components/dashboard/EventFeed';
import ViolationCharts from '@/components/dashboard/ViolationCharts';
import styles from './page.module.css';

export default function Dashboard() {
  const { isConnected, lastEvent } = useWebSocket();

  // Create a trigger number that increments on new event, to optionally pass to SummaryStats
  // so it knows when to refetch.
  const updateTrigger = lastEvent ? new Date(lastEvent.timestamp).getTime() : 0;

  return (
    <div className={styles.layout}>
      <Sidebar isConnected={isConnected} />
      
      <main className={styles.main}>
        <header className={styles.header}>
          <div>
            <h1 className={styles.title}>Live Monitoring Center</h1>
            <p className={styles.subtitle}>Real-time active construction zone PPE tracking</p>
          </div>
          
          <div className="glass-panel" style={{ padding: '8px 16px', borderRadius: '20px', display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div style={{ width: 8, height: 8, borderRadius: '50%', background: 'var(--status-safe)' }}></div>
            <span style={{ fontSize: '0.875rem', fontWeight: 600 }}>Cam 01 - Main Entrance</span>
          </div>
        </header>

        <div className={styles.topRow}>
          <SummaryStats updateTrigger={updateTrigger} />
        </div>

        <div className={styles.mainGrid}>
          <div className={styles.leftColumn}>
            <div className={styles.videoWrapper}>
              <LiveVideo />
            </div>
            <div className={styles.chartsWrapper}>
              <ViolationCharts />
            </div>
          </div>
          
          <div className={styles.rightColumn}>
            <EventFeed newEvent={lastEvent} />
          </div>
        </div>
      </main>
    </div>
  );
}
