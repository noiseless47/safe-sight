"use client";

import { useEffect, useState } from 'react';
import { fetchSummaryStats, SummaryStats as StatsType } from '@/lib/api';
import styles from './SummaryStats.module.css';

export default function SummaryStats({ updateTrigger }: { updateTrigger: number }) {
  const [stats, setStats] = useState<StatsType | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const loadStats = async () => {
      try {
        const data = await fetchSummaryStats();
        setStats(data);
      } catch (e) {
        console.error('Failed to load stats', e);
      } finally {
        setLoading(false);
      }
    };
    loadStats();
    
    // Poll every 10 seconds as a fallback, or rely on updateTrigger from WS
    const interval = setInterval(loadStats, 10000);
    return () => clearInterval(interval);
  }, [updateTrigger]);

  if (loading || !stats) {
    return <div className={styles.grid}>
      {[1, 2, 3, 4].map(i => (
        <div key={i} className={`glass-panel ${styles.card}`} style={{ opacity: 0.5, minHeight: '120px' }} />
      ))}
    </div>;
  }

  const getComplianceClass = (rate: number) => {
    if (rate >= 95) return styles.safe;
    if (rate >= 80) return styles.warning;
    return styles.danger;
  };

  return (
    <div className={styles.grid}>
      <div className={`glass-panel ${styles.card} ${getComplianceClass(stats.compliance_rate)}`}>
        <div className={styles.title}>Compliance Rate</div>
        <div className={styles.value}>{stats.compliance_rate}%</div>
        <div className={styles.subtext}>
          Current average across {stats.total_persons} people
        </div>
      </div>

      <div className={`glass-panel ${styles.card}`}>
        <div className={styles.title}>Active Detections</div>
        <div className={styles.value}>{stats.today_events}</div>
        <div className={styles.subtext}>
          Total events recorded today
        </div>
      </div>

      <div className={`glass-panel ${styles.card} ${stats.today_violations > 0 ? styles.danger : styles.safe}`}>
        <div className={styles.title}>Violations Today</div>
        <div className={styles.value}>{stats.today_violations}</div>
        <div className={styles.subtext}>
          <span className={`${styles.trend} ${stats.today_violations > 0 ? styles.down : styles.up}`}>
            {stats.today_violations > 0 ? 'Requires attention' : 'All clear'}
          </span>
        </div>
      </div>

      <div className={`glass-panel ${styles.card}`}>
        <div className={styles.title}>Historical Violations</div>
        <div className={styles.value}>{stats.total_violations}</div>
        <div className={styles.subtext}>
          Total recorded violations to date
        </div>
      </div>
    </div>
  );
}
