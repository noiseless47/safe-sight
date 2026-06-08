"use client";

import { useEffect, useState } from 'react';
import { fetchViolationsByPpe, PpeViolation } from '@/lib/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import styles from './ViolationCharts.module.css';

export default function ViolationCharts() {
  const [data, setData] = useState<PpeViolation[]>([]);

  useEffect(() => {
    const loadData = async () => {
      try {
        const result = await fetchViolationsByPpe();
        setData(result);
      } catch (e) {
        console.error('Failed to load chart data', e);
      }
    };
    loadData();
  }, []);

  const CustomTooltip = ({ active, payload, label }: any) => {
    if (active && payload && payload.length) {
      return (
        <div style={{ background: 'var(--bg-panel)', border: '1px solid var(--border-color)', padding: '8px 12px', borderRadius: '8px', color: 'var(--text-primary)' }}>
          <p style={{ margin: 0, fontWeight: 600 }}>{label}</p>
          <p style={{ margin: 0, color: 'var(--status-danger)' }}>{payload[0].value} missing</p>
        </div>
      );
    }
    return null;
  };

  return (
    <div className={`glass-panel ${styles.container}`}>
      <div className={styles.header}>
        Missing Equipment Breakdown
      </div>
      <div className={styles.chartContent}>
        {data.length === 0 ? (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%', color: 'var(--text-secondary)' }}>
            No violation data available
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ top: 0, right: 0, left: 30, bottom: 0 }}>
              <XAxis type="number" hide />
              <YAxis 
                dataKey="ppe_type" 
                type="category" 
                axisLine={false} 
                tickLine={false}
                tick={{ fill: 'var(--text-secondary)', fontSize: 12 }}
              />
              <Tooltip cursor={{ fill: 'rgba(255, 255, 255, 0.05)' }} content={<CustomTooltip />} />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill="var(--status-danger)" fillOpacity={0.8} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
