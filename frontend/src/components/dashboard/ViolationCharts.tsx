"use client";

import { useEffect, useState } from 'react';
import { fetchViolationsByPpe, PpeViolation } from '@/lib/api';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import styles from './ViolationCharts.module.css';

type TooltipPayload = {
  value: number;
};

type CustomTooltipProps = {
  active?: boolean;
  payload?: TooltipPayload[];
  label?: string;
};

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

  const CustomTooltip = ({ active, payload, label }: CustomTooltipProps) => {
    if (active && payload && payload.length) {
      return (
        <div className={styles.tooltip}>
          <p className={styles.tooltipTitle}>{label}</p>
          <p className={styles.tooltipValue}>{payload[0].value} missing</p>
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
          <div className={styles.emptyState}>
            No violation data available
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={data} layout="vertical" margin={{ top: 0, right: 0, left: 30, bottom: 0 }}>
              <XAxis type="number" hide />
              <YAxis 
                dataKey="ppe_type" 
                type="category" 
                axisLine={false} 
                tickLine={false}
                tick={{ fill: 'var(--text-secondary)', fontSize: 12, fontWeight: 700 }}
              />
              <Tooltip cursor={{ fill: 'rgba(219, 39, 119, 0.06)' }} content={<CustomTooltip />} />
              <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                {data.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill="var(--accent-pink)" fillOpacity={0.82} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
