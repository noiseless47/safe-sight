export const API_BASE_URL = 'http://localhost:8000/api';
export const WS_BASE_URL = 'ws://localhost:8000/api/ws';
export const VIDEO_STREAM_URL = 'http://localhost:8000/api/stream/live/feed';

export interface SummaryStats {
  total_events: number;
  today_events: number;
  total_violations: number;
  today_violations: number;
  total_persons: number;
  compliance_rate: number;
  last_updated: string;
}

export interface TimelineData {
  date: string;
  violations: number;
}

export interface PpeViolation {
  ppe_type: string;
  count: number;
}

export interface ComplianceEvent {
  id: string;
  person_id?: string;
  timestamp: string;
  video_source?: string;
  frame_number: number;
  detected_ppe: string[];
  missing_ppe: string[];
  action_violations: string[];
  is_violation: boolean;
}

export interface VideoAsset {
  filename: string;
  display_name: string;
  relative_path: string;
  path: string;
  source: string;
  size_mb: number;
}

export interface ProcessingJob {
  id: string;
  filename: string;
  video_path: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  total_frames: number;
  processed_frames: number;
  events: ComplianceEvent[];
  violations: ComplianceEvent[];
  error?: string | null;
  output_video?: string | null;
}

export async function fetchSummaryStats(): Promise<SummaryStats> {
  const res = await fetch(`${API_BASE_URL}/stats/summary`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch summary stats');
  return res.json();
}

export async function fetchTimeline(days: number = 7): Promise<TimelineData[]> {
  const res = await fetch(`${API_BASE_URL}/stats/timeline?days=${days}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch timeline');
  return res.json();
}

export async function fetchViolationsByPpe(): Promise<PpeViolation[]> {
  const res = await fetch(`${API_BASE_URL}/stats/by-ppe`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch PPE violations');
  return res.json();
}

export async function fetchRecentEvents(limit: number = 20): Promise<{events: ComplianceEvent[]}> {
  const res = await fetch(`${API_BASE_URL}/events?page=1&page_size=${limit}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch recent events');
  return res.json();
}

export async function fetchVideos(): Promise<{videos: VideoAsset[]}> {
  const res = await fetch(`${API_BASE_URL}/stream/videos`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch videos');
  return res.json();
}

export async function startVideoProcessing(videoPath: string): Promise<{job_id: string; video_path: string; reused?: boolean; message?: string}> {
  const res = await fetch(`${API_BASE_URL}/stream/process`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video_path: videoPath }),
  });
  if (!res.ok) throw new Error('Failed to start video processing');
  return res.json();
}

export async function fetchProcessingJob(jobId: string): Promise<ProcessingJob> {
  const res = await fetch(`${API_BASE_URL}/stream/jobs/${jobId}`, { cache: 'no-store' });
  if (!res.ok) throw new Error('Failed to fetch processing job');
  return res.json();
}

export function rawVideoUrl(videoPath: string): string {
  return `${API_BASE_URL}/stream/videos/raw?video_path=${encodeURIComponent(videoPath)}`;
}

export function processedVideoStreamUrl(jobId: string): string {
  return `${API_BASE_URL}/stream/processed/${jobId}/stream`;
}
