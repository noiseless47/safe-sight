"use client";

import { useEffect, useMemo, useState } from 'react';
import {
  fetchProcessingJob,
  fetchVideos,
  processedVideoStreamUrl,
  rawVideoUrl,
  startVideoProcessing,
  VIDEO_STREAM_URL,
  ProcessingJob,
  VideoAsset,
} from '@/lib/api';
import styles from './LiveVideo.module.css';

export default function LiveVideo() {
  const [mode, setMode] = useState<'live' | 'validation'>('validation');
  const [hasError, setHasError] = useState(false);
  const [videos, setVideos] = useState<VideoAsset[]>([]);
  const [selectedPath, setSelectedPath] = useState('');
  const [job, setJob] = useState<ProcessingJob | null>(null);
  const [loadingVideos, setLoadingVideos] = useState(true);
  const [actionError, setActionError] = useState('');

  useEffect(() => {
    let active = true;

    async function loadVideos() {
      try {
        const data = await fetchVideos();
        if (!active) return;

        setVideos(data.videos);
        const preferred =
          data.videos.find((video) => video.display_name.toLowerCase().includes('ppe_video')) ??
          data.videos.find((video) => video.display_name.toLowerCase().includes('ppe_red_zone')) ??
          data.videos[0];
        if (preferred) setSelectedPath(preferred.path);
      } catch (error) {
        console.error(error);
        if (active) setActionError('Video dataset unavailable');
      } finally {
        if (active) setLoadingVideos(false);
      }
    }

    loadVideos();
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (!job || job.status === 'completed' || job.status === 'failed') return;

    const interval = window.setInterval(async () => {
      try {
        const nextJob = await fetchProcessingJob(job.id);
        setJob(nextJob);
      } catch (error) {
        console.error(error);
        setJob(null);
        setActionError('Previous analysis session expired');
      }
    }, 1500);

    return () => window.clearInterval(interval);
  }, [job]);

  const selectedVideo = useMemo(
    () => videos.find((video) => video.path === selectedPath) ?? null,
    [selectedPath, videos],
  );

  const isProcessing = job?.status === 'pending' || job?.status === 'processing';

  const handleProcess = async () => {
    if (!selectedVideo || isProcessing) return;

    setActionError('');
    setJob(null);

    try {
      const started = await startVideoProcessing(selectedVideo.path);
      const nextJob = await fetchProcessingJob(started.job_id);
      setJob(nextJob);
    } catch (error) {
      console.error(error);
      setActionError('Could not start analysis');
    }
  };

  const videoSrc =
    mode === 'live'
      ? VIDEO_STREAM_URL
      : job?.status === 'completed'
        ? processedVideoStreamUrl(job.id)
        : selectedVideo
          ? rawVideoUrl(selectedVideo.path)
          : '';

  return (
    <div className={`glass-panel ${styles.container}`}>
      <div className={styles.controls}>
        <div className={styles.segmented}>
          <button
            type="button"
            className={mode === 'validation' ? styles.activeSegment : ''}
            onClick={() => {
              setMode('validation');
              setHasError(false);
            }}
          >
            Dataset
          </button>
          <button
            type="button"
            className={mode === 'live' ? styles.activeSegment : ''}
            onClick={() => {
              setMode('live');
              setHasError(false);
            }}
          >
            Live
          </button>
        </div>

        {mode === 'validation' ? (
          <div className={styles.datasetControls}>
            <select
              value={selectedPath}
              disabled={loadingVideos || videos.length === 0 || isProcessing}
              onChange={(event) => {
                setSelectedPath(event.target.value);
                setJob(null);
                setActionError('');
                setHasError(false);
              }}
            >
              {videos.map((video) => (
                <option key={video.path} value={video.path}>
                  {video.display_name}
                </option>
              ))}
            </select>
            <button type="button" onClick={handleProcess} disabled={!selectedVideo || isProcessing}>
              {isProcessing ? 'Analyzing' : 'Run PPE Analysis'}
            </button>
          </div>
        ) : null}
      </div>

      {!hasError && videoSrc ? (
        mode === 'validation' && job?.status !== 'completed' ? (
          <video
            key={videoSrc}
            src={videoSrc}
            className={styles.videoFeed}
            controls
            muted
            playsInline
            onError={() => setHasError(true)}
          />
        ) : (
          <>
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              key={videoSrc}
              src={videoSrc}
              alt={mode === 'live' ? 'Live camera feed' : 'Processed validation feed'}
              className={styles.videoFeed}
              onError={() => setHasError(true)}
            />
          </>
        )
      ) : (
        <div className={styles.offlineMessage}>
          <svg className={styles.offlineIcon} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M2 2l20 20"></path><path d="M15 15a4 4 0 01-5.8-5.8"></path><path d="M12 12v.01"></path><path d="M10 5.4a4 4 0 015.6 5.6"></path><path d="M8 8a2 2 0 00-2 2v4a2 2 0 002 2h8a2 2 0 002-2v-4a2 2 0 00-2-2"></path>
          </svg>
          <p>{mode === 'live' ? 'Camera feed offline' : 'Video unavailable'}</p>
        </div>
      )}

      <div className={styles.overlay}>
        {mode === 'live' ? <div className={styles.recDot}></div> : null}
        {mode === 'live' ? 'LIVE' : job?.status === 'completed' ? 'PROCESSED' : 'DATASET'}
      </div>

      {mode === 'validation' ? (
        <div className={styles.statusBar}>
          <span>{selectedVideo ? `${selectedVideo.size_mb} MB` : `${videos.length} videos`}</span>
          {job ? <span>{job.status} {Math.round(job.progress || 0)}%</span> : null}
          {actionError ? <span className={styles.errorText}>{actionError}</span> : null}
        </div>
      ) : null}
    </div>
  );
}
