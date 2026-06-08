"use client";

import styles from './Navbar.module.css';

function DashboardIcon() {
  return (
    <svg viewBox="0 0 24 24" aria-hidden="true">
      <rect x="3" y="3" width="7" height="7" rx="1.5" />
      <rect x="14" y="3" width="7" height="7" rx="1.5" />
      <rect x="3" y="14" width="7" height="7" rx="1.5" />
      <rect x="14" y="14" width="7" height="7" rx="1.5" />
    </svg>
  );
}

function scrollToDashboard() {
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

export default function Navbar({ isConnected }: { isConnected: boolean }) {
  return (
    <header className={styles.header}>
      <div className={styles.inner}>
        <button
          type="button"
          className={styles.brand}
          aria-label="SafeSight dashboard"
          onClick={scrollToDashboard}
        >
          <div className={styles.logoIcon}>S</div>
          <span className={styles.logoText}>Safe<span>Sight</span></span>
        </button>

        <nav className={styles.nav} aria-label="Primary navigation">
          <button
            type="button"
            className={`${styles.navItem} ${styles.active}`}
            aria-current="page"
            onClick={scrollToDashboard}
          >
            <DashboardIcon />
            <span>Dashboard</span>
          </button>
        </nav>

        <div className={`${styles.statusPill} ${!isConnected ? styles.disconnected : ''}`}>
          <span className={styles.statusDot} />
          <span>{isConnected ? 'System Online' : 'Disconnected'}</span>
        </div>
      </div>
    </header>
  );
}
