'use client';

import { useEffect, useState, useRef } from 'react';
import Link from 'next/link';

interface Log {
  time: string;
  name: string;
  score: string;
}

interface Stats {
  total_today: number;
  unique_today: number;
}

interface FaceResult {
    track_id: number;
    box: [number, number, number, number];
    message: string;
    color: string;
    liveness: number;
}

export default function Kiosk() {
  const [logs, setLogs] = useState<Log[]>([]);
  const [stats, setStats] = useState<Stats>({ total_today: 0, unique_today: 0 });
  const [currentTime, setCurrentTime] = useState<string>('');
  const [faces, setFaces] = useState<FaceResult[]>([]);
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const isProcessing = useRef(false); // Flow control lock

  // Clock
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
    setCurrentTime(new Date().toLocaleTimeString('en-US', { hour12: false }));
    const timer = setInterval(() => {
      setCurrentTime(new Date().toLocaleTimeString('en-US', { hour12: false }));
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // Initial Fetch & Logic ... (Same as before)
  const fetchInitialData = async () => {
    try {
      const [logsRes, statsRes] = await Promise.all([
        fetch('http://localhost:8000/api/attendance'),
        fetch('http://localhost:8000/api/stats')
      ]);
      
      if (logsRes.ok) {
        const data = await logsRes.json();
        setLogs(data);
      }
      
      if (statsRes.ok) {
        const data = await statsRes.json();
        setStats({
            total_today: data.total_today,
            unique_today: data.unique_today
        });
      }
    } catch (e) {
      console.error(e);
    }
  };

  // Camera & WebSocket Logic
  useEffect(() => {
    fetchInitialData();
    
    // Start Camera
    const startCamera = async () => {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                video: { width: 640, height: 480, frameRate: 30 } 
            });
            if (videoRef.current) {
                videoRef.current.srcObject = stream;
            }
        } catch (err) {
            console.error("Camera Error:", err);
        }
    };
    startCamera();

    // Connect WS
    const connect = () => {
        const ws = new WebSocket('ws://localhost:8000/ws');
        wsRef.current = ws;

        ws.onmessage = (event) => {
            try {
                const payload = JSON.parse(event.data);
                
                
                // Unlock processing on any response
                isProcessing.current = false;

                if (payload.type === 'log') {
                    setLogs(prev => [payload.data, ...prev]);
                    setStats(payload.stats);
                    setSuccessMsg({ name: payload.data.name, time: payload.data.time }); // Show Toast
                } else if (payload.type === 'result') {
                    setFaces(payload.faces);
                }
            } catch (e) {
                console.error('WS Parse Error:', e);
            }
        };

        ws.onclose = () => {
            setTimeout(connect, 3000);
        };
    };
    connect();

    // State to handle flow control
    // Moved to top level


    // Frame Loop with Flow Control
    const sendFrame = () => {
        if (!videoRef.current || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
        
        // FLOW CONTROL: Do not send if server is still processing the previous frame
        if (isProcessing.current) return;

        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (!canvas) return;

        const ctx = canvas.getContext('2d');
        if (!ctx) return;

        // Draw video (low res for performance)
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.drawImage(video, 0, 0);

        canvas.toBlob((blob) => {
            if (blob && wsRef.current?.readyState === WebSocket.OPEN) {
                isProcessing.current = true; // Lock
                wsRef.current.send(blob);
            }
        }, 'image/jpeg', 0.7); // Slightly lower quality for speed
    };

    // Check often (e.g. 30FPS), but only send if ready. 
    // This allows the system to go as fast as the server allows, but never clog it.
    const interval = setInterval(sendFrame, 33);

    return () => {
        clearInterval(interval);
        if (wsRef.current) wsRef.current.close();
    };
  }, []);

  // Handle Toast Notification
  const [successMsg, setSuccessMsg] = useState<{name: string, time: string} | null>(null);
  useEffect(() => {
    if (successMsg) {
        const timer = setTimeout(() => setSuccessMsg(null), 3000);
        return () => clearTimeout(timer);
    }
  }, [successMsg]);

  return (
    <main className="hud-container">
      {/* BACKGROUND VIDEO LAYER */}
      <div className="video-layer">
        <video 
            ref={videoRef}
            autoPlay 
            playsInline 
            muted 
            className="hud-video"
        />
        <canvas ref={canvasRef} style={{ display: 'none' }} />
      </div>

      {/* HUD OVERLAYS */}
      <div className="hud-overlay">
        
        {/* TOP BAR: BRAND & STATS */}
        <header className="hud-header">
            <div className="hud-brand" style={{ display: 'flex', alignItems: 'center', gap: '20px', pointerEvents: 'auto' }}>
                <div>
                    <div className="hud-logo">🛡️SECURE VIEW</div>
                    <div className="hud-subtitle">AI SURVEILLANCE SYSTEM</div>
                </div>
                <Link href="/admin" className="hud-mini-admin-btn" title="Open Admin Panel">
                    <span className="icon">⚙️</span>
                </Link>
            </div>
            {/* <div className="hud-stats">
                <div className="stat-item">
                    <span className="stat-label">TOTAL ENTRIES</span>
                    <span className="stat-value">{stats.total_today.toString().padStart(4, '0')}</span>
                </div>
                <div className="stat-separator"></div>
                <div className="stat-item">
                    <span className="stat-label">UNIQUE VISITORS</span>
                    <span className="stat-value highlight">{stats.unique_today.toString().padStart(3, '0')}</span>
                </div>
            </div> */}
            <div className="hud-clock">
                <div className="location">CAM-01 • MAIN GATE</div>
                <div className="time">{mounted ? currentTime : '--:--:--'}</div>
            </div>
        </header>

        {/* CENTER RETICLE / SCANNING EFFECT */}
        <div className="scan-line"></div>
        <div className="vignette"></div>

        {/* BOTTOM STATUS BAR */}
        <footer className="hud-footer">
            <div className="system-status">
                <div className={`status-dot ${wsRef.current?.readyState === 1 ? 'online' : 'offline'}`}></div>
                <span>SYSTEM {wsRef.current?.readyState === 1 ? 'ONLINE' : 'OFFLINE'}</span>
            </div>
            <div className="rec-indicator">
                <span>REC</span>
                <div className="rec-dot"></div>
            </div>
        </footer>

        {/* FACE BOXES LAYER */}
        {faces.map((face) => {
            if (!videoRef.current) return null;
            const vW = videoRef.current.videoWidth || 640;
            const vH = videoRef.current.videoHeight || 480;
            const [x, y, w, h] = face.box;
            
            const style = {
                left: `${((vW - x - w) / vW) * 100}%`,
                top: `${(y / vH) * 100}%`,
                width: `${(w / vW) * 100}%`,
                height: `${(h / vH) * 100}%`,
                borderColor: face.color,
                boxShadow: `0 0 20px ${face.color}40` // Add glow with transparency
            } as React.CSSProperties;

            return (
                <div key={face.track_id} className="hud-face-box" style={style}>
                    <div className="hud-face-label" style={{ backgroundColor: face.color }}>
                        <span className="face-name">{face.message}</span>
                        {face.liveness < 0.5 && <span className="liveness-warn">⚠ FAKE</span>}
                    </div>
                    {/* Decorators for face box */}
                    <div className="corner c-tl"></div>
                    <div className="corner c-tr"></div>
                    <div className="corner c-bl"></div>
                    <div className="corner c-br"></div>
                </div>
            );
        })}

        {/* RECENT ENTRY TOAST (Center Bottom) */}
        {successMsg && (
            <div className="hud-toast">
                <div className="toast-glow"></div>
                <div className="toast-icon-box">
                    <span className="check-icon">✓</span>
                </div>
                <div className="toast-info">
                    <div className="access-granted">ACCESS GRANTED</div>
                    <div className="toast-user">{successMsg.name}</div>
                    <div className="toast-meta">{successMsg.time}</div>
                </div>
            </div>
        )}

      </div>
    </main>
  );
}
