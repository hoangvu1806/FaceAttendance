

'use client';

import { useState, useEffect, useRef } from 'react';
import Link from 'next/link';

// Helper for safe fetching
const safeFetch = async (url: string, options?: RequestInit) => {
    try {
        const res = await fetch(url, options);
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const text = await res.text();
        return text ? JSON.parse(text) : null;
    } catch (e) {
        console.error(`Fetch Error (${url}):`, e);
        return null;
    }
};

export default function AdminPage() {
  const [token, setToken] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    const data = await safeFetch('http://localhost:8000/api/login', {
        method: 'POST', body: JSON.stringify({ password }), headers: {'Content-Type': 'application/json'}
    });
    
    if (data && data.token) {
        setToken(data.token);
        setError('');
    } else {
        setError('Invalid Password or Server Error');
    }
  };

  if (!token) return (
        <div className="hud-login-container">
            <div className="hud-login-box">
                <h1 className="hud-login-title">COMMAND CENTER</h1>
                <p className="hud-login-subtitle">System Access Control</p>
                <form onSubmit={handleLogin}>
                    <input 
                        type="password" 
                        placeholder="ENTER ACCESS CODE" 
                        value={password} 
                        onChange={e=>setPassword(e.target.value)} 
                        className="hud-input" 
                        style={{textAlign:'center', fontSize:'1.rem', letterSpacing:'8px', marginBottom:'20px'}}
                    />
                    
                    {/* TOUCH KEYPAD */}
                    <div className="numpad-grid" style={{display:'grid', gridTemplateColumns:'repeat(3, 1fr)', gap:'10px', marginBottom:'20px'}}>
                        {[1,2,3,4,5,6,7,8,9].map(num => (
                            <button key={num} type="button" onClick={()=>setPassword(p=>p+num)} className="hud-btn" style={{fontSize:'1.2rem', padding:'15px', background:'rgba(255,255,255,0.05)', color:'#fff'}}>
                                {num}
                            </button>
                        ))}
                        <button type="button" onClick={()=>setPassword('')} className="hud-btn" style={{fontSize:'1rem', padding:'15px', background:'rgba(239, 68, 68, 0.2)', color:'#ef4444', fontWeight:'bold'}}>CLR</button>
                        <button type="button" onClick={()=>setPassword(p=>p+'0')} className="hud-btn" style={{fontSize:'1.2rem', padding:'15px', background:'rgba(255,255,255,0.05)', color:'#fff'}}>0</button>
                        <button type="button" onClick={()=>setPassword(p=>p.slice(0,-1))} className="hud-btn" style={{fontSize:'1.2rem', padding:'15px', background:'rgba(255,255,255,0.05)', color:'#fff'}}>⌫</button>
                    </div>

                    <button type="submit" className="hud-btn primary" style={{width:'100%', padding:'15px', fontSize:'1.1rem'}}>AUTHENTICATE</button>
                    {error && <p className="error-msg" style={{marginTop:'15px'}}>{error}</p>}
                </form>
                <Link href="/" className="back-link" style={{color:'#64748b'}}>← RETURN TO KIOSK</Link>
            </div>
        </div>
    );

  return (
    <div className="admin-layout-hud">
        <aside className="admin-sidebar-hud">
            <div className="hud-admin-title">
                <span>🛡️ ADMIN PANEL</span>
            </div>
            <nav style={{flex:1}}>
                <button className={`hud-nav-btn ${activeTab==='dashboard'?'active':''}`} onClick={()=>setActiveTab('dashboard')}>
                    📊 Dashboard
                </button>
                <button className={`hud-nav-btn ${activeTab==='users'?'active':''}`} onClick={()=>setActiveTab('users')}>
                    👥 User Database
                </button>
                <button className={`hud-nav-btn ${activeTab==='enroll'?'active':''}`} onClick={()=>setActiveTab('enroll')}>
                    📸 Enrollment
                </button>
                <button className={`hud-nav-btn ${activeTab==='logs'?'active':''}`} onClick={()=>setActiveTab('logs')}>
                    📜 Access Logs
                </button>
                <button className={`hud-nav-btn ${activeTab==='kiosks'?'active':''}`} onClick={()=>setActiveTab('kiosks')}>
                    🖥 Device Status
                </button>
            </nav>
            <div className="sidebar-footer">
                <button onClick={()=>setToken(null)} className="hud-btn danger" style={{width:'100%'}}>LOGOUT</button>
            </div>
        </aside>
        <main className="admin-content-hud">
            {activeTab === 'dashboard' && <DashboardTab />}
            {activeTab === 'users' && <UsersTab />}
            {activeTab === 'enroll' && <EnrollTab />}
            {activeTab === 'logs' && <AttendanceTab />}
            {activeTab === 'kiosks' && <KiosksTab />}
        </main>
    </div>
  );
}

function DashboardTab() {
    const [stats, setStats] = useState<any>(null);
    useEffect(() => { 
        safeFetch('http://localhost:8000/api/stats').then(data => {
            if (data) setStats(data);
        }); 
    }, []);

    const openDoor = async () => {
        await safeFetch('http://localhost:8000/api/door/open', {
            method: 'POST', body: JSON.stringify({action: 'open', duration: 5}), headers: {'Content-Type':'application/json'}
        });
        alert("Door Unlocked for 5s");
        safeFetch('http://localhost:8000/api/stats').then(data => { if(data) setStats(data); });
    };

    if (!stats) return <div style={{color:'#fff'}}>Loading System Data...</div>;

    const hours = stats.hourly || {};
    const maxVal = Math.max(...Object.values(hours).map((v: any) => Number(v)), 1);

    return (
        <div className="tab-container">
            <div style={{display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'30px'}}>
                <h2 style={{margin:0}}>SYSTEM OVERVIEW</h2>
                <button onClick={openDoor} className="hud-btn primary" style={{background:'#f59e0b', fontSize:'1rem'}}>
                    🔓 OVERRIDE DOOR (5s)
                </button>
            </div>
            
            <div className="hud-stat-grid">
                <div className="hud-stat-card">
                    <h3>Today's Entries</h3>
                    <div className="value">{stats.total_today || 0}</div>
                    <div className="icon">📥</div>
                </div>
                <div className="hud-stat-card">
                    <h3>Unique Persons</h3>
                    <div className="value highlight">{stats.unique_today || 0}</div>
                    <div className="icon">👤</div>
                </div>
                <div className="hud-stat-card">
                    <h3>Security Status</h3>
                    <div className="value" style={{color: stats.door_status==='open'?'#10b981':'#ef4444', fontSize:'1.8rem'}}>
                        {stats.door_status === 'open' ? 'UNLOCKED' : 'SECURE'}
                    </div>
                    <div className="icon">🔒</div>
                </div>
            </div>

            <div className="hud-panel">
                <h3 style={{marginBottom:'20px', color:'#94a3b8', fontSize:'0.9rem', textTransform:'uppercase'}}>HOURLY TRAFFIC ANALYSIS</h3>
                <div style={{display:'flex', alignItems:'flex-end', height:'250px', gap:'6px'}}>
                    {Array.from({length:24}).map((_, h) => {
                        const val = Number(hours[h] || 0);
                        const hPct = (val / maxVal) * 100;
                        return (
                            <div key={h} style={{flex:1, display:'flex', flexDirection:'column', alignItems:'center', gap:'8px', height:'100%', justifySelf: 'flex-end'}}>
                                <div style={{marginTop: 'auto', width:'100%', background: val > 0 ? 'var(--accent)' : '#334155', height:`${Math.max(hPct, 1)}%`, borderRadius:'4px 4px 0 0', opacity: val>0?1:0.3, transition:'height 0.5s', boxShadow: val > 0 ? '0 0 10px var(--accent-glow)' : 'none'}}></div>
                                <span style={{fontSize:'0.65rem', color:'#64748b', fontFamily:'monospace'}}>{h}h</span>
                            </div>
                        )
                    })}
                </div>
            </div>
        </div>
    );
}

function KiosksTab() {
    const [data, setData] = useState<any>(null);
    useEffect(() => { safeFetch('http://localhost:8000/api/stats').then(setData); }, []);
    if (!data) return <div>Loading...</div>;

    return (
        <div className="tab-container">
             <h2 style={{marginBottom:'30px'}}>CONNECTED DEVICES</h2>
             <div className="hud-panel">
                 {data.kiosks && data.kiosks.map((k: any, i: number) => (
                     <div key={i} style={{display:'flex', justifyContent:'space-between', alignItems:'center', padding:'20px', borderBottom:'1px solid rgba(255,255,255,0.05)'}}>
                         <div>
                            <div style={{fontWeight:'bold', fontSize:'1.2rem', color:'#fff'}}>{k.id}</div>
                            <div style={{fontSize:'0.9rem', color:'#94a3b8', marginTop:'5px'}}>Uptime: <span style={{fontFamily:'monospace', color:'#fff'}}>{k.uptime}s</span></div>
                         </div>
                         <div style={{display:'flex', gap:'40px', alignItems:'center'}}>
                             <div style={{textAlign:'right'}}>
                                 <div style={{color: k.status==='online'?'#10b981':'#ef4444', fontWeight:'bold', display:'flex', alignItems:'center', gap:'8px', justifyContent:'flex-end'}}>
                                     <div style={{width:8, height:8, borderRadius:'50%', background: k.status==='online'?'#10b981':'#ef4444', boxShadow: `0 0 8px ${k.status==='online'?'#10b981':'#ef4444'}`}}></div>
                                     {k.status.toUpperCase()}
                                 </div>
                                 <div style={{fontSize:'0.75rem', color:'#64748b', marginTop:'4px'}}>CONNECTION</div>
                             </div>
                             <div style={{textAlign:'right'}}>
                                 <div style={{color: k.camera==='ok'?'#10b981':'#ef4444', fontWeight:'bold'}}>{k.camera.toUpperCase()}</div>
                                 <div style={{fontSize:'0.75rem', color:'#64748b', marginTop:'4px'}}>CAMERA FEED</div>
                             </div>
                         </div>
                     </div>
                 ))}
                 {(!data.kiosks || data.kiosks.length === 0) && <p style={{color:'#64748b', textAlign:'center', padding:'20px'}}>No Active Kiosks Found.</p>}
             </div>
        </div>
    )
}

function UsersTab() {
    const [users, setUsers] = useState<string[]>([]);
    const [selectedUser, setSelectedUser] = useState<string | null>(null);
    const [userImages, setUserImages] = useState<string[]>([]);
    const [userLogs, setUserLogs] = useState<any[]>([]);
    const [filter, setFilter] = useState('');
    const [refresh, setRefresh] = useState(0);
    const [editMode, setEditMode] = useState(false);
    const [newName, setNewName] = useState('');

    useEffect(() => {
        safeFetch('http://localhost:8000/api/users').then(data => { if(Array.isArray(data)) setUsers(data); });
    }, [refresh]);

    useEffect(() => {
        if (selectedUser) {
            safeFetch(`http://localhost:8000/api/users/${selectedUser}/images`)
                .then(data => { if(data) setUserImages(data.images || []); });
            
            safeFetch(`http://localhost:8000/api/users/${selectedUser}/logs`)
                .then(data => { if(Array.isArray(data)) setUserLogs(data); });
            setNewName(selectedUser);
        }
    }, [selectedUser]);

    const filtered = users.filter(u => u.toLowerCase().includes(filter.toLowerCase()));

    const deleteUser = async (name: string) => {
        if (!confirm(`Delete user ${name}? This cannot be undone.`)) return;
        await safeFetch(`http://localhost:8000/api/users/${name}`, { method: 'DELETE' });
        if (selectedUser === name) setSelectedUser(null);
        setRefresh(n => n + 1);
    };

    const handleRename = async () => {
        if (!selectedUser || !newName) return;
        await safeFetch(`http://localhost:8000/api/users/${selectedUser}`, {
            method:'PUT', body:JSON.stringify({new_name: newName}), headers:{'Content-Type':'application/json'}
        });
        setEditMode(false);
        setSelectedUser(null);
        setRefresh(n=>n+1);
    }

    return (
        <div className="tab-container" style={{display: 'flex', gap: '30px', height: '100%'}}>
            <div className="hud-panel" style={{flex: 1, display:'flex', flexDirection:'column', height:'100%', padding:0, overflow:'hidden'}}>
                <div style={{padding:'20px', borderBottom:'1px solid rgba(255,255,255,0.1)'}}>
                    <h2 style={{fontSize:'1.2rem', marginBottom:'15px'}}>PERSONNEL DATABASE</h2>
                    <input 
                        placeholder="🔍 Search ID / Name..." 
                        value={filter} 
                        onChange={e=>setFilter(e.target.value)} 
                        className="hud-input"
                    />
                </div>
                
                <div style={{flex:1, overflowY: 'auto', padding:'10px'}}>
                    {filtered.map(u => (
                        <div key={u} 
                             onClick={() => { setSelectedUser(u); setEditMode(false); }}
                             style={{
                                cursor: 'pointer', 
                                padding: '12px 16px', 
                                marginBottom: '4px',
                                borderRadius: '6px',
                                background: selectedUser === u ? 'rgba(59, 130, 246, 0.2)' : 'transparent',
                                border: `1px solid ${selectedUser === u ? 'rgba(59, 130, 246, 0.5)' : 'transparent'}`,
                                color: selectedUser === u ? '#fff' : '#94a3b8',
                                display: 'flex',
                                alignItems: 'center',
                                gap: '10px'
                             }}>
                            <div style={{width:8, height:8, borderRadius:'50%', background: selectedUser===u?'#3b82f6':'#334155'}}></div>
                            {u}
                        </div>
                    ))}
                </div>
            </div>
            
            <div className="hud-panel" style={{flex: 2, overflowY:'auto'}}>
            {selectedUser ? (
                <>
                    {editMode ? (
                        <div style={{marginBottom:'30px', background:'rgba(0,0,0,0.3)', padding:'20px', borderRadius:'8px', border:'1px solid rgba(255,255,255,0.1)'}}>
                            <h3 style={{marginTop:0}}>RENAME PERSONNEL</h3>
                            <div style={{display:'flex', gap:'10px'}}>
                                <input value={newName} onChange={e=>setNewName(e.target.value)} className="hud-input" style={{flex:1}}/>
                                <button onClick={handleRename} className="hud-btn primary">SAVE</button>
                                <button onClick={()=>setEditMode(false)} className="hud-btn" style={{background:'#333', color:'#fff'}}>CANCEL</button>
                            </div>
                        </div>
                    ) : (
                        <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '30px', borderBottom:'1px solid rgba(255,255,255,0.1)', paddingBottom:'20px'}}>
                            <div>
                                <h1 style={{margin:0, fontSize:'2rem', color:'#fff'}}>{selectedUser}</h1>
                                <div style={{color:'#64748b', fontSize:'0.8rem', marginTop:'5px'}}>ID: {btoa(selectedUser).substring(0,8)}</div>
                            </div>
                            <div style={{display:'flex', gap:'10px'}}>
                                <button onClick={()=>setEditMode(true)} className="hud-btn" style={{background:'rgba(255,255,255,0.1)', color:'#fff'}}>EDIT</button>
                                <button onClick={() => deleteUser(selectedUser)} className="hud-btn danger">DELETE</button>
                            </div>
                        </div>
                    )}

                    <div style={{display:'grid', gridTemplateColumns:'1fr 1fr', gap:'20px', marginBottom:'30px'}}>
                         <div className="hud-stat-card" style={{minHeight:'auto', padding:'15px'}}>
                             <h3>TOTAL ENTRIES</h3>
                             <div className="value" style={{fontSize:'1.8rem'}}>{userLogs.length}</div>
                         </div>
                         <div className="hud-stat-card" style={{minHeight:'auto', padding:'15px'}}>
                             <h3>BIOMETRIC SCANS</h3>
                             <div className="value" style={{fontSize:'1.8rem'}}>{userImages.length}</div>
                         </div>
                    </div>

                    <h3 style={{color:'#94a3b8', fontSize:'0.9rem', marginBottom:'15px'}}>BIOMETRIC DATA</h3>
                    <div className="photo-grid" style={{marginBottom: '30px', display:'flex', gap:'10px', overflowX:'auto', paddingBottom:'10px'}}>
                        {userImages.map((img, i) => (
                            <img key={i} src={img} style={{width: '100px', height: '100px', objectFit: 'cover', borderRadius:'6px', border:'1px solid rgba(255,255,255,0.2)'}} />
                        ))}
                    </div>

                    <h3 style={{color:'#94a3b8', fontSize:'0.9rem', marginBottom:'15px'}}>RECENT ACCESS LOGS</h3>
                    <div className="hud-table-wrapper">
                        <table className="hud-table">
                            <thead><tr><th>TIMESTAMP</th><th>CONFIDENCE SCORE</th><th>STATUS</th></tr></thead>
                            <tbody>
                                {userLogs.slice(-5).reverse().map((log, i) => (
                                    <tr key={i}>
                                        <td>{log.time}</td>
                                        <td style={{fontFamily:'monospace', color:'var(--accent)'}}>{(parseFloat(log.score) * 100).toFixed(1)}%</td>
                                        <td><span style={{padding:'2px 6px', borderRadius:'4px', background:'rgba(16,185,129,0.2)', color:'#10b981', fontSize:'0.7rem'}}>GRANTED</span></td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </>
            ) : (
                <div style={{height:'100%', display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center', color:'#64748b', opacity:0.5}}>
                    <div style={{fontSize:'4rem', marginBottom:'20px'}}>👤</div>
                    <p>SELECT A USER TO VIEW DETAILS</p>
                </div>
            )}
            </div>
        </div>
    );
}

function EnrollTab() {
    const [name, setName] = useState('');
    const [photos, setPhotos] = useState<string[]>([]);
    const [isCapturing, setIsCapturing] = useState(false);
    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [status, setStatus] = useState('');

    useEffect(() => {
        navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } })
            .then(stream => { if(videoRef.current) videoRef.current.srcObject = stream; })
            .catch(e => console.error(e));
    }, []);

    const capture = () => {
        if (!videoRef.current || !canvasRef.current) return;
        const ctx = canvasRef.current.getContext('2d');
        if (!ctx) return;
        canvasRef.current.width = videoRef.current.videoWidth;
        canvasRef.current.height = videoRef.current.videoHeight;
        ctx.translate(canvasRef.current.width, 0); ctx.scale(-1, 1);
        ctx.drawImage(videoRef.current, 0, 0);
        setPhotos(prev => [...prev, canvasRef.current!.toDataURL('image/jpeg')]);
    };

    useEffect(() => {
        let interval: NodeJS.Timeout;
        if (isCapturing) {
            interval = setInterval(() => {
                if (photos.length >= 5) { setIsCapturing(false); return; }
                capture();
            }, 500);
        }
        return () => clearInterval(interval);
    }, [isCapturing, photos.length]);

    const register = async () => {
        setStatus('Registering...');
        const res = await safeFetch('http://localhost:8000/api/register', {
            method: 'POST', headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ name, images: photos })
        });
        
        if (res && res.status === 'success') {
            setStatus(`SUCCESS: ${name} REGISTERED`);
            setName(''); setPhotos([]);
        } else {
            setStatus('ERROR: REGISTRATION FAILED');
        }
    };

    return (
        <div className="tab-container">
            <h2 style={{marginBottom:'30px'}}>NEW REGISTRATION</h2>
            <div style={{display:'flex', gap:'30px', height:'calc(100vh - 200px)'}}>
                <div className="hud-panel" style={{flex: 1, display:'flex', flexDirection:'column'}}>
                    <div style={{position:'relative', width:'100%', flex:1, background:'#000', borderRadius:'8px', overflow:'hidden', border:'1px solid #333'}}>
                        <video ref={videoRef} autoPlay playsInline muted style={{width:'100%', height:'100%', objectFit:'cover', transform: 'scaleX(-1)'}} />
                        <div style={{position:'absolute', top:0, left:0, padding:'10px', background:'rgba(0,0,0,0.5)', color:'red', fontWeight:'bold'}}>● LIVE FEED</div>
                    </div>
                    
                    <div className="controls" style={{marginTop:'20px', display:'flex', flexDirection:'column', gap:'15px'}}>
                        <input value={name} onChange={e => setName(e.target.value)} placeholder="ENTER FULL NAME" className="hud-input"/>
                        <div style={{display: 'flex', gap: '15px'}}>
                            <button onClick={capture} className="hud-btn" style={{flex: 1, background:'#333', color:'#fff'}}>
                                📷 CAPTURE ({photos.length}/5)
                            </button>
                            <button onClick={() => setIsCapturing(!isCapturing)} className="hud-btn primary" style={{flex: 1, background: isCapturing ? '#ef4444' : 'var(--accent)'}}>
                                {isCapturing ? '⏹ STOP' : '▶ AUTO CAPTURE (5)'}
                            </button>
                        </div>
                    </div>
                </div>

                <div className="hud-panel" style={{flex: 1, display:'flex', flexDirection:'column'}}>
                    <h3 style={{marginTop:0, marginBottom:'20px', color:'#94a3b8', fontSize:'0.9rem'}}>CAPTURED SAMPLES</h3>
                     <div style={{flex:1, overflowY:'auto', display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(100px, 1fr))', gap:'10px', alignContent:'start'}}>
                         {photos.map((p, i) => (
                            <img key={i} src={p} onClick={()=>setPhotos(photos.filter((_,x)=>x!==i))} style={{width:'100%', aspectRatio:'1', objectFit:'cover', borderRadius:'4px', cursor:'pointer', border:'1px solid rgba(255,255,255,0.2)'}} title="Click to remove" />
                         ))}
                         {photos.length === 0 && <p style={{gridColumn:'1/-1', color:'#64748b', textAlign:'center', marginTop:'50px'}}>No samples captured yet.</p>}
                     </div>
                     
                     <div style={{marginTop:'20px'}}>
                        <button onClick={register} disabled={!name || photos.length === 0} className="hud-btn primary" style={{width:'100%', opacity: (!name || photos.length === 0) ? 0.5 : 1}}>
                            REGISTER USER
                        </button>
                        {status && <p style={{marginTop:'10px', textAlign:'center', color: status.includes('SUCCESS') ? '#10b981' : '#ef4444'}}>{status}</p>}
                     </div>
                     
                     <canvas ref={canvasRef} style={{display:'none'}}/>
                </div>
            </div>
        </div>
    );
}

function AttendanceTab() {
    const [logs, setLogs] = useState<any[]>([]);
    useEffect(() => { 
        safeFetch('http://localhost:8000/api/full_stats').then(data => { if(Array.isArray(data)) setLogs(data); });
    }, []);
    
    return (
        <div className="tab-container">
            <h2 style={{marginBottom:'30px'}}>ACCESS LOGS</h2>
            <div className="hud-table-wrapper">
                <table className="hud-table">
                    <thead><tr><th>TIMESTAMP</th><th>PERSONNEL</th><th>CONFIDENCE</th><th>DEVICE</th></tr></thead>
                    <tbody>
                        {logs.slice().reverse().map((row, i) => (
                            <tr key={i}>
                                <td>{row[0]}</td>
                                <td style={{fontWeight:'bold', color:'#fff'}}>{row[1]}</td>
                                <td style={{fontFamily:'monospace', color:'var(--accent)'}}>{(parseFloat(row[2]) * 100).toFixed(1)}%</td>
                                <td>CAM-01</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </div>
    );
}
