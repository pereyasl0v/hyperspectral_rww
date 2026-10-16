import React, { useEffect, useMemo, useState } from 'react'
import { createRoot } from 'react-dom/client'
import { Activity, Database, Radio, Play, Upload, Layers, ScanSearch } from 'lucide-react'
import './styles.css'

const API = import.meta.env.VITE_API_URL || 'http://localhost:8000'

function fmtBytes(n) {
  if (n == null) return '—'
  const units = ['B', 'KB', 'MB', 'GB']
  let x = n, i = 0
  while (x >= 1024 && i < units.length - 1) { x /= 1024; i++ }
  return `${x.toFixed(i ? 2 : 0)} ${units[i]}`
}

function App() {
  const [meta, setMeta] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [analysis, setAnalysis] = useState(null)
  const [hovered, setHovered] = useState(null)
  const [view, setView] = useState('rgb')
  const [r, setR] = useState(48), [g, setG] = useState(32), [b, setB] = useState(16), [band, setBand] = useState(32)
  const [channelMbps, setChannelMbps] = useState(10)
  const [sessionSeconds, setSessionSeconds] = useState(5)
  const [params, setParams] = useState({stride:4, seed:99.5, growth:85, angle:12, minPixels:10})
  const [previewKey, setPreviewKey] = useState(0)

  async function refreshMeta() {
    try {
      const res = await fetch(`${API}/api/dataset`)
      if (!res.ok) throw new Error((await res.json()).detail || 'No dataset')
      setMeta(await res.json()); setError('')
    } catch (e) { setError(e.message); setMeta(null) }
  }
  useEffect(() => { refreshMeta() }, [])

  const preview = useMemo(() => {
    if (!meta) return ''
    if (view === 'band') return `${API}/api/preview.png?band=${band}&k=${previewKey}`
    return `${API}/api/preview.png?r=${r}&g=${g}&b=${b}&k=${previewKey}`
  }, [meta, view, band, r, g, b, previewKey])

  async function upload(e) {
    const f = e.target.files?.[0]; if (!f) return
    setBusy(true); setError(''); setAnalysis(null)
    const body = new FormData(); body.append('file', f)
    try {
      const res = await fetch(`${API}/api/dataset/upload`, {method:'POST', body})
      if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed')
      setMeta(await res.json()); setPreviewKey(k => k + 1)
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  async function run() {
    setBusy(true); setError(''); setHovered(null)
    try {
      const res = await fetch(`${API}/api/analyze`, {
        method:'POST', headers:{'Content-Type':'application/json'},
        body: JSON.stringify({
          channel_mbps:+channelMbps, session_seconds:+sessionSeconds,
          analysis_stride:+params.stride, seed_percentile:+params.seed,
          growth_percentile:+params.growth, spectral_angle_deg:+params.angle,
          min_region_pixels:+params.minPixels, pca_components:8, local_tile:32,
          max_regions:200, regularization:0.05
        })
      })
      const data = await res.json(); if (!res.ok) throw new Error(data.detail || 'Analysis failed')
      setAnalysis(data)
    } catch (e) { setError(e.message) } finally { setBusy(false) }
  }

  const overlay = hovered && analysis ? `${API}/api/analysis/${analysis.analysis_id}/region/${hovered}.png` : null

  return <div className="app">
    <header>
      <div><div className="eyebrow">ONBOARD HYPERSPECTRAL DECISION SUPPORT</div><h1>Hyperspectral Downlink Planner</h1></div>
      <div className="status"><span className={meta ? 'dot ok' : 'dot'}></span>{meta ? `${meta.frames} × ${meta.width} × ${meta.bands}` : 'No RWW loaded'}</div>
    </header>

    <main>
      <section className="leftCol">
        <div className="panel controls">
          <div className="controlGroup">
            <label><Database size={16}/> Dataset</label>
            <div className="datasetLine">{meta ? <><b>{fmtBytes(meta.file_size)}</b><span>1024-byte header discarded · uint8 ADC</span></> : <span>Load an .rww scene</span>}</div>
            <label className="uploadBtn"><Upload size={15}/> {busy ? 'Working…' : 'Load RWW'}<input type="file" accept=".rww" onChange={upload}/></label>
          </div>
          <div className="controlGroup">
            <label><Radio size={16}/> Downlink budget</label>
            <div className="row"><Field label="Mbit/s" value={channelMbps} set={setChannelMbps}/><Field label="seconds" value={sessionSeconds} set={setSessionSeconds}/></div>
            <div className="budget">Capacity: <b>{((channelMbps * sessionSeconds)).toFixed(1)} Mbit</b> / {(channelMbps*sessionSeconds/8).toFixed(2)} MB</div>
          </div>
          <button className="runBtn" disabled={!meta || busy} onClick={run}><Play size={17} fill="currentColor"/>{busy ? 'Computing LRXD…' : 'Analyze & optimize'}</button>
        </div>

        <div className="panel viewerPanel">
          <div className="viewerToolbar">
            <div className="seg"><button className={view==='rgb'?'active':''} onClick={()=>setView('rgb')}>Pseudo RGB</button><button className={view==='band'?'active':''} onClick={()=>setView('band')}>Single band</button></div>
            {view==='rgb' ? <div className="bands"><BandField name="R" v={r} set={setR}/><BandField name="G" v={g} set={setG}/><BandField name="B" v={b} set={setB}/></div> : <BandField name="Band" v={band} set={setBand}/>}          
          </div>
          <div className="imageStage">
            {meta ? <div className="imageStack">
              <img src={preview} className="scene"/>
              {analysis && <img src={`${API}/api/analysis/${analysis.analysis_id}/regions.png`} className="overlay regions"/>}
              {overlay && <img src={overlay} className="overlay hover"/>}
            </div> : <div className="empty"><Layers size={42}/><span>Load an RWW file to build the 64-band cube</span></div>}
          </div>
          <div className="legend"><span><i className="swatch regionSwatch"></i>detected regions</span><span><i className="swatch hoverSwatch"></i>hovered region</span><span>ADC channels are not calibrated wavelengths.</span></div>
        </div>
      </section>

      <aside className="rightCol">
        <div className="panel method">
          <div className="panelTitle"><ScanSearch size={17}/> Analysis parameters</div>
          <Param label="Analysis stride" value={params.stride} min={1} max={16} step={1} set={v=>setParams({...params,stride:v})}/>
          <Param label="Seed percentile" value={params.seed} min={95} max={99.99} step={0.1} set={v=>setParams({...params,seed:v})}/>
          <Param label="Growth percentile" value={params.growth} min={81} max={99.9} step={0.5} set={v=>setParams({...params,growth:v})}/>
          <Param label="SAM threshold, °" value={params.angle} min={1} max={30} step={0.5} set={v=>setParams({...params,angle:v})}/>
          <Param label="Min region, low-res px" value={params.minPixels} min={1} max={100} step={1} set={v=>setParams({...params,minPixels:v})}/>
        </div>

        <div className="panel queue">
          <div className="queueHead"><div><div className="panelTitle"><Activity size={17}/> Transmission queue</div>{analysis && <small>{analysis.selected_count}/{analysis.region_count} regions · {analysis.used_mbit.toFixed(2)}/{analysis.capacity_mbit.toFixed(2)} Mbit</small>}</div>{analysis && <span className="pill">PCA {(analysis.pca_explained_variance*100).toFixed(1)}%</span>}</div>
          {!analysis ? <div className="emptyQueue">Run the analysis. The knapsack optimizer will select the maximum-utility subset that fits the downlink budget.</div> :
            <div className="regionList">{analysis.regions.map(x => <RegionCard key={x.id} r={x} onEnter={()=>setHovered(x.id)} onLeave={()=>setHovered(null)}/>)}</div>}
        </div>
      </aside>
    </main>
    {error && <div className="toast">{error}</div>}
  </div>
}

function Field({label,value,set}) { return <label className="field"><span>{label}</span><input type="number" value={value} min="0.1" step="0.1" onChange={e=>set(+e.target.value)}/></label> }
function BandField({name,v,set}) { return <label className="bandField"><span>{name}</span><input type="number" min="0" max="63" value={v} onChange={e=>set(Math.max(0,Math.min(63,+e.target.value)))}/></label> }
function Param({label,value,set,min,max,step}) { return <label className="param"><div><span>{label}</span><b>{value}</b></div><input type="range" min={min} max={max} step={step} value={value} onChange={e=>set(+e.target.value)}/></label> }
function RegionCard({r,onEnter,onLeave}) {
  return <div className={`regionCard ${r.selected?'selected':'rejected'}`} onMouseEnter={onEnter} onMouseLeave={onLeave}>
    <div className="order">{r.selected ? r.transmission_order : '—'}</div>
    <div className="regionBody"><div className="regionTop"><b>Region {r.id}</b><span>{fmtBytes(r.bytes)}</span></div><div className="metrics"><span>max LRX <b>{r.max_lrx.toFixed(1)}</b></span><span>mean <b>{r.mean_lrx.toFixed(1)}</b></span><span>coh. <b>{(r.spectral_coherence*100).toFixed(0)}%</b></span></div><div className="bar"><i style={{width:`${Math.min(100,r.utility*100)}%`}}></i></div><div className="foot"><span>utility {r.utility.toFixed(3)}</span><span>{r.selected ? `${r.transmission_start_s.toFixed(2)}–${r.transmission_end_s.toFixed(2)} s` : 'not selected'}</span></div></div>
  </div>
}

createRoot(document.getElementById('root')).render(<App />)
