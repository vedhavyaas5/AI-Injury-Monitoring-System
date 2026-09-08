import React, { useState } from 'react'
import { c, card, input, btn, btnOutline, label, section } from '../styles'

const API = '/api'

const PAIN_LOCATIONS = ['Knee','Ankle','Hip','Shoulder','Elbow','Back','Hamstring','Quadriceps','Other']
const SYMPTOMS       = [
  'Muscle soreness','Joint stiffness','Swelling','Weakness',
  'Numbness / tingling','Reduced range of motion',
  'Difficulty walking','Difficulty running','Difficulty changing direction',
]
const INTENSITIES    = [
  { v:1, label:'Very Low' },{ v:2, label:'Low' },{ v:3, label:'Moderate' },
  { v:4, label:'High' },{ v:5, label:'Very High' },
]
const SESSION_TYPES  = ['Training','Match','Recovery','Warm-Up','Other']

function SliderField({ label: lbl, value, onChange, min=0, max=10, color }) {
  const pct = ((value - min) / (max - min)) * 100
  const levelColor = value >= 7 ? c.red : value >= 4 ? c.yellow : c.green
  return (
    <div style={{ marginBottom: 20 }}>
      <div style={{ display:'flex', justifyContent:'space-between',
        alignItems:'center', marginBottom: 8 }}>
        <span style={{ ...label, marginBottom:0 }}>{lbl}</span>
        <span style={{ fontSize:20, fontWeight:800,
          color: color || levelColor }}>{value}<span style={{
          fontSize:11, color:c.muted }}> / {max}</span></span>
      </div>
      <input type="range" min={min} max={max} value={value}
        onChange={e => onChange(Number(e.target.value))}
        style={{ width:'100%', accentColor: color || levelColor, height: 6 }} />
      <div style={{ display:'flex', justifyContent:'space-between',
        fontSize:10, color:c.muted, marginTop:4 }}>
        <span>None ({min})</span><span>Severe ({max})</span>
      </div>
    </div>
  )
}

function ToggleChip({ label: lbl, selected, onClick }) {
  return (
    <button onClick={onClick} style={{
      padding: '7px 14px', borderRadius: 20, fontSize: 12,
      fontWeight: 500, cursor: 'pointer', transition: 'all .15s',
      background: selected ? `${c.blue}30` : 'transparent',
      border: `1px solid ${selected ? c.blue : c.border}`,
      color: selected ? c.blue : c.muted,
    }}>
      {selected ? '✓ ' : ''}{lbl}
    </button>
  )
}

export default function SelfAssessment({ onBack, onDone }) {
  const [athleteId,    setAthleteId]    = useState('')
  const [sessionType,  setSessionType]  = useState('Training')
  const [duration,     setDuration]     = useState(60)
  const [intensity,    setIntensity]    = useState(3)
  const [painLevel,    setPainLevel]    = useState(0)
  const [painLocs,     setPainLocs]     = useState([])
  const [symptoms,     setSymptoms]     = useState([])
  const [fatigue,      setFatigue]      = useState(0)
  const [prevInj,      setPrevInj]      = useState(false)
  const [prevInjLoc,   setPrevInjLoc]   = useState('')
  const [recovery,     setRecovery]     = useState('')
  const [description,  setDescription]  = useState('')
  const [loading,      setLoading]      = useState(false)
  const [error,        setError]        = useState('')

  const toggleChip = (arr, setArr, val) =>
    setArr(arr.includes(val) ? arr.filter(v => v !== val) : [...arr, val])

  const handleSubmit = async () => {
    if (!athleteId.trim()) { setError('Please enter Athlete ID.'); return }
    setError('')
    setLoading(true)
    try {
      const payload = {
        athlete_id:             athleteId.trim(),
        session_type:           sessionType,
        training_duration_min:  duration,
        training_intensity:     intensity,
        pain_level:             painLevel,
        pain_locations:         painLocs,
        symptoms:               symptoms,
        fatigue_level:          fatigue,
        previous_injury:        prevInj,
        previous_injury_location: prevInj && prevInjLoc ? prevInjLoc : null,
        recovery_status:        prevInj && recovery ? recovery : null,
        description:            description || null,
      }
      const res  = await fetch(`${API}/self-assessment`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || `Server error ${res.status}`)
      }
      const data = await res.json()
      onDone(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const inputFocus = {
    outline: `1px solid ${c.blue}`,
    borderColor: c.blue,
  }

  return (
    <div style={{
      minHeight:'100vh', background:c.bg, padding:'24px 16px',
      display:'flex', flexDirection:'column', alignItems:'center',
    }}>
      <div style={{ width:'100%', maxWidth:680 }}>

        {/* Header */}
        <div style={{ display:'flex', alignItems:'center', gap:12,
          marginBottom:28 }}>
          <button onClick={onBack} style={{
            ...btnOutline, padding:'8px 14px', fontSize:13,
          }}>← Back</button>
          <div>
            <div style={{ fontSize:11, color:c.muted, fontWeight:700,
              letterSpacing:'0.1em', textTransform:'uppercase' }}>
              SPORTS INJURY MONITOR
            </div>
            <h1 style={{ fontSize:22, fontWeight:800, color:'#fff' }}>
              Athlete Self Assessment
            </h1>
          </div>
        </div>

        {/* ── Athlete Info ─────────────────────────────────────────────────── */}
        <div style={{ ...card, marginBottom:16 }}>
          <div style={{ fontSize:13, fontWeight:700, color:c.blue,
            letterSpacing:'0.08em', textTransform:'uppercase',
            marginBottom:18, display:'flex', alignItems:'center', gap:8 }}>
            <span>👤</span> Athlete Information
          </div>

          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14,
            marginBottom:14 }}>
            <div>
              <span style={label}>Athlete ID *</span>
              <input placeholder="e.g. 07" value={athleteId}
                onChange={e => setAthleteId(e.target.value)}
                style={input} />
            </div>
            <div>
              <span style={label}>Session Type</span>
              <select value={sessionType}
                onChange={e => setSessionType(e.target.value)}
                style={{ ...input, appearance:'none' }}>
                {SESSION_TYPES.map(t => (
                  <option key={t} value={t}>{t}</option>
                ))}
              </select>
            </div>
          </div>

          <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14 }}>
            <div>
              <span style={label}>Training Duration (min)</span>
              <input type="number" min={0} max={480} value={duration}
                onChange={e => setDuration(Number(e.target.value))}
                style={input} />
            </div>
            <div>
              <span style={label}>Training Intensity</span>
              <div style={{ display:'flex', gap:6, flexWrap:'wrap' }}>
                {INTENSITIES.map(({ v, label:lbl }) => (
                  <ToggleChip key={v} label={lbl}
                    selected={intensity === v}
                    onClick={() => setIntensity(v)} />
                ))}
              </div>
            </div>
          </div>
        </div>

        {/* ── Pain ─────────────────────────────────────────────────────────── */}
        <div style={{ ...card, marginBottom:16 }}>
          <div style={{ fontSize:13, fontWeight:700, color:c.red,
            letterSpacing:'0.08em', textTransform:'uppercase',
            marginBottom:18, display:'flex', alignItems:'center', gap:8 }}>
            <span>🔴</span> Pain
          </div>

          <SliderField label="Pain Level" value={painLevel}
            onChange={setPainLevel} />

          <span style={label}>Pain Location (select all that apply)</span>
          <div style={{ display:'flex', flexWrap:'wrap', gap:8, marginTop:4 }}>
            {PAIN_LOCATIONS.map(loc => (
              <ToggleChip key={loc} label={loc}
                selected={painLocs.includes(loc)}
                onClick={() => toggleChip(painLocs, setPainLocs, loc)} />
            ))}
          </div>
        </div>

        {/* ── Symptoms ─────────────────────────────────────────────────────── */}
        <div style={{ ...card, marginBottom:16 }}>
          <div style={{ fontSize:13, fontWeight:700, color:c.yellow,
            letterSpacing:'0.08em', textTransform:'uppercase',
            marginBottom:18, display:'flex', alignItems:'center', gap:8 }}>
            <span>⚡</span> Symptoms
          </div>
          <span style={label}>Select all that apply</span>
          <div style={{ display:'flex', flexWrap:'wrap', gap:8, marginTop:4 }}>
            {SYMPTOMS.map(s => (
              <ToggleChip key={s} label={s}
                selected={symptoms.includes(s)}
                onClick={() => toggleChip(symptoms, setSymptoms, s)} />
            ))}
          </div>
        </div>

        {/* ── Fatigue ──────────────────────────────────────────────────────── */}
        <div style={{ ...card, marginBottom:16 }}>
          <div style={{ fontSize:13, fontWeight:700, color:c.purple,
            letterSpacing:'0.08em', textTransform:'uppercase',
            marginBottom:18, display:'flex', alignItems:'center', gap:8 }}>
            <span>🔋</span> Fatigue
          </div>
          <SliderField label="Fatigue Level" value={fatigue}
            onChange={setFatigue} color={c.purple} />
        </div>

        {/* ── Previous Injury ───────────────────────────────────────────────── */}
        <div style={{ ...card, marginBottom:16 }}>
          <div style={{ fontSize:13, fontWeight:700, color:c.muted,
            letterSpacing:'0.08em', textTransform:'uppercase',
            marginBottom:18, display:'flex', alignItems:'center', gap:8 }}>
            <span>🏥</span> Previous Injury
          </div>

          <div style={{ display:'flex', gap:10, marginBottom: prevInj?16:0 }}>
            {[{v:false,l:'No'},{v:true,l:'Yes'}].map(({ v, l }) => (
              <button key={l} onClick={() => setPrevInj(v)} style={{
                padding:'10px 24px', borderRadius:8, fontSize:13,
                fontWeight:600, cursor:'pointer',
                background: prevInj === v ? (v ? `${c.red}22`:`${c.green}22`) : 'transparent',
                border:`1px solid ${prevInj===v ? (v?c.red:c.green) : c.border}`,
                color: prevInj === v ? (v?c.red:c.green) : c.muted,
              }}>{l}</button>
            ))}
          </div>

          {prevInj && (
            <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:14,
              marginTop:14 }}>
              <div>
                <span style={label}>Previous Injury Location</span>
                <select value={prevInjLoc}
                  onChange={e => setPrevInjLoc(e.target.value)}
                  style={{ ...input, appearance:'none' }}>
                  <option value="">Select</option>
                  {PAIN_LOCATIONS.map(l => (
                    <option key={l} value={l}>{l}</option>
                  ))}
                </select>
              </div>
              <div>
                <span style={label}>Recovery Status</span>
                <select value={recovery}
                  onChange={e => setRecovery(e.target.value)}
                  style={{ ...input, appearance:'none' }}>
                  <option value="">Select</option>
                  {['Fully recovered','Partially recovered','Ongoing'].map(r => (
                    <option key={r} value={r}>{r}</option>
                  ))}
                </select>
              </div>
            </div>
          )}
        </div>

        {/* ── Description ──────────────────────────────────────────────────── */}
        <div style={{ ...card, marginBottom:24 }}>
          <div style={{ fontSize:13, fontWeight:700, color:c.blue,
            letterSpacing:'0.08em', textTransform:'uppercase',
            marginBottom:18, display:'flex', alignItems:'center', gap:8 }}>
            <span>💬</span> Athlete's Description
          </div>
          <span style={label}>Describe how you feel (optional)</span>
          <textarea
            placeholder={'e.g. "My right knee hurts after running and feels stiff when I bend it."'}
            value={description}
            onChange={e => setDescription(e.target.value)}
            rows={4}
            style={{ ...input, resize:'vertical', lineHeight:1.6 }}
          />
          <p style={{ fontSize:11, color:c.muted, marginTop:6 }}>
            Symptom keywords will be extracted for risk analysis only.
            This is not used for medical diagnosis.
          </p>
        </div>

        {/* Error */}
        {error && (
          <div style={{
            padding:'12px 16px', borderRadius:10, marginBottom:16,
            background:`${c.red}15`, border:`1px solid ${c.red}40`,
            color:c.red, fontSize:13,
          }}>
            ⚠ {error}
          </div>
        )}

        {/* Submit */}
        <button onClick={handleSubmit} disabled={loading}
          style={{
            ...btn(), width:'100%', fontSize:16,
            opacity: loading ? 0.6 : 1,
            cursor: loading ? 'not-allowed' : 'pointer',
          }}>
          {loading ? 'Analysing…' : '📊 Submit Assessment'}
        </button>

        {/* Disclaimer */}
        <p style={{ textAlign:'center', fontSize:11, color:c.muted,
          marginTop:16, lineHeight:1.6 }}>
          Responses generate risk indicators for decision support only.
          They do not constitute a medical diagnosis.
        </p>

      </div>
    </div>
  )
}
