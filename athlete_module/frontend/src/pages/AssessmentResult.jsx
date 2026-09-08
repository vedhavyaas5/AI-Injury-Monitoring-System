import React from 'react'
import { c, card, btn, btnOutline } from '../styles'

function Row({ label, value, highlight }) {
  return (
    <div style={{
      display:'flex', justifyContent:'space-between', alignItems:'center',
      padding:'10px 0', borderBottom:`1px solid ${c.border}`,
    }}>
      <span style={{ fontSize:13, color:c.muted }}>{label}</span>
      <span style={{
        fontSize:13, fontWeight:600,
        color: highlight === 'detected' ? c.yellow
             : highlight === 'good'    ? c.green
             : '#fff',
      }}>{value}</span>
    </div>
  )
}

function PainBar({ value }) {
  const color = value >= 7 ? c.red : value >= 4 ? c.yellow : c.green
  return (
    <div style={{ marginBottom:4 }}>
      <div style={{ height:8, borderRadius:4, background:'#1e2d45',
        overflow:'hidden' }}>
        <div style={{
          width:`${(value/10)*100}%`, height:'100%',
          background:color, borderRadius:4,
          transition:'width .6s ease',
        }} />
      </div>
    </div>
  )
}

export default function AssessmentResult({ data, onMRI, onHome }) {
  if (!data) return null

  const f   = data.features
  const ind = data.indicators
  const status = data.status_label || 'Assessment Complete'
  const isElevated = status.toLowerCase().includes('elevated')
  const statusColor = isElevated ? c.red
    : status.toLowerCase().includes('moderate') ? c.yellow : c.green

  // Risk score 0–100
  const riskPct = Math.round((f.overall_risk_score || 0) * 100)

  // Symptom detected flags
  const detected = (v) => v ? 'Detected' : 'Not Detected'
  const detectedH = (v) => v ? 'detected' : 'good'

  return (
    <div style={{
      minHeight:'100vh', background:c.bg, padding:'24px 16px',
      display:'flex', flexDirection:'column', alignItems:'center',
    }}>
      <div style={{ width:'100%', maxWidth:620 }}>

        {/* Header */}
        <div style={{ textAlign:'center', marginBottom:28 }}>
          <div style={{ fontSize:11, color:c.muted, fontWeight:700,
            letterSpacing:'0.1em', textTransform:'uppercase', marginBottom:6 }}>
            SPORTS INJURY MONITOR
          </div>
          <h1 style={{ fontSize:24, fontWeight:800, color:'#fff' }}>
            Athlete Assessment
          </h1>
          <div style={{ fontSize:12, color:c.muted, marginTop:4 }}>
            Athlete #{data.athlete_id} · Session {data.session_id}
          </div>
        </div>

        {/* Risk Score Hero */}
        <div style={{
          ...card, textAlign:'center', marginBottom:16,
          background: isElevated ? `${c.red}0a` : `${c.green}0a`,
          border:`1px solid ${statusColor}40`,
        }}>
          <div style={{ fontSize:13, fontWeight:700, letterSpacing:'0.08em',
            textTransform:'uppercase', color:c.muted, marginBottom:12 }}>
            Self-Assessment Risk Score
          </div>
          <div style={{ fontSize:56, fontWeight:900,
            color:statusColor, lineHeight:1 }}>
            {riskPct}
            <span style={{ fontSize:24, color:c.muted }}>%</span>
          </div>
          <div style={{
            display:'inline-block', marginTop:12, padding:'6px 20px',
            borderRadius:20, fontSize:13, fontWeight:700,
            background:`${statusColor}20`,
            border:`1px solid ${statusColor}50`,
            color:statusColor,
          }}>
            {status}
          </div>
          <p style={{ fontSize:11, color:c.muted, marginTop:12 }}>
            Composite score based on pain, fatigue, symptoms and training load.
            This is an AI-based indicator — not a medical diagnosis.
          </p>
        </div>

        {/* Metrics */}
        <div style={{ ...card, marginBottom:16 }}>
          <div style={{ fontSize:12, fontWeight:700, letterSpacing:'0.08em',
            textTransform:'uppercase', color:c.blue, marginBottom:14 }}>
            📊 Assessment Summary
          </div>

          <div style={{ marginBottom:12 }}>
            <div style={{ display:'flex', justifyContent:'space-between',
              marginBottom:4, fontSize:13 }}>
              <span style={{ color:c.muted }}>Pain Level</span>
              <span style={{ fontWeight:700,
                color: f.pain_level >= 7 ? c.red : f.pain_level >= 4 ? c.yellow : c.green }}>
                {f.pain_level} / 10
              </span>
            </div>
            <PainBar value={f.pain_level} />
          </div>

          <div style={{ marginBottom:12 }}>
            <div style={{ display:'flex', justifyContent:'space-between',
              marginBottom:4, fontSize:13 }}>
              <span style={{ color:c.muted }}>Fatigue Level</span>
              <span style={{ fontWeight:700,
                color: f.fatigue_level >= 7 ? c.red : f.fatigue_level >= 4 ? c.yellow : c.green }}>
                {f.fatigue_level} / 10
              </span>
            </div>
            <PainBar value={f.fatigue_level} />
          </div>

          <Row label="Stiffness"           value={detected(ind.stiffness_detected)}   highlight={detectedH(ind.stiffness_detected)} />
          <Row label="Swelling"            value={detected(ind.swelling_detected)}    highlight={detectedH(ind.swelling_detected)} />
          <Row label="Weakness"            value={detected(ind.weakness_detected)}    highlight={detectedH(ind.weakness_detected)} />
          <Row label="Reduced ROM"         value={detected(ind.reduced_rom_detected)} highlight={detectedH(ind.reduced_rom_detected)} />
          <Row label="Movement Difficulty" value={detected(ind.movement_difficulty_detected)} highlight={detectedH(ind.movement_difficulty_detected)} />
          <Row label="Previous Injury"     value={detected(ind.previous_injury_detected)}    highlight={detectedH(ind.previous_injury_detected)} />
          <Row label="Training Intensity"  value={`${f.training_intensity} / 5`} />
          <Row label="Symptoms Reported"   value={f.n_symptoms} />
        </div>

        {/* Text keyword extraction */}
        {data.risk_fusion_payload?.text_keywords &&
          Object.keys(data.risk_fusion_payload.text_keywords).length > 0 && (
          <div style={{ ...card, marginBottom:16 }}>
            <div style={{ fontSize:12, fontWeight:700, letterSpacing:'0.08em',
              textTransform:'uppercase', color:c.purple, marginBottom:12 }}>
              🔍 Text Analysis — Symptom Keywords Detected
            </div>
            <p style={{ fontSize:11, color:c.muted, marginBottom:10,
              fontStyle:'italic' }}>
              Keyword extraction only — not a medical interpretation.
            </p>
            <div style={{ display:'flex', flexWrap:'wrap', gap:8 }}>
              {Object.entries(data.risk_fusion_payload.text_keywords).map(
                ([k, v]) => (
                  <div key={k} style={{
                    padding:'6px 12px', borderRadius:16,
                    background:`${c.purple}15`, border:`1px solid ${c.purple}30`,
                    fontSize:12, color:c.purple,
                  }}>
                    <span style={{ color:c.muted }}>{k}:</span> {v}
                  </div>
                )
              )}
            </div>
          </div>
        )}

        {/* Risk-fusion ready indicator */}
        <div style={{
          ...card, marginBottom:16,
          background:`${c.blue}08`, border:`1px solid ${c.blue}30`,
        }}>
          <div style={{ fontSize:12, fontWeight:700, letterSpacing:'0.08em',
            textTransform:'uppercase', color:c.blue, marginBottom:10 }}>
            🔗 Risk Fusion Payload Ready
          </div>
          <p style={{ fontSize:12, color:c.muted, marginBottom:10 }}>
            This assessment has been structured and is ready to combine with
            Movement Model, Sensor Model, and MRI results in the Risk Fusion Engine.
          </p>
          <div style={{ fontSize:11, fontFamily:'monospace',
            background:'#0a0e1a', border:`1px solid ${c.border}`,
            borderRadius:8, padding:12, color:'#94a3b8',
            whiteSpace:'pre-wrap', wordBreak:'break-all', maxHeight:120,
            overflow:'auto',
          }}>
            {JSON.stringify({
              athlete_id: data.athlete_id,
              self_assessment: {
                pain_level:  f.pain_level,
                fatigue_level: f.fatigue_level,
                overall_risk_score: f.overall_risk_score,
              },
              mri: { available: false },
            }, null, 2)}
          </div>
        </div>

        {/* Disclaimer */}
        <div style={{
          padding:'12px 16px', borderRadius:10, marginBottom:24,
          background:`${c.yellow}10`, border:`1px solid ${c.yellow}30`,
        }}>
          <p style={{ fontSize:11, color:`${c.yellow}cc`, lineHeight:1.6 }}>
            ⚠ {data.disclaimer}
          </p>
        </div>

        {/* Action buttons */}
        <div style={{ display:'flex', gap:12, flexWrap:'wrap' }}>
          <button onClick={onMRI} style={{
            ...btn(c.purple), flex:'1 1 180px', fontSize:14,
          }}>
            🩻 Upload MRI Scan
          </button>
          <button onClick={onHome} style={{
            ...btnOutline, flex:'1 1 140px', fontSize:14,
          }}>
            🏠 Home
          </button>
        </div>

      </div>
    </div>
  )
}
