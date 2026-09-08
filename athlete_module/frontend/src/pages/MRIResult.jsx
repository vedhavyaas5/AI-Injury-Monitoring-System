import React from 'react'
import { c, card, btn, btnOutline } from '../styles'

function ConfBar({ value }) {
  const color = value >= 0.75 ? c.green : value >= 0.5 ? c.yellow : c.red
  return (
    <div style={{ height:8, borderRadius:4, background:'#1e2d45', overflow:'hidden' }}>
      <div style={{
        width:`${value * 100}%`, height:'100%',
        background:color, borderRadius:4, transition:'width .6s',
      }} />
    </div>
  )
}

export default function MRIResult({ data, onHome }) {
  if (!data) return null

  const conf   = data.confidence ?? 0
  const confPct = data.confidence_pct ?? Math.round(conf * 100)
  const isDemo  = !!data.demo_warning

  return (
    <div style={{
      minHeight:'100vh', background:c.bg, padding:'24px 16px',
      display:'flex', flexDirection:'column', alignItems:'center',
    }}>
      <div style={{ width:'100%', maxWidth:560 }}>

        {/* Header */}
        <div style={{ textAlign:'center', marginBottom:28 }}>
          <div style={{ fontSize:11, color:c.muted, fontWeight:700,
            letterSpacing:'0.1em', textTransform:'uppercase', marginBottom:6 }}>
            SPORTS INJURY MONITOR
          </div>
          <h1 style={{ fontSize:24, fontWeight:800, color:'#fff' }}>
            MRI Analysis Result
          </h1>
          {data.filename && (
            <div style={{ fontSize:12, color:c.muted, marginTop:4 }}>
              File: {data.filename}
            </div>
          )}
        </div>

        {/* Demo warning banner */}
        {isDemo && (
          <div style={{
            padding:'12px 16px', borderRadius:10, marginBottom:16,
            background:`${c.yellow}15`, border:`2px solid ${c.yellow}60`,
          }}>
            <div style={{ fontSize:13, fontWeight:700, color:c.yellow,
              marginBottom:4 }}>
              ⚠ DEMO MODEL — Results carry no clinical meaning
            </div>
            <p style={{ fontSize:11, color:`${c.yellow}cc`, lineHeight:1.5 }}>
              {data.demo_warning}
            </p>
          </div>
        )}

        {/* Result card */}
        <div style={{ ...card, marginBottom:16, textAlign:'center' }}>
          <div style={{ fontSize:12, fontWeight:700, letterSpacing:'0.08em',
            textTransform:'uppercase', color:c.muted, marginBottom:16 }}>
            AI MODEL RESULT
          </div>

          {/* Prediction badge */}
          <div style={{
            display:'inline-block', padding:'12px 32px',
            borderRadius:12, marginBottom:20,
            background: data.prediction === 'Normal' ? `${c.green}20` : `${c.red}20`,
            border:`2px solid ${data.prediction === 'Normal' ? c.green : c.red}60`,
          }}>
            <div style={{ fontSize:28, fontWeight:900,
              color: data.prediction === 'Normal' ? c.green : c.red }}>
              {data.prediction}
            </div>
            <div style={{ fontSize:11, color:c.muted, marginTop:4 }}>
              AI Classification
            </div>
          </div>

          {/* Confidence */}
          <div style={{ marginBottom:16 }}>
            <div style={{ display:'flex', justifyContent:'space-between',
              fontSize:13, marginBottom:6 }}>
              <span style={{ color:c.muted }}>Model Confidence</span>
              <span style={{ fontWeight:700,
                color: conf >= 0.75 ? c.green : conf >= 0.5 ? c.yellow : c.red }}>
                {confPct}%
              </span>
            </div>
            <ConfBar value={conf} />
          </div>

          {/* Top-2 predictions */}
          {data.top_2 && data.top_2.length > 1 && (
            <div style={{ textAlign:'left' }}>
              <div style={{ fontSize:11, color:c.muted, fontWeight:600,
                letterSpacing:'0.07em', textTransform:'uppercase', marginBottom:8 }}>
                Class Probabilities
              </div>
              {data.top_2.map(([cls, pct]) => (
                <div key={cls} style={{ marginBottom:8 }}>
                  <div style={{ display:'flex', justifyContent:'space-between',
                    fontSize:12, marginBottom:4 }}>
                    <span style={{ color:'#fff' }}>{cls}</span>
                    <span style={{ color:c.muted }}>{pct.toFixed(1)}%</span>
                  </div>
                  <div style={{ height:6, borderRadius:3,
                    background:'#1e2d45', overflow:'hidden' }}>
                    <div style={{
                      width:`${pct}%`, height:'100%',
                      background: cls === data.prediction ? c.blue : c.muted,
                      borderRadius:3, transition:'width .5s',
                    }} />
                  </div>
                </div>
              ))}
            </div>
          )}

          <div style={{ fontSize:11, color:c.muted, marginTop:12,
            borderTop:`1px solid ${c.border}`, paddingTop:10 }}>
            Model: {data.model_name || 'Unknown'}
          </div>
        </div>

        {/* Fusion payload */}
        <div style={{
          ...card, marginBottom:16,
          background:`${c.blue}08`, border:`1px solid ${c.blue}30`,
        }}>
          <div style={{ fontSize:12, fontWeight:700, letterSpacing:'0.08em',
            textTransform:'uppercase', color:c.blue, marginBottom:8 }}>
            🔗 Risk Fusion Payload
          </div>
          <div style={{
            fontSize:11, fontFamily:'monospace',
            background:'#0a0e1a', borderRadius:8, padding:10,
            color:'#94a3b8', whiteSpace:'pre-wrap',
          }}>
            {JSON.stringify(data.mri_fusion_payload, null, 2)}
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

        <button onClick={onHome}
          style={{ ...btn(), width:'100%', fontSize:15 }}>
          🏠 Back to Home
        </button>
      </div>
    </div>
  )
}
