import React from 'react'
import { c, card } from '../styles'

export default function LandingPage({ onAssess, onMRI }) {
  return (
    <div style={{
      minHeight: '100vh', display: 'flex', flexDirection: 'column',
      alignItems: 'center', justifyContent: 'center',
      background: c.bg, padding: 20,
    }}>
      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: 48 }}>
        <div style={{
          display: 'inline-flex', alignItems: 'center', gap: 10,
          marginBottom: 12,
        }}>
          <div style={{
            width: 40, height: 40, borderRadius: 10,
            background: c.blue, display: 'flex',
            alignItems: 'center', justifyContent: 'center',
            fontSize: 20,
          }}>🏃</div>
          <span style={{ fontSize: 13, fontWeight: 700,
            letterSpacing: '0.15em', color: c.muted }}>
            SPORTS INJURY MONITOR
          </span>
        </div>
        <h1 style={{ fontSize: 32, fontWeight: 800, color: '#fff',
          marginBottom: 8 }}>
          How are you feeling today?
        </h1>
        <p style={{ color: c.muted, fontSize: 15 }}>
          Select an option to begin your assessment
        </p>
      </div>

      {/* Choice cards */}
      <div style={{
        display: 'flex', gap: 20, flexWrap: 'wrap',
        justifyContent: 'center', maxWidth: 700, width: '100%',
      }}>
        {/* Self Assessment */}
        <button onClick={onAssess} style={{
          ...card,
          flex: '1 1 280px', cursor: 'pointer',
          border: `1px solid ${c.border}`,
          textAlign: 'center', transition: 'border-color .2s',
          background: c.card,
        }}
          onMouseEnter={e => e.currentTarget.style.borderColor = c.blue}
          onMouseLeave={e => e.currentTarget.style.borderColor = c.border}
        >
          <div style={{ fontSize: 48, marginBottom: 16 }}>📝</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#fff',
            marginBottom: 8 }}>
            Self Assessment
          </div>
          <div style={{ color: c.muted, fontSize: 14, lineHeight: 1.6 }}>
            Report pain, fatigue, symptoms and describe how you feel.
            Generates a structured risk-indicator report.
          </div>
          <div style={{
            marginTop: 20, padding: '10px 20px',
            background: `${c.blue}22`, border: `1px solid ${c.blue}55`,
            borderRadius: 8, color: c.blue, fontSize: 13, fontWeight: 600,
          }}>
            Start Assessment →
          </div>
        </button>

        {/* MRI Upload */}
        <button onClick={onMRI} style={{
          ...card,
          flex: '1 1 280px', cursor: 'pointer',
          border: `1px solid ${c.border}`,
          textAlign: 'center', transition: 'border-color .2s',
          background: c.card,
        }}
          onMouseEnter={e => e.currentTarget.style.borderColor = c.purple}
          onMouseLeave={e => e.currentTarget.style.borderColor = c.border}
        >
          <div style={{ fontSize: 48, marginBottom: 16 }}>🩻</div>
          <div style={{ fontSize: 18, fontWeight: 700, color: '#fff',
            marginBottom: 8 }}>
            Upload MRI Scan
          </div>
          <div style={{ color: c.muted, fontSize: 14, lineHeight: 1.6 }}>
            Upload a medical image for AI-assisted classification.
            Supports JPG, PNG, WEBP formats.
          </div>
          <div style={{
            marginTop: 20, padding: '10px 20px',
            background: `${c.purple}22`, border: `1px solid ${c.purple}55`,
            borderRadius: 8, color: c.purple, fontSize: 13, fontWeight: 600,
          }}>
            Upload Image →
          </div>
        </button>
      </div>

      {/* Disclaimer */}
      <div style={{
        marginTop: 40, maxWidth: 580, textAlign: 'center',
        padding: '12px 20px',
        background: `${c.yellow}11`,
        border: `1px solid ${c.yellow}33`,
        borderRadius: 10,
      }}>
        <p style={{ fontSize: 11, color: `${c.yellow}cc`, lineHeight: 1.6 }}>
          ⚠ This system provides AI-based risk indicators for monitoring and decision support.
          It does not provide a medical diagnosis or replace professional assessment.
        </p>
      </div>
    </div>
  )
}
