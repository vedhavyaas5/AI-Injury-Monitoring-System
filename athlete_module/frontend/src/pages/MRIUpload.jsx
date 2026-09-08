import React, { useState, useRef } from 'react'
import { c, card, btn, btnOutline } from '../styles'

const API = '/api/mri'
const ACCEPTED = '.jpg,.jpeg,.png,.webp,.bmp,.tiff'
const MAX_MB   = 10

export default function MRIUpload({ onBack, onDone }) {
  const [file,      setFile]     = useState(null)
  const [preview,   setPreview]  = useState(null)
  const [athleteId, setAthleteId]= useState('')
  const [drag,      setDrag]     = useState(false)
  const [loading,   setLoading]  = useState(false)
  const [error,     setError]    = useState('')
  const inputRef = useRef()

  const validate = (f) => {
    if (!f) return 'No file selected.'
    const ext = f.name.split('.').pop().toLowerCase()
    if (!['jpg','jpeg','png','webp','bmp','tiff'].includes(ext))
      return `Unsupported format ".${ext}". Use JPG, JPEG, PNG, WEBP, BMP or TIFF.`
    if (f.size > MAX_MB * 1024 * 1024)
      return `File too large (${(f.size/1024/1024).toFixed(1)} MB). Max: ${MAX_MB} MB.`
    if (f.size === 0)
      return 'File is empty.'
    return null
  }

  const pickFile = (f) => {
    const err = validate(f)
    if (err) { setError(err); return }
    setError('')
    setFile(f)
    const reader = new FileReader()
    reader.onload = e => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }

  const onDrop = (e) => {
    e.preventDefault(); setDrag(false)
    const f = e.dataTransfer.files[0]
    if (f) pickFile(f)
  }

  const analyze = async () => {
    if (!file) { setError('Please upload an MRI image first.'); return }
    setError(''); setLoading(true)
    try {
      const form = new FormData()
      form.append('file',       file)
      form.append('athlete_id', athleteId.trim() || 'unknown')
      const res = await fetch(`${API}/analyze`, { method:'POST', body:form })
      if (!res.ok) {
        const e = await res.json()
        throw new Error(e.detail || `Server error ${res.status}`)
      }
      onDone(await res.json())
    } catch(e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  const borderColor = drag ? c.purple : error ? c.red : c.border

  return (
    <div style={{
      minHeight:'100vh', background:c.bg, padding:'24px 16px',
      display:'flex', flexDirection:'column', alignItems:'center',
    }}>
      <div style={{ width:'100%', maxWidth:560 }}>

        {/* Header */}
        <div style={{ display:'flex', alignItems:'center', gap:12,
          marginBottom:28 }}>
          <button onClick={onBack}
            style={{ ...btnOutline, padding:'8px 14px', fontSize:13 }}>
            ← Back
          </button>
          <div>
            <div style={{ fontSize:11, color:c.muted, fontWeight:700,
              letterSpacing:'0.1em', textTransform:'uppercase' }}>
              SPORTS INJURY MONITOR
            </div>
            <h1 style={{ fontSize:22, fontWeight:800, color:'#fff' }}>
              MRI Analysis
            </h1>
          </div>
        </div>

        <div style={{ ...card }}>
          {/* Athlete ID */}
          <div style={{ marginBottom:20 }}>
            <label style={{ fontSize:12, fontWeight:600, letterSpacing:'0.07em',
              textTransform:'uppercase', color:c.muted, display:'block',
              marginBottom:6 }}>
              Athlete ID (optional)
            </label>
            <input placeholder="e.g. 07" value={athleteId}
              onChange={e => setAthleteId(e.target.value)}
              style={{
                background:'#0d1624', border:`1px solid ${c.border}`,
                borderRadius:10, padding:'10px 14px', color:c.text,
                fontSize:14, width:'100%', outline:'none',
              }} />
          </div>

          {/* Drop zone */}
          <div
            onDragOver={e => { e.preventDefault(); setDrag(true) }}
            onDragLeave={()=> setDrag(false)}
            onDrop={onDrop}
            onClick={()=> inputRef.current.click()}
            style={{
              border:`2px dashed ${borderColor}`,
              borderRadius:14, padding:'36px 20px',
              textAlign:'center', cursor:'pointer',
              background: drag ? `${c.purple}10` : '#0d1624',
              transition:'all .2s',
            }}
          >
            {preview ? (
              <div>
                <img src={preview} alt="MRI preview"
                  style={{ maxHeight:220, maxWidth:'100%',
                    borderRadius:8, marginBottom:12 }} />
                <div style={{ fontSize:13, color:c.blue }}>
                  {file.name} — {(file.size/1024).toFixed(0)} KB
                </div>
                <div style={{ fontSize:11, color:c.muted, marginTop:4 }}>
                  Click to change
                </div>
              </div>
            ) : (
              <>
                <div style={{ fontSize:48, marginBottom:12 }}>🩻</div>
                <div style={{ fontSize:16, fontWeight:600, color:'#fff',
                  marginBottom:6 }}>
                  Drag &amp; Drop MRI Image
                </div>
                <div style={{ fontSize:13, color:c.muted, marginBottom:12 }}>
                  or click to browse
                </div>
                <div style={{
                  fontSize:11, color:c.muted, padding:'6px 14px',
                  border:`1px solid ${c.border}`, borderRadius:6,
                  display:'inline-block',
                }}>
                  JPG / JPEG / PNG / WEBP
                </div>
              </>
            )}
          </div>
          <input ref={inputRef} type="file" accept={ACCEPTED}
            style={{ display:'none' }}
            onChange={e => { if (e.target.files[0]) pickFile(e.target.files[0]) }} />

          {/* File size note */}
          <p style={{ fontSize:11, color:c.muted, marginTop:8,
            textAlign:'center' }}>
            Max file size: {MAX_MB} MB. Supported: JPG, JPEG, PNG, WEBP, BMP, TIFF
          </p>

          {/* Error */}
          {error && (
            <div style={{
              marginTop:14, padding:'10px 14px', borderRadius:8,
              background:`${c.red}15`, border:`1px solid ${c.red}40`,
              color:c.red, fontSize:13,
            }}>
              ⚠ {error}
            </div>
          )}

          {/* Analyze button */}
          <button onClick={analyze} disabled={loading || !file}
            style={{
              ...btn(c.purple), width:'100%', fontSize:15,
              marginTop:20, opacity: loading||!file ? 0.5 : 1,
              cursor: loading||!file ? 'not-allowed' : 'pointer',
            }}>
            {loading ? 'Analysing…' : '🔍 Analyze MRI'}
          </button>
        </div>

        {/* Disclaimer */}
        <div style={{
          marginTop:20, padding:'12px 16px',
          background:`${c.yellow}10`, border:`1px solid ${c.yellow}30`,
          borderRadius:10,
        }}>
          <p style={{ fontSize:11, color:`${c.yellow}cc`, lineHeight:1.6 }}>
            ⚠ The MRI analysis module provides AI-based image classification output only.
            It is NOT a medical diagnosis. Results must be reviewed by a qualified
            medical professional before any clinical decision is made.
          </p>
        </div>
      </div>
    </div>
  )
}
