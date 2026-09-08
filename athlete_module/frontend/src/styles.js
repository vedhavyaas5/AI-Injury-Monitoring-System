// Shared style tokens — used inline across all components
export const c = {
  bg:       '#0a0e1a',
  card:     '#111827',
  card2:    '#1a2235',
  border:   '#1e2d45',
  text:     '#e2e8f0',
  muted:    '#6b7fa3',
  blue:     '#3b82f6',
  green:    '#10b981',
  yellow:   '#f59e0b',
  red:      '#ef4444',
  purple:   '#8b5cf6',
}

export const card = {
  background:   c.card,
  border:       `1px solid ${c.border}`,
  borderRadius: 16,
  padding:      24,
}

export const input = {
  background:   '#0d1624',
  border:       `1px solid ${c.border}`,
  borderRadius: 10,
  padding:      '10px 14px',
  color:        c.text,
  fontSize:     14,
  width:        '100%',
  outline:      'none',
}

export const btn = (color = c.blue) => ({
  background:   color,
  border:       'none',
  borderRadius: 10,
  padding:      '12px 28px',
  color:        '#fff',
  fontSize:     15,
  fontWeight:   600,
  cursor:       'pointer',
  transition:   'opacity .2s',
})

export const btnOutline = {
  background:   'transparent',
  border:       `1px solid ${c.border}`,
  borderRadius: 10,
  padding:      '10px 22px',
  color:        c.muted,
  fontSize:     14,
  cursor:       'pointer',
}

export const label = {
  fontSize:   12,
  fontWeight: 600,
  letterSpacing: '0.07em',
  textTransform: 'uppercase',
  color:      c.muted,
  display:    'block',
  marginBottom: 6,
}

export const section = {
  marginBottom: 28,
}
