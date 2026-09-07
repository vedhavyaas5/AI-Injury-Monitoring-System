"use client";
import React from "react";

interface Props {
  score: number;       // 0–100
  level: string;
  movementRisk: number;
  sensorRisk:   number;
  visionRisk:   number;
  updatedAgo?:  string;
}

function levelColor(level: string) {
  if (level === "HIGH")     return { stroke: "#ef4444", text: "text-red-400",    bg: "bg-red-500/10"    };
  if (level === "MODERATE") return { stroke: "#f59e0b", text: "text-yellow-400", bg: "bg-yellow-500/10" };
  return                           { stroke: "#10b981", text: "text-green-400",  bg: "bg-green-500/10"  };
}

function RiskBar({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs">
        <span className="text-slate-400">{label}</span>
        <span className="text-white font-medium">{Math.round(value * 100)}%</span>
      </div>
      <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700"
             style={{ width: `${Math.min(value * 100, 100)}%`, background: color }} />
      </div>
    </div>
  );
}

export function RiskGauge({ score, level, movementRisk, sensorRisk, visionRisk, updatedAgo }: Props) {
  const { stroke, text, bg } = levelColor(level);

  // SVG arc gauge
  const r = 54, cx = 70, cy = 70;
  const startAngle = -210, totalAngle = 240;
  const angle = startAngle + (score / 100) * totalAngle;
  const toRad = (d: number) => (d * Math.PI) / 180;
  const arcPath = (a1: number, a2: number) => {
    const x1 = cx + r * Math.cos(toRad(a1));
    const y1 = cy + r * Math.sin(toRad(a1));
    const x2 = cx + r * Math.cos(toRad(a2));
    const y2 = cy + r * Math.sin(toRad(a2));
    const large = Math.abs(a2 - a1) > 180 ? 1 : 0;
    return `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2}`;
  };

  return (
    <div className="glass p-5 flex flex-col gap-4">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase">
          Overall Injury Risk
        </span>
        <span className="text-[10px] text-slate-600">
          {updatedAgo ?? "Live"}
        </span>
      </div>

      <div className="flex items-center gap-6">
        {/* Gauge */}
        <div className="relative shrink-0">
          <svg width="140" height="110" viewBox="0 0 140 110">
            {/* Track */}
            <path d={arcPath(-210, 30)} fill="none"
              stroke="rgba(255,255,255,0.06)" strokeWidth="8"
              strokeLinecap="round" />
            {/* Fill */}
            <path d={arcPath(-210, startAngle + (score / 100) * totalAngle)}
              fill="none" stroke={stroke} strokeWidth="8"
              strokeLinecap="round"
              style={{ transition: "all 0.8s ease" }} />
            {/* Score */}
            <text x={cx} y={cy + 6} textAnchor="middle"
              className="font-bold" fontSize="22"
              fill="white">{Math.round(score)}</text>
            <text x={cx} y={cy + 22} textAnchor="middle"
              fontSize="9" fill="#94a3b8">RISK SCORE</text>
          </svg>
        </div>

        {/* Level + breakdown */}
        <div className="flex-1 space-y-3">
          <div className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-lg ${bg}`}>
            <span className={`w-2 h-2 rounded-full ${stroke === "#ef4444" ? "bg-red-400" : stroke === "#f59e0b" ? "bg-yellow-400" : "bg-green-400"}`} />
            <span className={`font-bold text-sm ${text}`}>{level}</span>
          </div>
          <RiskBar label="Movement" value={movementRisk} color="#3b82f6" />
          <RiskBar label="Sensor"   value={sensorRisk}   color="#8b5cf6" />
          <RiskBar label="Vision"   value={visionRisk}   color="#06b6d4" />
        </div>
      </div>

      <p className="text-[10px] text-slate-600 border-t pt-2"
         style={{ borderColor: "var(--border)" }}>
        ⚠ Elevated risk indicators — not a medical diagnosis. Consider professional assessment.
      </p>
    </div>
  );
}
