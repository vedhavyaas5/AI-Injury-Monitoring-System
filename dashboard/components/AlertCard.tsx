"use client";
import React from "react";
import { AlertTriangle, Info, CheckCircle, XCircle } from "lucide-react";

interface Props {
  alert: {
    id: string; athlete_id: string; level: string;
    risk_score: number; reasons: string[]; timestamp: number;
    acknowledged: boolean; demo?: boolean;
  };
  onAck?: (id: string) => void;
}

const LEVEL_CFG = {
  HIGH:     { bg: "bg-red-500/10",    border: "border-red-500/30",    text: "text-red-400",    icon: XCircle        },
  CRITICAL: { bg: "bg-red-600/15",    border: "border-red-600/40",    text: "text-red-300",    icon: AlertTriangle  },
  MODERATE: { bg: "bg-yellow-500/10", border: "border-yellow-500/30", text: "text-yellow-400", icon: AlertTriangle  },
  LOW:      { bg: "bg-blue-500/10",   border: "border-blue-500/20",   text: "text-blue-400",   icon: Info           },
  INFO:     { bg: "bg-blue-500/10",   border: "border-blue-500/20",   text: "text-blue-400",   icon: Info           },
};

function timeAgo(ts: number) {
  const diff = Math.floor((Date.now() / 1000) - ts);
  if (diff < 60)  return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

export function AlertCard({ alert, onAck }: Props) {
  const cfg = LEVEL_CFG[alert.level as keyof typeof LEVEL_CFG] ?? LEVEL_CFG.INFO;
  const Icon = cfg.icon;

  return (
    <div className={`rounded-xl border p-4 space-y-3 transition-all
      ${cfg.bg} ${cfg.border} ${alert.acknowledged ? "opacity-50" : ""}`}>

      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <Icon size={16} className={cfg.text} />
          <span className={`text-sm font-bold ${cfg.text}`}>{alert.level} RISK</span>
          {alert.demo && <span className="demo-badge">DEMO</span>}
        </div>
        <span className="text-[10px] text-slate-500 shrink-0">{timeAgo(alert.timestamp)}</span>
      </div>

      <p className="text-xs text-slate-300">
        Elevated injury-risk indicators detected — Athlete #{alert.athlete_id}
      </p>

      <div className="flex items-center gap-3 text-xs">
        <span className="text-slate-500">Risk Score</span>
        <span className={`font-bold ${cfg.text}`}>{alert.risk_score.toFixed(1)}%</span>
      </div>

      {alert.reasons.length > 0 && (
        <ul className="space-y-0.5">
          {alert.reasons.map((r, i) => (
            <li key={i} className="text-xs text-slate-400 flex gap-1.5">
              <span className={cfg.text}>•</span>{r}
            </li>
          ))}
        </ul>
      )}

      <p className="text-[10px] text-slate-600 italic">
        These are possible contributing indicators. This is not a medical diagnosis.
        Consider professional assessment.
      </p>

      {!alert.acknowledged && onAck && (
        <button onClick={() => onAck(alert.id)}
          className="text-xs px-3 py-1.5 rounded-lg bg-white/5
                     border border-white/10 text-slate-300 hover:bg-white/10 transition">
          Acknowledge
        </button>
      )}
      {alert.acknowledged && (
        <div className="flex items-center gap-1 text-xs text-green-500/60">
          <CheckCircle size={11} /> Acknowledged
        </div>
      )}
    </div>
  );
}
