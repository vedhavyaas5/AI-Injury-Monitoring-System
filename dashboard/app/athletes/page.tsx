"use client";
import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Activity, TrendingUp, User } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function levelColor(risk: number) {
  if (risk >= 0.7)  return "text-red-400 bg-red-500/10 border-red-500/20";
  if (risk >= 0.4)  return "text-yellow-400 bg-yellow-500/10 border-yellow-500/20";
  return                   "text-green-400 bg-green-500/10 border-green-500/20";
}
function levelLabel(risk: number) {
  if (risk >= 0.7) return "HIGH";
  if (risk >= 0.4) return "MODERATE";
  return "LOW";
}

export default function AthletesPage() {
  const [athletes, setAthletes] = useState<any[]>([]);
  useEffect(() => {
    fetch(`${API}/api/athletes`).then(r => r.json()).then(setAthletes).catch(() => {});
    const t = setInterval(() =>
      fetch(`${API}/api/athletes`).then(r => r.json()).then(setAthletes).catch(() => {}), 2000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="space-y-5 max-w-screen-xl mx-auto">
      <div>
        <h1 className="text-xl font-bold text-white">Athletes</h1>
        <p className="text-sm text-slate-500">Live monitoring across all athletes</p>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {athletes.map(a => (
          <Link key={a.id} href={`/athletes/${a.id}`}>
            <div className="glass p-4 hover:border-blue-500/30 transition-all cursor-pointer space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className="w-9 h-9 rounded-full bg-blue-600/20 border border-blue-500/30
                                  flex items-center justify-center">
                    <User size={16} className="text-blue-400" />
                  </div>
                  <div>
                    <div className="font-semibold text-sm text-white">{a.name}</div>
                    <div className="text-[10px] text-slate-500">{a.position}</div>
                  </div>
                </div>
                <span className={`text-[10px] px-2 py-0.5 rounded border
                  ${a.status === "active"
                    ? "bg-green-500/10 border-green-500/20 text-green-400"
                    : "bg-slate-500/10 border-slate-500/20 text-slate-400"}`}>
                  {a.status === "active" ? "● Active" : "● Resting"}
                </span>
              </div>

              <div className="grid grid-cols-3 gap-2">
                <div className="text-center">
                  <div className="text-[9px] text-slate-500 uppercase">Risk</div>
                  <div className={`text-sm font-bold ${a.current_risk >= 0.7 ? "text-red-400" : a.current_risk >= 0.4 ? "text-yellow-400" : "text-green-400"}`}>
                    {(a.current_risk * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-[9px] text-slate-500 uppercase">Fatigue</div>
                  <div className="text-sm font-bold text-white">
                    {(a.current_fatigue * 100).toFixed(0)}%
                  </div>
                </div>
                <div className="text-center">
                  <div className="text-[9px] text-slate-500 uppercase">Load</div>
                  <div className="text-sm font-bold text-white">
                    {levelLabel(a.current_load)}
                  </div>
                </div>
              </div>

              <div className={`text-[10px] px-2 py-1 rounded border text-center font-medium
                ${levelColor(a.current_risk)}`}>
                {levelLabel(a.current_risk)} RISK
              </div>
            </div>
          </Link>
        ))}
      </div>
    </div>
  );
}
