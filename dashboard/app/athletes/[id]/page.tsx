"use client";
import React, { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import { RiskGauge } from "@/components/RiskGauge";
import { RiskChart } from "@/components/RiskChart";
import { AlertCard } from "@/components/AlertCard";
import { useLive } from "@/components/LiveDataProvider";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function AthleteProfile() {
  const { id } = useParams<{ id: string }>();
  const [athlete, setAthlete] = useState<any>(null);
  const [sessions, setSessions] = useState<any[]>([]);
  const [alerts,   setAlerts]   = useState<any[]>([]);
  const { frame } = useLive();

  useEffect(() => {
    fetch(`${API}/api/athletes/${id}`).then(r=>r.json()).then(setAthlete).catch(()=>{});
    fetch(`${API}/api/sessions?athlete_id=${id}`).then(r=>r.json()).then(setSessions).catch(()=>{});
    fetch(`${API}/api/alerts?athlete_id=${id}&last_n=10`).then(r=>r.json()).then(setAlerts).catch(()=>{});
  }, [id]);

  const r = frame?.risk;

  return (
    <div className="space-y-5 max-w-screen-xl mx-auto">
      {/* Header */}
      <div className="glass p-5 flex flex-wrap gap-6 items-center">
        <div className="w-14 h-14 rounded-full bg-blue-600/20 border border-blue-500/30
                        flex items-center justify-center text-xl font-bold text-blue-400">
          {id}
        </div>
        <div className="flex-1 min-w-0">
          <h1 className="text-xl font-bold text-white">{athlete?.name ?? `Athlete #${id}`}</h1>
          <div className="flex gap-3 mt-1 flex-wrap text-xs text-slate-400">
            <span>{athlete?.position}</span>
            <span>Age {athlete?.age}</span>
            <span className={athlete?.status === "active" ? "text-green-400" : "text-slate-500"}>
              ● {athlete?.status}
            </span>
          </div>
        </div>
        <div className="text-center">
          <div className="text-2xl font-bold text-white">
            {((athlete?.current_risk ?? 0)*100).toFixed(0)}%
          </div>
          <div className="text-[10px] text-slate-500">Current Risk</div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="space-y-4">
          <RiskGauge
            score={r?.overall_risk_pct ?? (athlete?.current_risk ?? 0)*100}
            level={r?.risk_level ?? "LOW"}
            movementRisk={r?.movement_risk ?? 0}
            sensorRisk={r?.sensor_risk ?? 0}
            visionRisk={r?.vision_risk ?? 0} />

          {/* Sessions */}
          <div className="glass p-4 space-y-2">
            <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
              Recent Sessions
            </span>
            {sessions.slice(0,4).map(s => (
              <div key={s.id} className="flex justify-between items-center py-2 border-b last:border-0
                                         text-xs" style={{ borderColor: "var(--border)" }}>
                <div>
                  <div className="text-white">{s.type}</div>
                  <div className="text-slate-500">{s.duration_min.toFixed(0)} min</div>
                </div>
                <div className="text-right">
                  <div className={`font-bold ${s.peak_risk >= 0.7 ? "text-red-400" : s.peak_risk >= 0.4 ? "text-yellow-400" : "text-green-400"}`}>
                    Peak {(s.peak_risk*100).toFixed(0)}%
                  </div>
                  <div className="text-slate-500">{s.alerts_count} alerts</div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="space-y-4">
          <RiskChart />
        </div>

        <div className="space-y-3">
          <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
            Alerts
          </span>
          {alerts.length === 0
            ? <div className="glass p-4 text-xs text-slate-500 text-center">No alerts</div>
            : alerts.map(a => <AlertCard key={a.id} alert={a} />)}
        </div>
      </div>
    </div>
  );
}
