"use client";
import React from "react";
import { useLive } from "@/components/LiveDataProvider";
import { MetricCard } from "@/components/MetricCard";
import { RiskGauge } from "@/components/RiskGauge";
import { RiskChart } from "@/components/RiskChart";
import { JointAnalysis } from "@/components/JointAnalysis";
import { SystemStatus } from "@/components/SystemStatus";
import { AlertCard } from "@/components/AlertCard";
import {
  Heart, Thermometer, Zap, Activity,
  TrendingUp, Droplets, Brain, Eye
} from "lucide-react";

function levelStatus(pct: number): "low" | "moderate" | "high" {
  if (pct >= 70) return "high";
  if (pct >= 40) return "moderate";
  return "low";
}

export default function Dashboard() {
  const { frame, demo, alerts, athleteId } = useLive();
  const s = frame?.sensors;
  const r = frame?.risk;
  const j = frame?.joints;

  const overallPct   = r?.overall_risk_pct ?? 0;
  const movementPct  = (r?.movement_risk   ?? 0) * 100;
  const sensorPct    = (r?.sensor_risk     ?? 0) * 100;
  const visionPct    = (r?.vision_risk     ?? 0) * 100;

  return (
    <div className="space-y-5 max-w-screen-2xl mx-auto">

      {/* Header */}
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-white">Sports Injury Monitoring</h1>
          <p className="text-sm text-slate-500 mt-0.5">
            Real-time athlete risk intelligence &nbsp;·&nbsp;
            Athlete #{athleteId} &nbsp;·&nbsp; Training Session &nbsp;·&nbsp;
            {new Date().toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" })}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {demo && <span className="demo-badge">DEMO MODE</span>}
          <span className={`text-xs px-2 py-1 rounded border
            ${overallPct >= 70
              ? "bg-red-500/10 border-red-500/30 text-red-400"
              : overallPct >= 40
              ? "bg-yellow-500/10 border-yellow-500/30 text-yellow-400"
              : "bg-green-500/10 border-green-500/30 text-green-400"}`}>
            {r?.risk_level ?? "—"}
          </span>
        </div>
      </div>

      {/* KPI cards — row 1 */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <MetricCard title="Heart Rate" icon={<Heart size={14}/>}
          value={s?.heart_rate_bpm?.toFixed(0) ?? "—"} unit="BPM"
          status={s?.heart_rate_bpm && s.heart_rate_bpm > 165 ? "high" : s?.heart_rate_bpm && s.heart_rate_bpm > 140 ? "moderate" : "ok"}
          statusLabel={s?.heart_rate_bpm && s.heart_rate_bpm > 165 ? "Elevated" : "Normal"} />

        <MetricCard title="SpO₂" icon={<Droplets size={14}/>}
          value={s?.oxygen_saturation_spo2?.toFixed(1) ?? "—"} unit="%"
          status={s?.oxygen_saturation_spo2 && s.oxygen_saturation_spo2 < 93 ? "high" : "ok"}
          statusLabel={s?.oxygen_saturation_spo2 && s.oxygen_saturation_spo2 < 93 ? "Low" : "Normal"} />

        <MetricCard title="Fatigue" icon={<Brain size={14}/>}
          value={s?.fatigue_level !== undefined ? `${(s.fatigue_level*100).toFixed(0)}` : "—"} unit="%"
          status={levelStatus((s?.fatigue_level ?? 0)*100)}
          statusLabel={levelStatus((s?.fatigue_level ?? 0)*100) === "high" ? "High" : levelStatus((s?.fatigue_level ?? 0)*100) === "moderate" ? "Moderate" : "Low"} />

        <MetricCard title="Training Load" icon={<Activity size={14}/>}
          value={s?.training_load?.toFixed(0) ?? "—"}
          status={s?.training_load && s.training_load > 380 ? "high" : s?.training_load && s.training_load > 250 ? "moderate" : "low"}
          statusLabel={s?.training_load && s.training_load > 380 ? "High" : s?.training_load && s.training_load > 250 ? "Moderate" : "Low"} />

        <MetricCard title="Skin Temp" icon={<Thermometer size={14}/>}
          value={s?.skin_temperature_celsius?.toFixed(1) ?? "—"} unit="°C"
          status="normal" statusLabel="Normal" />

        <MetricCard title="Hydration" icon={<Droplets size={14}/>}
          value={s?.hydration_level !== undefined ? `${(s.hydration_level*100).toFixed(0)}` : "—"} unit="%"
          status={s?.hydration_level && s.hydration_level < 0.4 ? "high" : "ok"}
          statusLabel={s?.hydration_level && s.hydration_level < 0.4 ? "Low" : "Good"} />
      </div>

      {/* KPI cards — row 2 (model risks) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard title="Movement Risk" icon={<Zap size={14}/>}
          value={movementPct.toFixed(1)} unit="%"
          status={levelStatus(movementPct)} statusLabel={r?.risk_level} />

        <MetricCard title="Sensor Risk" icon={<Activity size={14}/>}
          value={sensorPct.toFixed(1)} unit="%"
          status={levelStatus(sensorPct)} statusLabel={levelStatus(sensorPct)} />

        <MetricCard title="Vision Risk" icon={<Eye size={14}/>}
          value={visionPct.toFixed(1)} unit="%"
          status={levelStatus(visionPct)} statusLabel={levelStatus(visionPct)} />

        <MetricCard title="Postural Stability" icon={<TrendingUp size={14}/>}
          value={j?.postural_instability_index !== undefined
            ? `${(j.postural_instability_index*100).toFixed(1)}` : "—"} unit="%"
          sub="Instability index"
          status={j?.postural_instability_index && j.postural_instability_index > 0.6 ? "high" : "ok"} />
      </div>

      {/* Main 3-column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Col 1 — Gauge + alerts */}
        <div className="space-y-4">
          <RiskGauge
            score={overallPct}
            level={r?.risk_level ?? "LOW"}
            movementRisk={r?.movement_risk ?? 0}
            sensorRisk={r?.sensor_risk ?? 0}
            visionRisk={r?.vision_risk ?? 0} />

          {alerts.length > 0 && (
            <div className="space-y-2">
              <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase">
                Recent Alerts
              </span>
              {alerts.slice(0,2).map(a => (
                <AlertCard key={a.id} alert={a} />
              ))}
            </div>
          )}
        </div>

        {/* Col 2 — Chart */}
        <div className="space-y-4">
          <RiskChart />
          <JointAnalysis />
        </div>

        {/* Col 3 — System */}
        <div className="space-y-4">
          <SystemStatus />

          {/* Fusion weights */}
          <div className="glass p-4 space-y-3">
            <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
              Fusion Weights
            </span>
            {[
              { label: "Movement", val: (r?.weights?.movement ?? 0.35)*100, color: "#3b82f6" },
              { label: "Sensor",   val: (r?.weights?.sensor   ?? 0.30)*100, color: "#8b5cf6" },
              { label: "Vision",   val: (r?.weights?.vision   ?? 0.35)*100, color: "#06b6d4" },
            ].map(({ label, val, color }) => (
              <div key={label} className="space-y-1">
                <div className="flex justify-between text-xs">
                  <span className="text-slate-400">{label}</span>
                  <span className="text-white">{val.toFixed(0)}%</span>
                </div>
                <div className="h-1.5 rounded-full bg-white/5 overflow-hidden">
                  <div className="h-full rounded-full"
                       style={{ width: `${val}%`, background: color }} />
                </div>
              </div>
            ))}
            <p className="text-[10px] text-slate-600 pt-1">
              Configurable engineering weights — not medically validated.
            </p>
          </div>

          {/* Disclaimer */}
          <div className="glass p-3 border border-yellow-500/10 bg-yellow-500/5">
            <p className="text-[10px] text-yellow-300/70 leading-relaxed">
              This system provides AI-based risk indicators for monitoring and decision support.
              It does not provide a medical diagnosis or replace professional assessment.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
