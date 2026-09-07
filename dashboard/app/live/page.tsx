"use client";
import React from "react";
import { useLive } from "@/components/LiveDataProvider";
import { JointAnalysis } from "@/components/JointAnalysis";
import { RiskGauge } from "@/components/RiskGauge";
import { PerformanceMonitor } from "@/components/PerformanceMonitor";
import { Camera, Radio } from "lucide-react";

export default function LivePage() {
  const { frame, connected, demo } = useLive();
  const j = frame?.joints;
  const p = frame?.performance;

  return (
    <div className="space-y-5 max-w-screen-2xl mx-auto">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div>
          <h1 className="text-xl font-bold text-white">Live Movement Analysis</h1>
          <p className="text-sm text-slate-500">
            Real-time pose estimation and biomechanical risk assessment
          </p>
        </div>
        <div className="flex items-center gap-2">
          {demo && <span className="demo-badge">DEMO MODE</span>}
          <span className={`flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-lg border
            ${connected ? "bg-green-500/10 border-green-500/30 text-green-400"
                        : "bg-red-500/10 border-red-500/30 text-red-400"}`}>
            <Radio size={11} />
            {connected ? "LIVE" : "DISCONNECTED"}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">

        {/* Camera panel */}
        <div className="lg:col-span-2 glass p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase">
              Live Camera Feed
            </span>
            <div className="flex items-center gap-2 text-xs">
              <span className="text-slate-500">FPS</span>
              <span className="text-white font-mono">{p?.fps?.toFixed(1) ?? "—"}</span>
              <span className="text-slate-500 ml-2">Latency</span>
              <span className="text-white font-mono">{p?.e2e_latency_ms?.toFixed(0) ?? "—"} ms</span>
            </div>
          </div>

          {/* Camera placeholder */}
          <div className="relative rounded-xl overflow-hidden bg-black/40"
               style={{ aspectRatio: "16/9" }}>
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-4">
              <Camera size={48} className="text-slate-700" />
              <div className="text-center">
                <p className="text-slate-500 text-sm font-medium">Camera not connected</p>
                <p className="text-slate-600 text-xs mt-1">
                  Start Phase 7 pipeline to stream pose data
                </p>
                <p className="text-slate-700 text-xs mt-0.5">
                  python phase7_computer_vision.py
                </p>
              </div>
              {demo && (
                <div className="demo-badge text-center">
                  DEMO MODE — simulated joint data displayed below
                </div>
              )}
            </div>

            {/* Overlay stats when running */}
            <div className="absolute bottom-3 left-3 right-3 flex gap-2 flex-wrap">
              {[
                { label: "L Knee", val: j?.l_knee },
                { label: "R Knee", val: j?.r_knee },
                { label: "L Hip",  val: j?.l_hip  },
                { label: "R Hip",  val: j?.r_hip  },
              ].map(({ label, val }) => val !== undefined ? (
                <div key={label}
                  className="bg-black/70 rounded px-2 py-1 text-[10px] text-white border border-white/10">
                  {label}: <span className="font-mono text-blue-300">{val.toFixed(1)}°</span>
                </div>
              ) : null)}
            </div>
          </div>

          {/* Movement metrics */}
          <div className="grid grid-cols-3 gap-2">
            {[
              { label: "Asymmetry",   val: `${(j?.movement_asymmetry ?? 0).toFixed(1)}°`,
                flag: (j?.movement_asymmetry ?? 0) > 20 },
              { label: "Post. Instability", val: `${((j?.postural_instability_index ?? 0)*100).toFixed(1)}%`,
                flag: (j?.postural_instability_index ?? 0) > 0.6 },
              { label: "Ang. Velocity", val: `${(j?.angular_velocity ?? 0).toFixed(2)} °/s`,
                flag: false },
            ].map(({ label, val, flag }) => (
              <div key={label} className="p-2 rounded bg-white/5 text-center">
                <div className="text-[9px] text-slate-500 uppercase mb-1">{label}</div>
                <div className={`text-sm font-bold font-mono ${flag ? "text-yellow-400" : "text-white"}`}>
                  {val}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Right column */}
        <div className="space-y-4">
          <RiskGauge
            score={frame?.risk.overall_risk_pct ?? 0}
            level={frame?.risk.risk_level ?? "LOW"}
            movementRisk={frame?.risk.movement_risk ?? 0}
            sensorRisk={frame?.risk.sensor_risk ?? 0}
            visionRisk={frame?.risk.vision_risk ?? 0} />
          <JointAnalysis />
          <PerformanceMonitor />
        </div>
      </div>
    </div>
  );
}
