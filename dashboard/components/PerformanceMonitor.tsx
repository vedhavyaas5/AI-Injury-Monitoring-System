"use client";
import React from "react";
import { useLive } from "./LiveDataProvider";
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from "recharts";

export function PerformanceMonitor() {
  const { history } = useLive();

  const recent = history.slice(-30);
  const latencies = recent.map(f => f.performance?.e2e_latency_ms ?? 0).filter(v => v > 0);
  const fps_vals  = recent.map(f => f.performance?.fps ?? 0).filter(v => v > 0);

  const avgLat = latencies.length ? latencies.reduce((a,b) => a+b,0)/latencies.length : 0;
  const p95Lat = latencies.length >= 5
    ? latencies.sort((a,b)=>a-b)[Math.floor(latencies.length*0.95)] : avgLat;
  const avgFps = fps_vals.length ? fps_vals.reduce((a,b)=>a+b,0)/fps_vals.length : 0;

  const chartData = recent.map((f, i) => ({
    t: i,
    e2e:   f.performance?.e2e_latency_ms   ?? 0,
    pose:  f.performance?.pose_latency_ms  ?? 0,
    ml:    f.performance?.ml_latency_ms    ?? 0,
    fuse:  f.performance?.fusion_latency_ms?? 0,
  }));

  const frame = history[history.length - 1];
  const p = frame?.performance;

  return (
    <div className="glass p-4 space-y-4">
      <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
        System Performance
      </span>

      {/* KPI row */}
      <div className="grid grid-cols-4 gap-2">
        {[
          { label: "FPS",         value: avgFps.toFixed(1),   unit: ""    },
          { label: "Avg Latency", value: avgLat.toFixed(1),   unit: "ms"  },
          { label: "P95 Latency", value: p95Lat.toFixed(1),   unit: "ms"  },
          { label: "E2E",         value: (p?.e2e_latency_ms ?? 0).toFixed(1), unit: "ms" },
        ].map(({ label, value, unit }) => (
          <div key={label} className="p-2 rounded bg-white/5 text-center">
            <div className="text-[9px] text-slate-500 uppercase">{label}</div>
            <div className="text-base font-bold text-white">{value}
              <span className="text-[10px] text-slate-500 ml-0.5">{unit}</span>
            </div>
          </div>
        ))}
      </div>

      {/* Pipeline breakdown */}
      <div className="space-y-1.5 text-xs">
        {[
          { label: "Pose Detection", val: p?.pose_latency_ms  ?? 0, color: "#3b82f6" },
          { label: "ML Inference",   val: p?.ml_latency_ms    ?? 0, color: "#8b5cf6" },
          { label: "Risk Fusion",    val: p?.fusion_latency_ms?? 0, color: "#10b981" },
          { label: "End-to-End",     val: p?.e2e_latency_ms   ?? 0, color: "#f59e0b" },
        ].map(({ label, val, color }) => (
          <div key={label} className="flex items-center justify-between">
            <span className="text-slate-500 w-28">{label}</span>
            <div className="flex-1 mx-3 h-1.5 rounded-full bg-white/5 overflow-hidden">
              <div className="h-full rounded-full transition-all duration-500"
                   style={{ width: `${Math.min((val/100)*100,100)}%`, background: color }} />
            </div>
            <span className="text-white font-mono w-12 text-right">{val.toFixed(1)} ms</span>
          </div>
        ))}
      </div>

      {/* Latency chart */}
      <ResponsiveContainer width="100%" height={100}>
        <BarChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}
          barSize={6} barGap={1}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis dataKey="t" tick={false} axisLine={false} />
          <YAxis tick={{ fontSize: 9, fill: "#475569" }} axisLine={false} tickLine={false} />
          <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1e2d45",
            borderRadius: 6, fontSize: 10 }}
            formatter={(v, n) => [`${Number(v ?? 0).toFixed(1)}ms`, String(n)]} />
          <Bar dataKey="pose" fill="#3b82f6" stackId="a" />
          <Bar dataKey="ml"   fill="#8b5cf6" stackId="a" />
          <Bar dataKey="fuse" fill="#10b981" stackId="a" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
