"use client";
import React from "react";
import { useLive } from "@/components/LiveDataProvider";
import {
  LineChart, Line, BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from "recharts";

export default function AnalyticsPage() {
  const { history } = useLive();

  const data = history.map((f, i) => ({
    t: i,
    overall:  Math.round((f.risk?.smoothed_risk ?? 0)*100),
    movement: Math.round((f.risk?.movement_risk ?? 0)*100),
    sensor:   Math.round((f.risk?.sensor_risk   ?? 0)*100),
    vision:   Math.round((f.risk?.vision_risk   ?? 0)*100),
    fatigue:  Math.round((f.sensors?.fatigue_level ?? 0)*100),
    load:     Math.round(f.sensors?.training_load ?? 0),
    hr:       Math.round(f.sensors?.heart_rate_bpm ?? 0),
    asymm:    +(f.joints?.movement_asymmetry ?? 0).toFixed(1),
  }));

  const ChartCard = ({ title, children }: { title: string; children: React.ReactNode }) => (
    <div className="glass p-4 space-y-3">
      <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
        {title}
      </span>
      {children}
    </div>
  );

  const TT = {
    contentStyle: { background: "#111827", border: "1px solid #1e2d45",
      borderRadius: 8, fontSize: 11 },
    labelStyle: { display: "none" as const },
  };

  return (
    <div className="space-y-5 max-w-screen-2xl mx-auto">
      <div>
        <h1 className="text-xl font-bold text-white">Analytics</h1>
        <p className="text-sm text-slate-500">Session trends and model comparison</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">

        <ChartCard title="Risk Over Time">
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={data} margin={{ top:4, right:4, bottom:0, left:-20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="t" tick={false} axisLine={false} />
              <YAxis domain={[0,100]} tick={{ fontSize:10, fill:"#475569" }}
                axisLine={false} tickLine={false} />
              <Tooltip {...TT} formatter={(v,n)=>[`${Number(v??0)}%`,String(n)]} />
              <Line type="monotone" dataKey="overall"  stroke="#f59e0b" strokeWidth={2} dot={false} isAnimationActive={false} name="Overall" />
              <Line type="monotone" dataKey="movement" stroke="#3b82f6" strokeWidth={1.5} dot={false} isAnimationActive={false} name="Movement" />
              <Line type="monotone" dataKey="sensor"   stroke="#8b5cf6" strokeWidth={1.5} dot={false} isAnimationActive={false} name="Sensor" />
              <Line type="monotone" dataKey="vision"   stroke="#06b6d4" strokeWidth={1.5} dot={false} isAnimationActive={false} name="Vision" />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Fatigue Trend">
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={data} margin={{ top:4, right:4, bottom:0, left:-20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="t" tick={false} axisLine={false} />
              <YAxis domain={[0,100]} tick={{ fontSize:10, fill:"#475569" }}
                axisLine={false} tickLine={false} />
              <Tooltip {...TT} formatter={(v)=>[`${Number(v??0)}%`,"Fatigue"]} />
              <Line type="monotone" dataKey="fatigue" stroke="#ef4444"
                strokeWidth={2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Training Load">
          <ResponsiveContainer width="100%" height={180}>
            <BarChart data={data.slice(-30)} margin={{ top:4, right:4, bottom:0, left:-20 }} barSize={8}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="t" tick={false} axisLine={false} />
              <YAxis tick={{ fontSize:10, fill:"#475569" }} axisLine={false} tickLine={false} />
              <Tooltip {...TT} formatter={(v)=>[`${Number(v??0)}`,"Load"]} />
              <Bar dataKey="load" fill="#3b82f6" radius={[2,2,0,0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Movement Asymmetry (°)">
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={data} margin={{ top:4, right:4, bottom:0, left:-20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="t" tick={false} axisLine={false} />
              <YAxis tick={{ fontSize:10, fill:"#475569" }} axisLine={false} tickLine={false} />
              <Tooltip {...TT} formatter={(v)=>[`${Number(v??0)}°`,"Asymmetry"]} />
              <Line type="monotone" dataKey="asymm" stroke="#f59e0b"
                strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Heart Rate">
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={data} margin={{ top:4, right:4, bottom:0, left:-20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="t" tick={false} axisLine={false} />
              <YAxis domain={[50,200]} tick={{ fontSize:10, fill:"#475569" }}
                axisLine={false} tickLine={false} />
              <Tooltip {...TT} formatter={(v)=>[`${Number(v??0)} bpm`,"HR"]} />
              <Line type="monotone" dataKey="hr" stroke="#ef4444"
                strokeWidth={1.5} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Model Comparison (current frame)">
          <ResponsiveContainer width="100%" height={180}>
            <BarChart
              data={[
                { model: "Movement", risk: Math.round((history[history.length-1]?.risk?.movement_risk??0)*100) },
                { model: "Sensor",   risk: Math.round((history[history.length-1]?.risk?.sensor_risk??0)*100)   },
                { model: "Vision",   risk: Math.round((history[history.length-1]?.risk?.vision_risk??0)*100)   },
                { model: "Fused",    risk: Math.round((history[history.length-1]?.risk?.overall_risk_pct??0))  },
              ]}
              margin={{ top:4, right:4, bottom:0, left:-20 }} barSize={32}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis dataKey="model" tick={{ fontSize:10, fill:"#94a3b8" }} axisLine={false} />
              <YAxis domain={[0,100]} tick={{ fontSize:10, fill:"#475569" }} axisLine={false} tickLine={false} />
              <Tooltip {...TT} formatter={(v)=>[`${Number(v??0)}%`,"Risk"]} />
              <Bar dataKey="risk" radius={[4,4,0,0]}
                fill="#3b82f6"
                label={{ position:"top", fontSize:10, fill:"#94a3b8",
                  formatter:(v: unknown)=>`${Number(v??0)}%` }} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

      </div>
    </div>
  );
}
