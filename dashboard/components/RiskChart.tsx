"use client";
import React, { useState } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, ReferenceLine, Legend
} from "recharts";
import { useLive } from "./LiveDataProvider";

const LINES = [
  { key: "overall", label: "Overall", color: "#f59e0b" },
  { key: "movement", label: "Movement", color: "#3b82f6" },
  { key: "sensor",   label: "Sensor",   color: "#8b5cf6" },
  { key: "vision",   label: "Vision",   color: "#06b6d4" },
];

export function RiskChart() {
  const { history } = useLive();
  const [visible, setVisible] = useState<Record<string, boolean>>({
    overall: true, movement: true, sensor: true, vision: true,
  });

  const data = history.map((f, i) => ({
    t: i,
    overall:  Math.round((f.risk.smoothed_risk ?? 0) * 100),
    movement: Math.round((f.risk.movement_risk ?? 0) * 100),
    sensor:   Math.round((f.risk.sensor_risk   ?? 0) * 100),
    vision:   Math.round((f.risk.vision_risk   ?? 0) * 100),
  }));

  return (
    <div className="glass p-4 space-y-3">
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase">
          Live Risk Trend
        </span>
        <div className="flex gap-2 flex-wrap">
          {LINES.map(({ key, label, color }) => (
            <button key={key}
              onClick={() => setVisible(v => ({ ...v, [key]: !v[key] }))}
              className={`text-[10px] px-2 py-0.5 rounded border transition
                ${visible[key] ? "opacity-100" : "opacity-30"}`}
              style={{ color, borderColor: color + "55" }}>
              {label}
            </button>
          ))}
        </div>
      </div>

      <ResponsiveContainer width="100%" height={180}>
        <LineChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: -20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis dataKey="t" tick={false} axisLine={false} />
          <YAxis domain={[0, 100]} tick={{ fontSize: 10, fill: "#475569" }}
            axisLine={false} tickLine={false} />
          <Tooltip
            contentStyle={{ background: "#111827", border: "1px solid #1e2d45",
              borderRadius: 8, fontSize: 11 }}
            labelStyle={{ display: "none" }}
            formatter={(v, name) => [`${Number(v ?? 0)}%`, String(name)]} />
          <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="4 2" strokeOpacity={0.4} />
          <ReferenceLine y={40} stroke="#f59e0b" strokeDasharray="4 2" strokeOpacity={0.4} />
          {LINES.map(({ key, color }) =>
            visible[key] ? (
              <Line key={key} type="monotone" dataKey={key}
                stroke={color} strokeWidth={2} dot={false}
                isAnimationActive={false} />
            ) : null
          )}
        </LineChart>
      </ResponsiveContainer>

      <div className="flex gap-4 text-[10px] text-slate-600">
        <span className="text-red-400/60">── High (70%)</span>
        <span className="text-yellow-400/60">── Moderate (40%)</span>
      </div>
    </div>
  );
}
