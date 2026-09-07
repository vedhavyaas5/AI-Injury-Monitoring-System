"use client";
import React from "react";
import { useLive } from "./LiveDataProvider";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer
} from "recharts";

function asym(l: number, r: number) {
  const avg = (l + r) / 2;
  return avg > 0 ? Math.abs((l - r) / avg) * 100 : 0;
}

function Joint({ name, left, right }: { name: string; left: number; right: number }) {
  const a = asym(left, right);
  const flag = a > 10;
  return (
    <tr className="border-b" style={{ borderColor: "var(--border)" }}>
      <td className="py-2 pr-4 text-xs text-slate-400">{name}</td>
      <td className="py-2 pr-4 text-xs font-mono text-white">{left.toFixed(1)}°</td>
      <td className="py-2 pr-4 text-xs font-mono text-white">{right.toFixed(1)}°</td>
      <td className={`py-2 text-xs font-mono ${flag ? "text-yellow-400" : "text-green-400"}`}>
        {a.toFixed(1)}%
      </td>
    </tr>
  );
}

export function JointAnalysis() {
  const { frame, history } = useLive();
  const j = frame?.joints;

  const chartData = history.slice(-30).map((f, i) => ({
    t: i,
    knee: ((f.joints?.l_knee ?? 0) + (f.joints?.r_knee ?? 0)) / 2,
    hip:  ((f.joints?.l_hip  ?? 0) + (f.joints?.r_hip  ?? 0)) / 2,
    ankle:((f.joints?.l_ankle?? 0) + (f.joints?.r_ankle?? 0)) / 2,
  }));

  if (!j) return (
    <div className="glass p-4 text-sm text-slate-500 text-center">
      Waiting for joint data…
    </div>
  );

  return (
    <div className="glass p-4 space-y-4">
      <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
        Joint Analysis
      </span>

      <table className="w-full">
        <thead>
          <tr className="text-[10px] text-slate-600 uppercase">
            <th className="pb-2 text-left font-medium">Joint</th>
            <th className="pb-2 text-left font-medium">Left</th>
            <th className="pb-2 text-left font-medium">Right</th>
            <th className="pb-2 text-left font-medium">Asymmetry</th>
          </tr>
        </thead>
        <tbody>
          <Joint name="Hip"   left={j.l_hip   ?? 0} right={j.r_hip   ?? 0} />
          <Joint name="Knee"  left={j.l_knee  ?? 0} right={j.r_knee  ?? 0} />
          <Joint name="Ankle" left={j.l_ankle ?? 0} right={j.r_ankle ?? 0} />
        </tbody>
      </table>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="p-2 rounded bg-white/5">
          <div className="text-slate-500 text-[10px]">Angular Velocity</div>
          <div className="text-white font-mono">{(j.angular_velocity ?? 0).toFixed(2)} °/s</div>
        </div>
        <div className="p-2 rounded bg-white/5">
          <div className="text-slate-500 text-[10px]">GRF</div>
          <div className="text-white font-mono">{Math.round(j.ground_reaction_force ?? 0)} N</div>
        </div>
        <div className="p-2 rounded bg-white/5">
          <div className="text-slate-500 text-[10px]">Postural Instability</div>
          <div className="text-white font-mono">
            {((j.postural_instability_index ?? 0) * 100).toFixed(1)}%
          </div>
        </div>
        <div className="p-2 rounded bg-white/5">
          <div className="text-slate-500 text-[10px]">Movement Asymmetry</div>
          <div className={`font-mono ${(j.movement_asymmetry ?? 0) > 20 ? "text-yellow-400" : "text-green-400"}`}>
            {(j.movement_asymmetry ?? 0).toFixed(1)}°
          </div>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={120}>
        <LineChart data={chartData} margin={{ top: 4, right: 4, bottom: 0, left: -24 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
          <XAxis dataKey="t" tick={false} axisLine={false} />
          <YAxis domain={[60, 190]} tick={{ fontSize: 9, fill: "#475569" }}
            axisLine={false} tickLine={false} />
          <Tooltip contentStyle={{ background: "#111827", border: "1px solid #1e2d45",
            borderRadius: 6, fontSize: 10 }} labelStyle={{ display: "none" }}
            formatter={(v, n) => [`${Number(v ?? 0).toFixed(1)}°`, String(n)]} />
          <Line type="monotone" dataKey="knee"  stroke="#3b82f6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="hip"   stroke="#8b5cf6" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          <Line type="monotone" dataKey="ankle" stroke="#06b6d4" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        </LineChart>
      </ResponsiveContainer>
      <div className="flex gap-3 text-[10px]">
        <span className="text-blue-400">─ Knee</span>
        <span className="text-purple-400">─ Hip</span>
        <span className="text-cyan-400">─ Ankle</span>
      </div>
    </div>
  );
}
