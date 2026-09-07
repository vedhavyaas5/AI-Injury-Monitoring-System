"use client";
import React, { useState, useEffect } from "react";
import { Save, RotateCcw } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const DEFAULTS = {
  riskLow: 40, riskMod: 70,
  wMovement: 35, wSensor: 30, wVision: 35,
  emaAlpha: 0.30, historyWindow: 10, consecHigh: 3,
  demoMode: true,
};

function Slider({
  label, value, min, max, step, unit, onChange, note
}: {
  label: string; value: number; min: number; max: number;
  step: number; unit?: string; onChange: (v: number) => void; note?: string;
}) {
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs">
        <span className="text-slate-300">{label}</span>
        <span className="font-mono text-white">{value}{unit}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={e => onChange(Number(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none bg-white/10
                   accent-blue-500 cursor-pointer" />
      {note && <p className="text-[10px] text-slate-600">{note}</p>}
    </div>
  );
}

export default function SettingsPage() {
  const [cfg, setCfg] = useState(DEFAULTS);
  const [saved, setSaved] = useState(false);

  const set = (key: keyof typeof DEFAULTS) => (val: number | boolean) =>
    setCfg(c => ({ ...c, [key]: val }));

  const wTotal = cfg.wMovement + cfg.wSensor + cfg.wVision;
  const wValid = Math.abs(wTotal - 100) < 1;

  const save = async () => {
    try {
      await fetch(`${API}/api/system/mode?demo=${cfg.demoMode}`, { method: "POST" });
      setSaved(true);
      setTimeout(() => setSaved(false), 2500);
    } catch {}
  };

  const reset = () => { setCfg(DEFAULTS); setSaved(false); };

  return (
    <div className="space-y-5 max-w-screen-md mx-auto">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-bold text-white">Settings</h1>
          <p className="text-sm text-slate-500">
            Configure risk thresholds, fusion weights, and system behaviour
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={reset}
            className="flex items-center gap-1.5 text-xs px-3 py-2 rounded-lg
                       bg-white/5 border border-white/10 text-slate-400 hover:text-white transition">
            <RotateCcw size={12} /> Reset
          </button>
          <button onClick={save}
            className={`flex items-center gap-1.5 text-xs px-3 py-2 rounded-lg transition
              ${saved
                ? "bg-green-600/20 border border-green-500/30 text-green-400"
                : "bg-blue-600/20 border border-blue-500/30 text-blue-400 hover:bg-blue-600/30"}`}>
            <Save size={12} /> {saved ? "Saved ✓" : "Save"}
          </button>
        </div>
      </div>

      {/* Risk Thresholds */}
      <div className="glass p-5 space-y-4">
        <div>
          <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
            Risk Thresholds
          </span>
          <p className="text-[10px] text-slate-600 mt-0.5">
            Application-level thresholds — NOT clinically validated.
          </p>
        </div>
        <div className="grid grid-cols-3 gap-3 text-center text-xs mb-2">
          <div className="p-2 rounded bg-green-500/10 border border-green-500/20">
            <div className="text-green-400 font-bold">LOW</div>
            <div className="text-slate-500">0 – {cfg.riskLow - 1}%</div>
          </div>
          <div className="p-2 rounded bg-yellow-500/10 border border-yellow-500/20">
            <div className="text-yellow-400 font-bold">MODERATE</div>
            <div className="text-slate-500">{cfg.riskLow} – {cfg.riskMod - 1}%</div>
          </div>
          <div className="p-2 rounded bg-red-500/10 border border-red-500/20">
            <div className="text-red-400 font-bold">HIGH</div>
            <div className="text-slate-500">≥ {cfg.riskMod}%</div>
          </div>
        </div>
        <Slider label="LOW / MODERATE boundary" value={cfg.riskLow}
          min={20} max={60} step={1} unit="%" onChange={set("riskLow")} />
        <Slider label="MODERATE / HIGH boundary" value={cfg.riskMod}
          min={50} max={90} step={1} unit="%" onChange={set("riskMod")} />
      </div>

      {/* Fusion Weights */}
      <div className="glass p-5 space-y-4">
        <div>
          <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
            Fusion Weights
          </span>
          <p className="text-[10px] text-slate-600 mt-0.5">
            Engineering defaults — not medically validated. Must sum to 100%.
          </p>
        </div>
        <Slider label="Movement Model" value={cfg.wMovement}
          min={0} max={100} step={5} unit="%" onChange={set("wMovement")} />
        <Slider label="Sensor Model" value={cfg.wSensor}
          min={0} max={100} step={5} unit="%" onChange={set("wSensor")} />
        <Slider label="Vision Model" value={cfg.wVision}
          min={0} max={100} step={5} unit="%" onChange={set("wVision")} />
        <div className={`text-xs px-3 py-2 rounded border text-center
          ${wValid
            ? "bg-green-500/10 border-green-500/20 text-green-400"
            : "bg-red-500/10 border-red-500/20 text-red-400"}`}>
          Total: {wTotal}% {wValid ? "✓" : `— must equal 100% (off by ${wTotal - 100}%)`}
        </div>
      </div>

      {/* Smoothing */}
      <div className="glass p-5 space-y-4">
        <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
          Temporal Smoothing
        </span>
        <Slider label="EMA Alpha" value={cfg.emaAlpha}
          min={0.05} max={0.95} step={0.05} onChange={set("emaAlpha") as any}
          note="Higher = more responsive to changes. Lower = smoother." />
        <Slider label="Risk History Window" value={cfg.historyWindow}
          min={3} max={30} step={1} unit=" frames" onChange={set("historyWindow")}
          note="Frames used for trend analysis." />
        <Slider label="Consecutive High Before Alert" value={cfg.consecHigh}
          min={1} max={10} step={1} unit=" frames" onChange={set("consecHigh")}
          note="Prevents single-frame false alarms." />
      </div>

      {/* Demo Mode */}
      <div className="glass p-5 space-y-3">
        <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
          Data Mode
        </span>
        <div className="flex gap-3">
          {[
            { label: "DEMO MODE",  val: true,  desc: "Synthetic simulated data" },
            { label: "LIVE MODE",  val: false, desc: "Real sensors and camera"  },
          ].map(({ label, val, desc }) => (
            <button key={label} onClick={() => set("demoMode")(val)}
              className={`flex-1 p-3 rounded-lg border text-left transition
                ${cfg.demoMode === val
                  ? "bg-blue-600/20 border-blue-500/40 text-blue-400"
                  : "bg-white/5 border-white/10 text-slate-400 hover:border-white/20"}`}>
              <div className="font-bold text-sm">{label}</div>
              <div className="text-[10px] mt-0.5 opacity-70">{desc}</div>
            </button>
          ))}
        </div>
        {cfg.demoMode && (
          <p className="text-[10px] text-yellow-400/70">
            DEMO MODE: all displayed data is simulated and not from real sensors.
          </p>
        )}
      </div>

      {/* Disclaimer */}
      <div className="glass p-4 border border-yellow-500/10 bg-yellow-500/5">
        <p className="text-[10px] text-yellow-300/70">
          All configuration values are application-level engineering parameters.
          None have been clinically validated. Consult domain experts before
          adjusting thresholds for any semi-clinical or research use.
        </p>
      </div>
    </div>
  );
}
