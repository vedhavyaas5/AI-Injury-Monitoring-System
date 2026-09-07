"use client";
import React, { useEffect, useState } from "react";
import { Camera, Wifi, Cpu, Database, Activity, Server } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

interface Status {
  camera_connected: boolean; sensors_connected: boolean;
  movement_model: boolean; sensor_model: boolean;
  vision_model: boolean; risk_fusion: boolean;
  websocket_clients: number; demo_mode: boolean;
  cpu_pct: number; ram_gb: number;
  movement_model_name: string; sensor_model_name: string;
}

function Row({ icon: Icon, label, ok, value }: {
  icon: React.ElementType; label: string; ok: boolean; value?: string;
}) {
  return (
    <div className="flex items-center justify-between py-2 border-b last:border-0"
         style={{ borderColor: "var(--border)" }}>
      <div className="flex items-center gap-2">
        <Icon size={13} className="text-slate-500" />
        <span className="text-xs text-slate-300">{label}</span>
      </div>
      <div className="flex items-center gap-2">
        {value && <span className="text-[10px] text-slate-500">{value}</span>}
        <span className={`w-1.5 h-1.5 rounded-full ${ok ? "bg-green-400" : "bg-red-400"}`} />
        <span className={`text-[10px] font-medium ${ok ? "text-green-400" : "text-red-400"}`}>
          {ok ? "Online" : "Offline"}
        </span>
      </div>
    </div>
  );
}

export function SystemStatus() {
  const [status, setStatus] = useState<Status | null>(null);

  useEffect(() => {
    const load = () =>
      fetch(`${API}/api/system/status`)
        .then(r => r.json()).then(setStatus).catch(() => {});
    load();
    const t = setInterval(load, 5000);
    return () => clearInterval(t);
  }, []);

  if (!status) return (
    <div className="glass p-4 text-xs text-slate-500 text-center">Loading system status…</div>
  );

  return (
    <div className="glass p-4 space-y-1">
      <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block mb-3">
        System Status
      </span>

      <Row icon={Camera}   label="Camera"          ok={status.camera_connected} />
      <Row icon={Wifi}     label="Sensors"          ok={status.sensors_connected} />
      <Row icon={Database} label="Movement Model"   ok={status.movement_model}
           value={status.movement_model_name} />
      <Row icon={Database} label="Sensor Model"     ok={status.sensor_model}
           value={status.sensor_model_name} />
      <Row icon={Activity} label="Risk Fusion"       ok={status.risk_fusion} />
      <Row icon={Server}   label="WebSocket"         ok={status.websocket_clients > 0}
           value={`${status.websocket_clients} client(s)`} />

      <div className="pt-3 grid grid-cols-2 gap-2">
        <div className="p-2 rounded bg-white/5 text-center">
          <div className="text-[10px] text-slate-500">CPU</div>
          <div className="text-sm font-bold text-white">{status.cpu_pct}%</div>
        </div>
        <div className="p-2 rounded bg-white/5 text-center">
          <div className="text-[10px] text-slate-500">RAM</div>
          <div className="text-sm font-bold text-white">{status.ram_gb} GB</div>
        </div>
      </div>

      {status.demo_mode && (
        <div className="pt-2">
          <span className="demo-badge">DEMO MODE ACTIVE — Data is simulated</span>
        </div>
      )}
    </div>
  );
}
