"use client";
import React, { useEffect, useState } from "react";
import { AlertCard } from "@/components/AlertCard";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

const FILTERS = ["ALL", "HIGH", "MODERATE", "LOW"];

export default function AlertsPage() {
  const [alerts, setAlerts] = useState<any[]>([]);
  const [filter, setFilter] = useState("ALL");

  const load = () =>
    fetch(`${API}/api/alerts?last_n=50`).then(r => r.json()).then(setAlerts).catch(() => {});

  useEffect(() => {
    load();
    const t = setInterval(load, 3000);
    return () => clearInterval(t);
  }, []);

  const ack = (id: string) => {
    fetch(`${API}/api/alerts/${id}/acknowledge`, { method: "POST" })
      .then(load).catch(() => {});
  };

  const shown = filter === "ALL"
    ? alerts
    : alerts.filter(a => a.level?.toUpperCase() === filter);

  return (
    <div className="space-y-5 max-w-screen-lg mx-auto">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-bold text-white">Alert Center</h1>
          <p className="text-sm text-slate-500">
            {alerts.filter(a => !a.acknowledged).length} unacknowledged alerts
          </p>
        </div>
        <div className="flex gap-1.5">
          {FILTERS.map(f => (
            <button key={f} onClick={() => setFilter(f)}
              className={`text-xs px-3 py-1.5 rounded-lg border transition
                ${filter === f
                  ? "bg-blue-600/20 border-blue-500/40 text-blue-400"
                  : "bg-white/5 border-white/10 text-slate-400 hover:text-white"}`}>
              {f}
            </button>
          ))}
        </div>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-4 gap-3">
        {[
          { label: "Total",      count: alerts.length,                                  color: "text-white"        },
          { label: "High",       count: alerts.filter(a=>a.level==="HIGH").length,      color: "text-red-400"      },
          { label: "Moderate",   count: alerts.filter(a=>a.level==="MODERATE").length,  color: "text-yellow-400"   },
          { label: "Unacked",    count: alerts.filter(a=>!a.acknowledged).length,       color: "text-orange-400"   },
        ].map(({ label, count, color }) => (
          <div key={label} className="glass p-3 text-center">
            <div className={`text-xl font-bold ${color}`}>{count}</div>
            <div className="text-[10px] text-slate-500 uppercase">{label}</div>
          </div>
        ))}
      </div>

      {/* Alert list */}
      <div className="space-y-3">
        {shown.length === 0
          ? <div className="glass p-8 text-center text-slate-500">No alerts to display.</div>
          : shown.map(a => <AlertCard key={a.id} alert={a} onAck={ack} />)}
      </div>
    </div>
  );
}
