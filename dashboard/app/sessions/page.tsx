"use client";
import React, { useEffect, useState } from "react";
import Link from "next/link";
import { Calendar, Clock, TrendingUp, Bell } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function fmt(ts: number) {
  return new Date(ts * 1000).toLocaleDateString("en-GB", {
    day: "numeric", month: "short", year: "numeric",
  });
}

export default function SessionsPage() {
  const [sessions, setSessions] = useState<any[]>([]);

  useEffect(() => {
    fetch(`${API}/api/sessions`).then(r => r.json()).then(setSessions).catch(() => {});
    const t = setInterval(() =>
      fetch(`${API}/api/sessions`).then(r => r.json()).then(setSessions).catch(() => {}), 4000);
    return () => clearInterval(t);
  }, []);

  return (
    <div className="space-y-5 max-w-screen-xl mx-auto">
      <div>
        <h1 className="text-xl font-bold text-white">Training Sessions</h1>
        <p className="text-sm text-slate-500">{sessions.length} sessions recorded</p>
      </div>

      {/* Summary stats */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: "Total Sessions",  val: sessions.length },
          { label: "Active Now",      val: sessions.filter(s => s.status === "active").length },
          { label: "Avg Peak Risk",   val: sessions.length
              ? `${(sessions.reduce((a,s) => a + s.peak_risk, 0) / sessions.length * 100).toFixed(0)}%`
              : "—" },
          { label: "Total Alerts",    val: sessions.reduce((a,s) => a + (s.alerts_count||0), 0) },
        ].map(({ label, val }) => (
          <div key={label} className="glass p-3 text-center">
            <div className="text-xl font-bold text-white">{val}</div>
            <div className="text-[10px] text-slate-500 uppercase">{label}</div>
          </div>
        ))}
      </div>

      {/* Session list */}
      <div className="space-y-3">
        {sessions.length === 0 ? (
          <div className="glass p-8 text-center text-slate-500">No sessions yet.</div>
        ) : sessions.map(s => (
          <div key={s.id} className="glass p-4 hover:border-blue-500/30 transition-all">
            <div className="flex flex-wrap items-center justify-between gap-3">

              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-blue-600/15 border border-blue-500/20
                                flex items-center justify-center">
                  <Calendar size={16} className="text-blue-400" />
                </div>
                <div>
                  <div className="font-semibold text-sm text-white">
                    {s.type} — Athlete #{s.athlete_id}
                  </div>
                  <div className="text-[10px] text-slate-500">{fmt(s.start_time)}</div>
                </div>
              </div>

              <div className="flex items-center gap-6 text-xs flex-wrap">
                <div className="flex items-center gap-1.5 text-slate-400">
                  <Clock size={12} />
                  <span>{s.duration_min.toFixed(0)} min</span>
                </div>
                <div className="flex items-center gap-1.5">
                  <TrendingUp size={12} className="text-red-400" />
                  <span className={`font-medium ${s.peak_risk >= 0.7 ? "text-red-400" : s.peak_risk >= 0.4 ? "text-yellow-400" : "text-green-400"}`}>
                    Peak {(s.peak_risk * 100).toFixed(0)}%
                  </span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-400">
                  <span>Avg {(s.avg_risk * 100).toFixed(0)}%</span>
                </div>
                <div className="flex items-center gap-1.5 text-slate-400">
                  <Bell size={12} />
                  <span>{s.alerts_count || 0} alerts</span>
                </div>
                <span className={`text-[10px] px-2 py-0.5 rounded border
                  ${s.status === "active"
                    ? "bg-green-500/10 border-green-500/20 text-green-400"
                    : "bg-slate-500/10 border-slate-500/20 text-slate-400"}`}>
                  {s.status}
                </span>
              </div>

              <Link href={`/reports?session=${s.id}`}
                className="text-xs px-3 py-1.5 rounded-lg bg-white/5
                           border border-white/10 text-slate-300
                           hover:bg-white/10 transition">
                View Report
              </Link>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
