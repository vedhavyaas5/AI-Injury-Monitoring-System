"use client";
import React, { useEffect, useState, Suspense } from "react";
import { useSearchParams } from "next/navigation";
import { FileText, Download } from "lucide-react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function ReportsInner() {
  const params   = useSearchParams();
  const sessionId = params.get("session");
  const [sessions, setSessions] = useState<any[]>([]);
  const [report,   setReport]   = useState<any>(null);
  const [loading,  setLoading]  = useState(false);

  useEffect(() => {
    fetch(`${API}/api/sessions`).then(r => r.json()).then(setSessions).catch(() => {});
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    setLoading(true);
    fetch(`${API}/api/reports/${sessionId}`)
      .then(r => r.json()).then(setReport)
      .catch(() => setReport(null))
      .finally(() => setLoading(false));
  }, [sessionId]);

  const exportCSV = () => {
    if (!report) return;
    const rows = [
      ["Metric", "Value"],
      ["Session ID",      report.session?.id],
      ["Athlete",         report.session?.athlete_id],
      ["Duration (min)",  report.session?.duration_min],
      ["Peak Risk",       (report.session?.peak_risk * 100).toFixed(1) + "%"],
      ["Avg Risk",        (report.session?.avg_risk  * 100).toFixed(1) + "%"],
      ["Peak HR",         report.session?.peak_hr],
      ["Alerts",          report.session?.alerts_count],
    ];
    const csv = rows.map(r => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url  = URL.createObjectURL(blob);
    const a = document.createElement("a"); a.href = url;
    a.download = `report_${sessionId}.csv`; a.click();
  };

  return (
    <div className="space-y-5 max-w-screen-xl mx-auto">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-xl font-bold text-white">Reports</h1>
          <p className="text-sm text-slate-500">Session and athlete analysis reports</p>
        </div>
        {report && (
          <button onClick={exportCSV}
            className="flex items-center gap-2 text-xs px-3 py-2 rounded-lg
                       bg-blue-600/20 border border-blue-500/30 text-blue-400 hover:bg-blue-600/30 transition">
            <Download size={13} /> Export CSV
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">

        {/* Session list */}
        <div className="glass p-4 space-y-2">
          <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block mb-3">
            Select Session
          </span>
          {sessions.map(s => (
            <a key={s.id} href={`/reports?session=${s.id}`}
              className={`block p-2 rounded-lg text-xs cursor-pointer transition
                ${sessionId === s.id
                  ? "bg-blue-600/20 text-blue-400 border border-blue-500/30"
                  : "hover:bg-white/5 text-slate-400"}`}>
              <div className="font-medium">#{s.athlete_id} — {s.type}</div>
              <div className="text-slate-600">{s.duration_min.toFixed(0)} min · Peak {(s.peak_risk*100).toFixed(0)}%</div>
            </a>
          ))}
        </div>

        {/* Report content */}
        <div className="lg:col-span-3">
          {loading && (
            <div className="glass p-8 text-center text-slate-500">Loading report…</div>
          )}

          {!sessionId && !loading && (
            <div className="glass p-8 text-center">
              <FileText size={32} className="text-slate-700 mx-auto mb-3" />
              <p className="text-slate-500">Select a session to generate a report</p>
            </div>
          )}

          {report && !loading && (
            <div className="space-y-4">

              {/* Header */}
              <div className="glass p-5 space-y-1">
                <div className="text-lg font-bold text-white">
                  Session Report — Athlete #{report.session?.athlete_id}
                </div>
                <div className="text-xs text-slate-500">
                  {report.session?.type} · Duration: {report.session?.duration_min?.toFixed(0)} min
                </div>
              </div>

              {/* KPIs */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: "Avg Risk",   val: `${((report.session?.avg_risk??0)*100).toFixed(1)}%`,  color: "text-yellow-400" },
                  { label: "Peak Risk",  val: `${((report.session?.peak_risk??0)*100).toFixed(1)}%`, color: "text-red-400" },
                  { label: "Peak HR",    val: `${report.session?.peak_hr?.toFixed(0)} bpm`,          color: "text-white" },
                  { label: "Alerts",     val: report.session?.alerts_count,                          color: "text-orange-400" },
                ].map(({ label, val, color }) => (
                  <div key={label} className="glass p-3 text-center">
                    <div className={`text-xl font-bold ${color}`}>{val}</div>
                    <div className="text-[10px] text-slate-500 uppercase">{label}</div>
                  </div>
                ))}
              </div>

              {/* Model evaluation */}
              {report.model_eval && (
                <div className="glass p-4 space-y-3">
                  <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
                    Model Performance
                  </span>
                  <div className="overflow-x-auto">
                    <table className="w-full text-xs">
                      <thead>
                        <tr className="text-[10px] text-slate-600 uppercase">
                          <th className="pb-2 text-left">Model</th>
                          <th className="pb-2">Accuracy</th>
                          <th className="pb-2">Precision</th>
                          <th className="pb-2">Recall</th>
                          <th className="pb-2">F1</th>
                          <th className="pb-2">AUC</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(report.model_eval).map(([key, m]: [string, any]) => (
                          !m.error && (
                            <tr key={key} className="border-t" style={{ borderColor: "var(--border)" }}>
                              <td className="py-2 text-slate-300 capitalize">{key}</td>
                              <td className="py-2 text-center font-mono text-white">{(m.accuracy*100).toFixed(1)}%</td>
                              <td className="py-2 text-center font-mono text-white">{(m.precision*100).toFixed(1)}%</td>
                              <td className="py-2 text-center font-mono text-green-400 font-bold">{(m.recall*100).toFixed(1)}%</td>
                              <td className="py-2 text-center font-mono text-white">{(m.f1*100).toFixed(1)}%</td>
                              <td className="py-2 text-center font-mono text-blue-400">{m.roc_auc != null ? (m.roc_auc*100).toFixed(1)+"%" : "—"}</td>
                            </tr>
                          )
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Alerts */}
              {report.alerts?.length > 0 && (
                <div className="glass p-4 space-y-2">
                  <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
                    Alert History ({report.alerts.length})
                  </span>
                  <table className="w-full text-xs">
                    <thead>
                      <tr className="text-[10px] text-slate-600 uppercase">
                        <th className="pb-2 text-left">Time</th>
                        <th className="pb-2">Level</th>
                        <th className="pb-2 text-right">Risk</th>
                        <th className="pb-2 text-left pl-4">Reason</th>
                      </tr>
                    </thead>
                    <tbody>
                      {report.alerts.map((a: any) => (
                        <tr key={a.id} className="border-t" style={{ borderColor: "var(--border)" }}>
                          <td className="py-1.5 text-slate-500 font-mono">
                            {new Date(a.timestamp*1000).toLocaleTimeString()}
                          </td>
                          <td className={`py-1.5 text-center font-medium
                            ${a.level === "HIGH" ? "text-red-400" : a.level === "MODERATE" ? "text-yellow-400" : "text-green-400"}`}>
                            {a.level}
                          </td>
                          <td className="py-1.5 text-right font-mono text-white">
                            {a.risk_score?.toFixed(1)}%
                          </td>
                          <td className="py-1.5 pl-4 text-slate-400">
                            {a.reasons?.[0] ?? "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}

              <div className="glass p-3 border border-yellow-500/10 bg-yellow-500/5">
                <p className="text-[10px] text-yellow-300/70">{report.disclaimer}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default function ReportsPage() {
  return (
    <Suspense fallback={
      <div className="flex items-center justify-center h-64 text-slate-500">
        Loading reports…
      </div>
    }>
      <ReportsInner />
    </Suspense>
  );
}