"use client";
import React, { useEffect, useState } from "react";
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis,
  ResponsiveContainer, LineChart, Line,
  XAxis, YAxis, CartesianGrid, Tooltip
} from "recharts";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

function MetricBadge({ label, value, good }: { label: string; value: number; good: boolean }) {
  return (
    <div className={`p-2 rounded text-center border
      ${good ? "bg-green-500/10 border-green-500/20" : "bg-yellow-500/10 border-yellow-500/20"}`}>
      <div className={`text-lg font-bold ${good ? "text-green-400" : "text-yellow-400"}`}>
        {(value * 100).toFixed(1)}%
      </div>
      <div className="text-[9px] text-slate-500 uppercase">{label}</div>
    </div>
  );
}

function ConfusionMatrix({ cm, labels }: { cm: number[][]; labels: string[] }) {
  if (!cm || cm.length === 0) return null;
  return (
    <div className="overflow-x-auto">
      <table className="text-xs w-full">
        <thead>
          <tr>
            <th className="p-2 text-slate-600 font-normal text-left">Actual \ Pred</th>
            {labels.map(l => (
              <th key={l} className="p-2 text-slate-400 font-medium">{l}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cm.map((row, i) => (
            <tr key={i}>
              <td className="p-2 text-slate-400 font-medium">{labels[i]}</td>
              {row.map((val, j) => (
                <td key={j} className={`p-2 text-center font-mono rounded
                  ${i === j ? "bg-green-500/20 text-green-300" : "bg-red-500/10 text-red-300"}`}>
                  {val}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

interface ModelResult {
  model_name: string;
  accuracy: number; precision: number;
  recall: number; f1: number; roc_auc: number | null;
  confusion_matrix: number[][];
  high_risk_class: { precision: number; recall: number; f1: number; note: string };
  inference_latency_ms_per_sample: number;
  test_samples: number; positive_samples: number;
  error?: string;
}

export default function EvaluationPage() {
  const [results, setResults] = useState<{ movement?: ModelResult; sensor?: ModelResult } | null>(null);
  const [rocM, setRocM] = useState<any>(null);
  const [rocS, setRocS] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([
      fetch(`${API}/api/evaluation/models`).then(r => r.json()),
      fetch(`${API}/api/evaluation/roc/movement`).then(r => r.json()),
      fetch(`${API}/api/evaluation/roc/sensor`).then(r => r.json()),
    ]).then(([eval_, rM, rS]) => {
      setResults(eval_);
      setRocM(rM);
      setRocS(rS);
      setLoading(false);
    }).catch(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="flex items-center justify-center h-64 text-slate-500">
      Loading evaluation results…
    </div>
  );

  const ModelCard = ({ name, data }: { name: string; data?: ModelResult }) => {
    if (!data) return null;
    if (data.error) return (
      <div className="glass p-4 text-red-400 text-xs">Error: {data.error}</div>
    );

    const radarData = [
      { metric: "Accuracy",  val: data.accuracy  * 100 },
      { metric: "Precision", val: data.precision * 100 },
      { metric: "Recall",    val: data.recall    * 100 },
      { metric: "F1",        val: data.f1        * 100 },
      { metric: "AUC",       val: (data.roc_auc ?? 0) * 100 },
    ];

    return (
      <div className="glass p-5 space-y-4">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase">
              {name} Model
            </span>
            <div className="text-xs text-slate-500 mt-0.5">
              {data.model_name} · {data.test_samples} test samples ·
              {data.positive_samples} positive · {data.inference_latency_ms_per_sample.toFixed(3)} ms/sample
            </div>
          </div>
        </div>

        {/* Metric badges */}
        <div className="grid grid-cols-5 gap-2">
          <MetricBadge label="Accuracy"  value={data.accuracy}  good={data.accuracy  > 0.85} />
          <MetricBadge label="Precision" value={data.precision} good={data.precision > 0.80} />
          <MetricBadge label="Recall"    value={data.recall}    good={data.recall    > 0.80} />
          <MetricBadge label="F1"        value={data.f1}        good={data.f1        > 0.80} />
          <MetricBadge label="ROC-AUC"   value={data.roc_auc ?? 0} good={(data.roc_auc ?? 0) > 0.85} />
        </div>

        {/* High-risk class focus */}
        <div className="p-3 rounded-lg bg-red-500/5 border border-red-500/15 space-y-2">
          <span className="text-[11px] font-semibold text-red-400 uppercase tracking-widest">
            High-Risk Class Performance
          </span>
          <div className="grid grid-cols-3 gap-2">
            <MetricBadge label="Precision" value={data.high_risk_class.precision} good={data.high_risk_class.precision > 0.7} />
            <MetricBadge label="Recall"    value={data.high_risk_class.recall}    good={data.high_risk_class.recall    > 0.7} />
            <MetricBadge label="F1"        value={data.high_risk_class.f1}        good={data.high_risk_class.f1        > 0.7} />
          </div>
          <p className="text-[10px] text-slate-500">{data.high_risk_class.note}</p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* Radar */}
          <div>
            <div className="text-[10px] text-slate-500 mb-2">Performance Radar</div>
            <ResponsiveContainer width="100%" height={180}>
              <RadarChart data={radarData}>
                <PolarGrid stroke="rgba(255,255,255,0.06)" />
                <PolarAngleAxis dataKey="metric"
                  tick={{ fontSize: 9, fill: "#64748b" }} />
                <Radar dataKey="val" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.2} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Confusion matrix */}
          <div>
            <div className="text-[10px] text-slate-500 mb-2">Confusion Matrix (Test Set)</div>
            <ConfusionMatrix cm={data.confusion_matrix}
              labels={["No Injury", "Injury"]} />
          </div>
        </div>
      </div>
    );
  };

  // ROC chart data
  const rocChart = (roc: any) => {
    if (!roc || roc.error) return [];
    return roc.fpr.map((fpr: number, i: number) => ({
      fpr: +(fpr * 100).toFixed(1),
      tpr: +((roc.tpr[i] ?? 0) * 100).toFixed(1),
    }));
  };

  const TT = {
    contentStyle: { background: "#111827", border: "1px solid #1e2d45", borderRadius: 8, fontSize: 11 },
  };

  return (
    <div className="space-y-5 max-w-screen-xl mx-auto">
      <div>
        <h1 className="text-xl font-bold text-white">AI Model Evaluation</h1>
        <p className="text-sm text-slate-500">
          Real metrics computed from held-out test sets — no fabricated numbers.
        </p>
      </div>

      <ModelCard name="Physical Movement" data={results?.movement} />
      <ModelCard name="Sensor-Based"      data={results?.sensor}   />

      {/* ROC curves */}
      <div className="glass p-5 space-y-4">
        <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase block">
          ROC Curves
        </span>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {[
            { label: "Movement Model", data: rocChart(rocM), auc: rocM?.auc, color: "#3b82f6" },
            { label: "Sensor Model",   data: rocChart(rocS), auc: rocS?.auc, color: "#8b5cf6" },
          ].map(({ label, data, auc, color }) => (
            <div key={label}>
              <div className="text-xs text-slate-400 mb-1">
                {label} &nbsp;
                {auc != null && (
                  <span className="text-white font-medium">AUC = {auc.toFixed(4)}</span>
                )}
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={data} margin={{ top:4, right:4, bottom:16, left:-16 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
                  <XAxis dataKey="fpr" type="number" domain={[0,100]}
                    label={{ value:"FPR %", position:"insideBottom", offset:-8,
                      fill:"#475569", fontSize:9 }}
                    tick={{ fontSize:9, fill:"#475569" }} />
                  <YAxis domain={[0,100]}
                    label={{ value:"TPR %", angle:-90, position:"insideLeft",
                      fill:"#475569", fontSize:9 }}
                    tick={{ fontSize:9, fill:"#475569" }} />
                  <Tooltip {...TT}
                    formatter={(v, n) => [`${Number(v??0)}%`, n === "tpr" ? "TPR" : "FPR"]} />
                  {/* Diagonal reference */}
                  <Line dataKey="fpr" stroke="rgba(255,255,255,0.1)"
                    strokeDasharray="4 2" dot={false} isAnimationActive={false} name="Baseline" />
                  <Line type="monotone" dataKey="tpr" stroke={color}
                    strokeWidth={2} dot={false} isAnimationActive={false} name="TPR" />
                </LineChart>
              </ResponsiveContainer>
            </div>
          ))}
        </div>
      </div>

      {/* Disclaimer */}
      <div className="glass p-4 border border-yellow-500/10 bg-yellow-500/5">
        <p className="text-[11px] text-yellow-300/70">
          Metrics are computed on the held-out test split from Phase 3.
          High performance on this dataset may reflect the dataset's characteristics.
          Always validate on independent, real-world data before deployment.
        </p>
      </div>
    </div>
  );
}
