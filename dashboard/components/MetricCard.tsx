"use client";
import React from "react";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";

interface Props {
  title: string;
  value: string | number;
  unit?: string;
  status?: "low" | "moderate" | "high" | "ok" | "normal";
  statusLabel?: string;
  trend?: "up" | "down" | "stable";
  trendValue?: string;
  icon?: React.ReactNode;
  sub?: string;
}

const STATUS_COLOR = {
  low:      "text-green-400",
  ok:       "text-green-400",
  normal:   "text-green-400",
  moderate: "text-yellow-400",
  high:     "text-red-400",
};

const STATUS_DOT = {
  low:      "bg-green-400",
  ok:       "bg-green-400",
  normal:   "bg-green-400",
  moderate: "bg-yellow-400",
  high:     "bg-red-400",
};

export function MetricCard({ title, value, unit, status, statusLabel, trend, trendValue, icon, sub }: Props) {
  const sc = status ? STATUS_COLOR[status] : "text-slate-400";
  const sd = status ? STATUS_DOT[status]   : "bg-slate-400";

  const TrendIcon = trend === "up" ? TrendingUp : trend === "down" ? TrendingDown : Minus;
  const trendColor = trend === "up" ? "text-red-400" : trend === "down" ? "text-green-400" : "text-slate-400";

  return (
    <div className="glass p-4 flex flex-col gap-2 hover:border-blue-500/30 transition-all">
      <div className="flex items-center justify-between">
        <span className="text-[11px] font-semibold tracking-widest text-slate-400 uppercase">
          {title}
        </span>
        {icon && <span className="text-slate-600">{icon}</span>}
      </div>

      <div className="flex items-end gap-1.5">
        <span className="text-2xl font-bold text-white leading-none">{value}</span>
        {unit && <span className="text-sm text-slate-500 mb-0.5">{unit}</span>}
      </div>

      {sub && <span className="text-xs text-slate-500">{sub}</span>}

      <div className="flex items-center justify-between mt-auto">
        {status && (
          <div className="flex items-center gap-1.5">
            <span className={`w-1.5 h-1.5 rounded-full ${sd}`} />
            <span className={`text-xs font-medium ${sc}`}>{statusLabel ?? status}</span>
          </div>
        )}
        {trend && trendValue && (
          <div className={`flex items-center gap-1 text-xs ${trendColor}`}>
            <TrendIcon size={12} />
            <span>{trendValue}</span>
          </div>
        )}
      </div>
    </div>
  );
}
