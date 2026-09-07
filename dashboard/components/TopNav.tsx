"use client";
import { Bell, Search, User, Camera, Wifi, Cpu } from "lucide-react";
import { useLive } from "./LiveDataProvider";

export function TopNav() {
  const { connected, demo, frame, athleteId } = useLive();

  const Dot = ({ ok }: { ok: boolean }) => (
    <span className={`w-1.5 h-1.5 rounded-full inline-block mr-1
      ${ok ? "bg-green-400" : "bg-red-400"}`} />
  );

  return (
    <header className="h-12 border-b flex items-center justify-between px-4 shrink-0"
      style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>

      {/* Left — brand + status chips */}
      <div className="flex items-center gap-4">
        <span className="font-semibold text-sm text-white hidden lg:block">
          SPORTS AI MONITOR
        </span>
        <div className="flex items-center gap-3 text-[11px] text-slate-400">
          <span><Dot ok={connected} />AI Engine</span>
          <span><Dot ok={connected} />WebSocket</span>
          <span><Dot ok={false} />Camera</span>
          <span><Dot ok={false} />Sensors</span>
        </div>
      </div>

      {/* Centre — session info */}
      <div className="hidden lg:flex items-center gap-3 text-xs text-slate-400">
        {frame && (
          <>
            <span className="text-slate-500">Session</span>
            <span className="text-white font-mono">{frame.session_id}</span>
            <span className="text-slate-500">Athlete</span>
            <span className="text-white font-mono">#{athleteId}</span>
            {demo && <span className="demo-badge">DEMO</span>}
          </>
        )}
      </div>

      {/* Right */}
      <div className="flex items-center gap-2">
        <div className="relative hidden lg:block">
          <Search size={14} className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-500" />
          <input placeholder="Search…"
            className="bg-white/5 border border-white/10 rounded-lg pl-8 pr-3 py-1.5
                       text-xs text-slate-300 placeholder:text-slate-600 outline-none w-36
                       focus:border-blue-500/50 transition" />
        </div>
        <button className="relative p-1.5 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition">
          <Bell size={16} />
          <span className="absolute top-0.5 right-0.5 w-1.5 h-1.5 rounded-full bg-red-500" />
        </button>
        <button className="p-1.5 rounded-lg hover:bg-white/5 text-slate-400 hover:text-white transition">
          <User size={16} />
        </button>
      </div>
    </header>
  );
}
