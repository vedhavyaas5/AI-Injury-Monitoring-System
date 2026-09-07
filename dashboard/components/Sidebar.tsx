"use client";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard, Radio, Users, Calendar,
  BarChart2, Bell, FileText, Settings, Zap
} from "lucide-react";
import { useLive } from "./LiveDataProvider";

const NAV = [
  { href: "/",           icon: LayoutDashboard, label: "Dashboard"    },
  { href: "/live",       icon: Radio,           label: "Live Monitor" },
  { href: "/athletes",   icon: Users,           label: "Athletes"     },
  { href: "/sessions",   icon: Calendar,        label: "Sessions"     },
  { href: "/analytics",  icon: BarChart2,       label: "Analytics"    },
  { href: "/alerts",     icon: Bell,            label: "Alerts"       },
  { href: "/reports",    icon: FileText,        label: "Reports"      },
  { href: "/evaluation", icon: Zap,             label: "AI Evaluation"},
  { href: "/settings",   icon: Settings,        label: "Settings"     },
];

export function Sidebar() {
  const path      = usePathname();
  const { connected, demo } = useLive();

  return (
    <aside className="hidden md:flex flex-col w-56 shrink-0 border-r"
      style={{ background: "var(--bg-card)", borderColor: "var(--border)" }}>
      {/* Brand */}
      <div className="px-5 py-4 border-b flex items-center gap-2"
           style={{ borderColor: "var(--border)" }}>
        <div className="w-7 h-7 rounded-lg bg-blue-600 flex items-center justify-center">
          <Zap size={14} className="text-white" />
        </div>
        <span className="font-bold text-sm tracking-wide text-white">
          SPORTS AI<br />
          <span className="text-[10px] font-normal text-slate-400 tracking-widest">
            MONITOR
          </span>
        </span>
      </div>

      {/* Nav */}
      <nav className="flex-1 px-3 py-4 space-y-0.5 overflow-y-auto">
        {NAV.map(({ href, icon: Icon, label }) => {
          const active = path === href || (href !== "/" && path.startsWith(href));
          return (
            <Link key={href} href={href}
              className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm transition-all
                ${active
                  ? "bg-blue-600/20 text-blue-400 font-medium"
                  : "text-slate-400 hover:text-slate-200 hover:bg-white/5"}`}>
              <Icon size={16} />
              {label}
              {label === "Alerts" && (
                <span className="ml-auto w-4 h-4 rounded-full bg-red-500 text-white
                                 text-[9px] flex items-center justify-center font-bold">
                  !
                </span>
              )}
            </Link>
          );
        })}
      </nav>

      {/* System status */}
      <div className="px-4 py-3 border-t" style={{ borderColor: "var(--border)" }}>
        <div className="flex items-center gap-2 text-xs">
          <span className={`w-2 h-2 rounded-full ${connected ? "bg-green-400 pulse-green" : "bg-red-400 pulse-red"}`} />
          <span className={connected ? "text-green-400" : "text-red-400"}>
            {connected ? "System Online" : "Reconnecting…"}
          </span>
        </div>
        {demo && <span className="demo-badge mt-1 inline-block">DEMO MODE</span>}
      </div>
    </aside>
  );
}
