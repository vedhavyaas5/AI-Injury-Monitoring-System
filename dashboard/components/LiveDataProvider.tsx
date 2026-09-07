"use client";
import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const WS  = API.replace("http", "ws");

export interface LiveFrame {
  type: string;
  demo: boolean;
  athlete_id: string;
  session_id: string;
  timestamp: number;
  risk: {
    movement_risk: number; sensor_risk: number; vision_risk: number;
    fused_risk: number; smoothed_risk: number; overall_risk_pct: number;
    risk_level: "LOW" | "MODERATE" | "HIGH";
    alert_triggered: boolean; trend: string;
    weights: { movement: number; sensor: number; vision: number };
  };
  sensors: {
    heart_rate_bpm: number; oxygen_saturation_spo2: number;
    skin_temperature_celsius: number; muscle_activation_emg: number;
    training_load: number; hydration_level: number;
    stress_index: number; sensor_fatigue_index: number; fatigue_level: number;
  };
  joints: {
    l_knee: number; r_knee: number; l_hip: number; r_hip: number;
    l_ankle: number; r_ankle: number; movement_asymmetry: number;
    postural_instability_index: number; angular_velocity: number;
    ground_reaction_force: number;
  };
  performance: {
    fps: number; pose_latency_ms: number; ml_latency_ms: number;
    fusion_latency_ms: number; e2e_latency_ms: number;
  };
}

interface Ctx {
  frame: LiveFrame | null;
  history: LiveFrame[];
  connected: boolean;
  demo: boolean;
  athleteId: string;
  setAthleteId: (id: string) => void;
  alerts: Alert[];
}

interface Alert {
  id: string; athlete_id: string; level: string;
  risk_score: number; reasons: string[]; timestamp: number;
  acknowledged: boolean; demo: boolean;
}

const LiveCtx = createContext<Ctx>({
  frame: null, history: [], connected: false, demo: true,
  athleteId: "07", setAthleteId: () => {}, alerts: [],
});

export function LiveDataProvider({ children }: { children: React.ReactNode }) {
  const [frame, setFrame]       = useState<LiveFrame | null>(null);
  const [history, setHistory]   = useState<LiveFrame[]>([]);
  const [connected, setConn]    = useState(false);
  const [athleteId, setAthlete] = useState("07");
  const [alerts, setAlerts]     = useState<Alert[]>([]);
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;
    const ws = new WebSocket(`${WS}/ws/monitor/${athleteId}`);
    wsRef.current = ws;

    ws.onopen  = () => setConn(true);
    ws.onclose = () => {
      setConn(false);
      retryRef.current = setTimeout(connect, 3000);
    };
    ws.onerror = () => ws.close();
    ws.onmessage = (e) => {
      try {
        const data: LiveFrame = JSON.parse(e.data);
        setFrame(data);
        setHistory(h => [...h.slice(-59), data]);
        // Poll alerts
        fetch(`${API}/api/alerts?athlete_id=${athleteId}&last_n=20`)
          .then(r => r.json()).then(setAlerts).catch(() => {});
      } catch {}
    };
  }, [athleteId]);

  useEffect(() => {
    if (retryRef.current) clearTimeout(retryRef.current);
    wsRef.current?.close();
    connect();
    return () => {
      wsRef.current?.close();
      if (retryRef.current) clearTimeout(retryRef.current);
    };
  }, [connect]);

  return (
    <LiveCtx.Provider value={{
      frame, history, connected, demo: frame?.demo ?? true,
      athleteId, setAthleteId: setAthlete, alerts,
    }}>
      {children}
    </LiveCtx.Provider>
  );
}

export const useLive = () => useContext(LiveCtx);
