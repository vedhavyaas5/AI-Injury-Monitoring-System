import type { Metadata } from "next";
import "./globals.css";
import { Sidebar } from "@/components/Sidebar";
import { TopNav } from "@/components/TopNav";
import { LiveDataProvider } from "@/components/LiveDataProvider";

export const metadata: Metadata = {
  title: "Sports AI Monitor",
  description: "Real-Time AI-Based Sports Injury Risk Monitoring System",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="flex h-screen overflow-hidden" style={{ background: "var(--bg-base)" }}>
        <LiveDataProvider>
          <Sidebar />
          <div className="flex flex-col flex-1 min-w-0">
            <TopNav />
            <main className="flex-1 overflow-y-auto p-4 lg:p-6">
              {children}
            </main>
          </div>
        </LiveDataProvider>
      </body>
    </html>
  );
}
