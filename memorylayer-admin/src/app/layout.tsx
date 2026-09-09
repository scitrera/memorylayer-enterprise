// SPDX-FileCopyrightText: 2026 Scitrera LLC
// SPDX-License-Identifier: AGPL-3.0-only

import type { Metadata } from "next";
import { DM_Sans, JetBrains_Mono, Instrument_Serif } from "next/font/google";
import { Toaster } from "sonner";
import { QueryProvider } from "@/providers/query-provider";
import { ConnectionProvider } from "@/providers/connection-provider";
import { WorkspaceProvider } from "@/providers/workspace-provider";
import { AdminShell } from "@/components/layout/admin-shell";
import "./globals.css";

const dmSans = DM_Sans({
  subsets: ["latin"],
  variable: "--font-dm-sans",
  display: "swap",
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-jetbrains-mono",
  display: "swap",
});

const instrumentSerif = Instrument_Serif({
  subsets: ["latin"],
  weight: "400",
  variable: "--font-instrument-serif",
  display: "swap",
});

export const metadata: Metadata = {
  title: "MemoryLayer Admin",
  description: "Enterprise admin dashboard for MemoryLayer",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html
      lang="en"
      className={`${dmSans.variable} ${jetbrainsMono.variable} ${instrumentSerif.variable}`}
    >
      <body className="font-sans antialiased">
        <QueryProvider>
          <ConnectionProvider>
            <WorkspaceProvider>
              <AdminShell>{children}</AdminShell>
              <Toaster position="bottom-right" />
            </WorkspaceProvider>
          </ConnectionProvider>
        </QueryProvider>
      </body>
    </html>
  );
}
