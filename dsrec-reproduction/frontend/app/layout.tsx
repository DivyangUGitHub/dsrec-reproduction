import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DSRec — Intelligent Recommendations",
  description: "A production-style recommendation experience powered by DSRec.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
