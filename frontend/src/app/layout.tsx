import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SafeSight | PPE Safety Dashboard",
  description: "Real-time AI-powered PPE compliance monitoring system.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
