import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Vajra — Browser-Native WebGPU Kernel Auto-Tuner",
  description:
    "Tunes per-device on any modern WebGPU browser, and falls back to recorded Preview where WebGPU is unavailable.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} min-h-screen bg-[#FAFAFA] text-[#171717] font-sans antialiased selection:bg-[#EAEAEA] selection:text-[#171717]`}
      >
        {children}
      </body>
    </html>
  );
}
