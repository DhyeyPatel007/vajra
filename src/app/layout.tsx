import type { Metadata } from "next";
import { Plus_Jakarta_Sans, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const plusJakartaSans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  variable: "--font-sans",
  weight: ["400", "500", "600", "700"],
});

const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
  weight: ["400", "500", "600", "700"],
});

export const metadata: Metadata = {
  title: "Vajra — Browser-Native WebGPU Kernel Auto-Tuner",
  description:
    "AI writes GPU shaders, but only your hardware knows the truth. Vajra benchmarks every Nemotron WGSL kernel live on your GPU and crowns the fastest.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={`${plusJakartaSans.variable} ${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-[#0F1419] text-[#F1F5F9] font-sans antialiased selection:bg-[#F59E0B] selection:text-[#0F1419]">
        {children}
      </body>
    </html>
  );
}
