import type { Metadata } from "next";
import "./globals.css";

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
    <html lang="en">
      <body className="min-h-screen bg-[#0B0C0E] text-[#F3F4F6] antialiased selection:bg-[#D4AF37] selection:text-[#0B0C0E]">
        {children}
      </body>
    </html>
  );
}
