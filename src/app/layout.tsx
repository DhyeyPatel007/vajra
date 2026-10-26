import type { Metadata } from "next";
import { Space_Grotesk, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const spaceGrotesk = Space_Grotesk({
  subsets: ["latin"],
  variable: "--font-sans",
  weight: ["500", "600", "700"],
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
    <html lang="en" className={`${spaceGrotesk.variable} ${jetbrainsMono.variable}`}>
      <body className="min-h-screen bg-[#07080A] text-[#F4F6F8] font-sans antialiased selection:bg-[#E5B842] selection:text-[#07080A]">
        {children}
      </body>
    </html>
  );
}
