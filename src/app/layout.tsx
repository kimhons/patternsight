import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";
import { AuthProvider } from "@/components/ui/AuthContextFixed";
import Navigation from "@/components/layout/Navigation";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "PatternSight - Advanced Pattern Intelligence Cloud Platform",
  description: "PatternSight is the world's first AI-powered pattern recognition cloud platform combining deep learning, advanced analytics, and intelligent insights. Discover hidden patterns in your data with multiple AI providers.",
  keywords: "pattern recognition, AI analysis, data analytics, machine learning, cloud platform, business intelligence",
  authors: [{ name: "PatternSight Team" }],
  openGraph: {
    title: "PatternSight - Where Mathematics Meets Possibility",
    description: "Advanced AI-powered pattern recognition platform for data analysis and insights",
    type: "website",
    siteName: "PatternSight"
  },
  twitter: {
    card: "summary_large_image",
    title: "PatternSight - Advanced Pattern Intelligence",
    description: "AI-powered pattern recognition cloud platform"
  }
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <AuthProvider>
          <Navigation />
          {children}
        </AuthProvider>
      </body>
    </html>
  );
}
