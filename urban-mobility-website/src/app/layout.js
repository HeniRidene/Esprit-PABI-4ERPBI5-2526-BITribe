import { Inter } from "next/font/google";
import "./globals.css";

const inter = Inter({
  subsets: ["latin"],
  weight: ["400", "500", "600", "700", "900"],
  variable: "--font-inter",
});

export const metadata = {
  title: "UrbanMobility BI — Intelligent Urban Mobility Suite",
  description:
    "Professional executive dashboard for urban transport authorities — monitoring transit performance, sustainability, and safety in real-time.",
};

export default function RootLayout({ children }) {
  return (
    <html lang="fr" className={`${inter.variable} h-full`}>
      <body className="min-h-full" style={{ fontFamily: "'Inter', sans-serif" }}>
        {children}
      </body>
    </html>
  );
}
