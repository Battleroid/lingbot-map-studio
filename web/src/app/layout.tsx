import "./globals.css";

import type { Metadata } from "next";
import { Roboto_Mono } from "next/font/google";
import type { ReactNode } from "react";

import { Providers } from "@/components/Providers";

export const metadata: Metadata = {
  title: "lingbot-map studio",
  description: "browser-driven 3D reconstruction",
};

const robotoMono = Roboto_Mono({
  subsets: ["latin"],
  weight: ["300", "400", "500", "700"],
  display: "swap",
  variable: "--font-roboto-mono",
});

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={robotoMono.variable}>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
