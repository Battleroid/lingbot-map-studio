import "./globals.css";

import type { Metadata } from "next";
import localFont from "next/font/local";
import { Roboto_Mono } from "next/font/google";
import type { ReactNode } from "react";

import { Providers } from "@/components/Providers";

export const metadata: Metadata = {
  title: "lingbot-map studio",
  description: "browser-driven 3D reconstruction",
};

const useBerkeley = process.env.NEXT_PUBLIC_USE_BERKELEY_MONO === "1";

const berkeleyMono = useBerkeley
  ? localFont({
      src: "../../public/fonts/berkeley-mono.woff2",
      weight: "100 900",
      display: "swap",
      variable: "--font-mono",
    })
  : null;

const robotoMono = Roboto_Mono({
  subsets: ["latin"],
  weight: ["300", "400", "500", "700"],
  display: "swap",
  variable: "--font-mono",
});

const fontClass = (berkeleyMono ?? robotoMono).variable;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={fontClass}>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
