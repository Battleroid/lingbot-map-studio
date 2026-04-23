import "./globals.css";

import fs from "node:fs";
import path from "node:path";

import type { Metadata } from "next";
import { Roboto_Mono } from "next/font/google";
import type { ReactNode } from "react";

import { Providers } from "@/components/Providers";

export const metadata: Metadata = {
  title: "vid3d studio",
  description: "browser-driven 3D reconstruction",
};

const robotoMono = Roboto_Mono({
  subsets: ["latin"],
  weight: ["300", "400", "500", "700"],
  display: "swap",
  variable: "--font-roboto-mono",
});

const hasBerkeleyMono = (() => {
  try {
    return fs.existsSync(
      path.join(process.cwd(), "public", "fonts", "berkeley-mono.woff2"),
    );
  } catch {
    return false;
  }
})();

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" className={robotoMono.variable}>
      <head>
        {hasBerkeleyMono && (
          <link rel="stylesheet" href="/fonts/berkeley-mono.css" />
        )}
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
