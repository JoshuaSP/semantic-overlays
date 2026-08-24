import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Semantic Overlays",
  description:
    "Tiny adapters added to a frozen language model. They only fire at marked token positions, and they change how the model perceives those spans.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
