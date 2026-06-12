import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Gurbani Guidance",
  description:
    "Ask questions grounded in Sri Guru Granth Sahib Ji — the eternal Guru of the Sikhs.",
  keywords: ["Gurbani", "SGGS", "Sikh", "scripture", "Guru Granth Sahib"],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link
          rel="preconnect"
          href="https://fonts.gstatic.com"
          crossOrigin="anonymous"
        />
      </head>
      <body className="min-h-screen bg-stone-50 text-stone-900 antialiased">
        {children}
      </body>
    </html>
  );
}
