import type { Metadata } from "next";
import { headers } from "next/headers";
import "./globals.css";

export async function generateMetadata(): Promise<Metadata> {
  const incoming = await headers();
  const host = incoming.get("x-forwarded-host") || incoming.get("host") || "localhost:3000";
  const protocol = incoming.get("x-forwarded-proto") || (host.startsWith("localhost") ? "http" : "https");
  const socialImage = `${protocol}://${host}/og.png`;
  const description = "Interactive wildfire monitoring using existing cameras, multi-frame confirmation, approximate mapping, and operator-guided response.";

  return {
    title: "PyroFinder — Live Operations Prototype",
    description,
    icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
    openGraph: { title: "PyroFinder — Live Operations Prototype", description, images: [{ url: socialImage, width: 1732, height: 910 }] },
    twitter: { card: "summary_large_image", title: "PyroFinder — Live Operations Prototype", description, images: [socialImage] },
  };
}

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
