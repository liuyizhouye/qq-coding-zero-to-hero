import type { CSSProperties, ReactNode } from "react";

export const metadata = {
  title: "React Architecture Demo",
  description: "CSR / SSR / SSG learning project"
};

const pageStyle: CSSProperties = {
  fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
  margin: 0,
  background: "linear-gradient(180deg, #eef6ff 0%, #f8fafc 100%)",
  color: "#0f172a"
};

const containerStyle: CSSProperties = {
  maxWidth: "960px",
  margin: "0 auto",
  padding: "24px"
};

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="zh-CN">
      <body style={pageStyle}>
        <main style={containerStyle}>{children}</main>
      </body>
    </html>
  );
}
