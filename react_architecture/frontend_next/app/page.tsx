import Link from "next/link";
import type { CSSProperties } from "react";

const cardStyle: CSSProperties = {
  border: "1px solid #cbd5e1",
  borderRadius: 12,
  padding: 16,
  background: "#ffffff",
  marginBottom: 12
};

export default function HomePage() {
  return (
    <div>
      <h1>React 架构学习首页</h1>
      <p>这个页面用于对比不同架构下数据获取和渲染时机。</p>

      <section style={cardStyle}>
        <h2>MPA（概念）</h2>
        <p>每次跳转由后端返回新 HTML，浏览器整页刷新。</p>
      </section>

      <section style={cardStyle}>
        <h2>SPA(CSR)</h2>
        <p>客户端组件在浏览器发起 API 请求并更新页面。</p>
        <Link href="/csr">进入 /csr 示例</Link>
      </section>

      <section style={cardStyle}>
        <h2>SSR</h2>
        <p>服务端在请求时获取数据并渲染 HTML。</p>
        <Link href="/ssr">进入 /ssr 示例</Link>
      </section>

      <section style={cardStyle}>
        <h2>SSG</h2>
        <p>构建时生成静态数据快照并渲染页面。</p>
        <Link href="/ssg">进入 /ssg 示例</Link>
      </section>
    </div>
  );
}
