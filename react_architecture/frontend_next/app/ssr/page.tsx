import Link from "next/link";
import type { CSSProperties } from "react";

import { DEMO_ORDER_REQUEST, fetchOrderSummaryFromServer } from "../../lib/api";

export const dynamic = "force-dynamic";

const panelStyle: CSSProperties = {
  border: "1px solid #cbd5e1",
  borderRadius: 12,
  background: "#ffffff",
  padding: 16
};

export default async function SSRPage() {
  let summary: object | null = null;
  let error: string | null = null;

  try {
    summary = await fetchOrderSummaryFromServer(DEMO_ORDER_REQUEST);
  } catch (err) {
    error = err instanceof Error ? err.message : String(err);
  }

  return (
    <div>
      <h1>SSR 页面</h1>
      <p>数据在服务器渲染阶段获取，然后返回已经带数据的 HTML。</p>

      <div style={panelStyle}>
        <h2>请求体</h2>
        <pre>{JSON.stringify(DEMO_ORDER_REQUEST, null, 2)}</pre>
      </div>

      <div style={{ marginTop: 12, ...panelStyle }}>
        <h2>响应结果</h2>
        {error ? <p style={{ color: "#b91c1c" }}>请求失败：{error}</p> : <pre>{JSON.stringify(summary, null, 2)}</pre>}
      </div>

      <p style={{ marginTop: 20 }}>
        <Link href="/">返回首页</Link>
      </p>
    </div>
  );
}
