"use client";

import Link from "next/link";
import type { CSSProperties } from "react";
import { useEffect, useState } from "react";

import {
  BROWSER_API_BASE_URL,
  DEMO_ORDER_REQUEST,
  fetchOrderSummaryFromBrowser,
  type OrderSummaryResponse
} from "../../lib/api";

const panelStyle: CSSProperties = {
  border: "1px solid #cbd5e1",
  borderRadius: 12,
  background: "#ffffff",
  padding: 16
};

export default function CSRPage() {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [summary, setSummary] = useState<OrderSummaryResponse | null>(null);

  async function refresh() {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchOrderSummaryFromBrowser(DEMO_ORDER_REQUEST);
      setSummary(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  return (
    <div>
      <h1>CSR 页面</h1>
      <p>数据在浏览器侧加载。打开 DevTools Network 可以看到请求发生在客户端。</p>
      <p>API Base URL: {BROWSER_API_BASE_URL}</p>

      <div style={panelStyle}>
        <pre>{JSON.stringify(DEMO_ORDER_REQUEST, null, 2)}</pre>
      </div>

      <button onClick={() => void refresh()} style={{ marginTop: 12, marginBottom: 12 }}>
        重新请求
      </button>

      {loading && <p>加载中...</p>}
      {error && <p style={{ color: "#b91c1c" }}>请求失败：{error}</p>}
      {!loading && !error && (
        <div style={panelStyle}>
          <h2>响应数据</h2>
          <pre>{JSON.stringify(summary, null, 2)}</pre>
        </div>
      )}

      <p style={{ marginTop: 20 }}>
        <Link href="/">返回首页</Link>
      </p>
    </div>
  );
}
