import Link from "next/link";
import type { CSSProperties } from "react";

import snapshot from "../../data/ssg_snapshot.json";

export const dynamic = "force-static";

const panelStyle: CSSProperties = {
  border: "1px solid #cbd5e1",
  borderRadius: 12,
  background: "#ffffff",
  padding: 16
};

export default function SSGPage() {
  return (
    <div>
      <h1>SSG 页面</h1>
      <p>页面使用构建期快照，首屏无需实时请求 API。</p>

      <div style={panelStyle}>
        <h2>构建快照</h2>
        <pre>{JSON.stringify(snapshot, null, 2)}</pre>
      </div>

      <p style={{ marginTop: 20 }}>
        <Link href="/">返回首页</Link>
      </p>
    </div>
  );
}
