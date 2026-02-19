# react_architecture 模块

这个模块的目标是让你看清“同一个业务需求，放在不同网页架构里，数据流和渲染时机为什么不同”。

## 为什么要比较架构

很多初学者一开始只学 React 语法，但不知道页面为什么慢、SEO 为什么差、首屏为什么不稳定。根因通常在架构层，而不是组件语法层。

本模块聚焦四种基础形态：

- `MPA`（多页应用）
- `SPA(CSR)`（单页客户端渲染）
- `SSR`（服务端渲染）
- `SSG`（静态生成）

## 核心差异（背后原理）

### 1) 渲染时机

- MPA：每次跳转都由后端生成新页面。
- CSR：浏览器加载 JS 后再渲染。
- SSR：每次请求都在服务端先渲染 HTML。
- SSG：构建时就生成好 HTML，线上直接返回静态文件。

### 2) 数据获取时机

- MPA/SSR：通常在服务器阶段拿数据。
- CSR：通常在浏览器端请求 API。
- SSG：通常在构建阶段拿数据（或通过再生成策略更新）。

### 3) 体验与工程权衡

- CSR 交互灵活，但首屏和 SEO 需要补偿。
- SSR 首屏与 SEO 通常更稳，但服务器压力更高。
- SSG 性能极佳，但数据实时性需要额外策略。

## 代码实现映射

### Python 教学组件（`component.py`）

- `build_architecture_matrix()`：返回结构化对比矩阵。
- `run_demo()`：输出可读 `trace`，明确每种架构的数据流步骤。

固定事件顺序：

1. `architecture_matrix_built`
2. `mpa_flow`
3. `spa_csr_flow`
4. `ssr_flow`
5. `ssg_flow`
6. `beginner_recommendation`
7. `model_final_answer`

### Next.js 实战工程（`frontend_next/`）

- `/csr`：Client Component，在浏览器 `fetch` Python API。
- `/ssr`：Server Component，在请求时拉 Python API。
- `/ssg`：读取构建期快照 `data/ssg_snapshot.json`。

## 如何与 Python API 连接

本模块前端默认读取：

- `NEXT_PUBLIC_API_BASE_URL`（浏览器端）
- `API_BASE_URL`（服务端）

可在 `frontend_next/.env.local` 配置：

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
API_BASE_URL=http://127.0.0.1:8000
```

## 运行方式

### 1) 先启动 Python API（终端 A）

```bash
python web_data_flow/component.py --mode serve --host 127.0.0.1 --port 8000
```

### 2) 启动 Next.js 前端（终端 B）

```bash
cd react_architecture/frontend_next
npm install
npm run dev
```

浏览器访问：

- `http://localhost:3000/`
- `http://localhost:3000/csr`
- `http://localhost:3000/ssr`
- `http://localhost:3000/ssg`

### 3) 刷新 SSG 快照（可选）

```bash
npm run refresh:ssg
```

## 常见错误与排查

- 页面提示 `fetch failed`
原因：Python API 没启动或端口不一致。
排查：确认 `web_data_flow` 服务已运行，且 URL 与 `.env.local` 一致。

- CSR 页面有 CORS 错误
原因：浏览器跨域请求被拒绝。
排查：确认后端 `create_app()` 已启用 CORS，且来源是 `localhost:3000`。

- SSG 页面数据没变化
原因：SSG 使用的是构建快照。
排查：先执行 `npm run refresh:ssg`，再重新构建或重启。

## 建议学习路径

1. 先跑 `python react_architecture/component.py` 看总体对比。
2. 再跑 Next.js 三个页面，观察真实差异。
3. 最后回到业务问题：按 SEO、首屏、交互、实时性做架构选择。
