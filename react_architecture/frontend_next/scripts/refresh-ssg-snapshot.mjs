import { writeFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const API_BASE_URL = process.env.API_BASE_URL ?? process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";

const payload = {
  items: [
    { sku: "Keyboard", qty: 2, unit_price: 49.5 },
    { sku: "Mouse", qty: 1, unit_price: 25.0 }
  ],
  tax_rate: 0.1
};

async function main() {
  const response = await fetch(`${API_BASE_URL}/api/orders/summary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    const body = await response.text();
    throw new Error(`refresh failed (${response.status}): ${body}`);
  }

  const summary = await response.json();
  const snapshot = {
    generated_at: new Date().toISOString(),
    source: "refresh-ssg-snapshot",
    request: payload,
    response: summary
  };

  const currentFile = fileURLToPath(import.meta.url);
  const dataFile = path.resolve(path.dirname(currentFile), "../data/ssg_snapshot.json");
  await writeFile(dataFile, `${JSON.stringify(snapshot, null, 2)}\n`, "utf-8");

  console.log(`updated: ${dataFile}`);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
