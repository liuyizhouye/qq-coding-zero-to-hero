export type OrderItem = {
  sku: string;
  qty: number;
  unit_price: number;
};

export type OrderSummaryRequest = {
  items: OrderItem[];
  tax_rate: number;
};

export type OrderSummaryResponse = {
  subtotal: number;
  tax_amount: number;
  total: number;
  currency: string;
};

export const DEMO_ORDER_REQUEST: OrderSummaryRequest = {
  items: [
    { sku: "Keyboard", qty: 2, unit_price: 49.5 },
    { sku: "Mouse", qty: 1, unit_price: 25.0 }
  ],
  tax_rate: 0.1
};

export const BROWSER_API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://127.0.0.1:8000";
const SERVER_API_BASE_URL = process.env.API_BASE_URL ?? BROWSER_API_BASE_URL;

async function parseSummaryResponse(response: Response): Promise<OrderSummaryResponse> {
  const data = (await response.json()) as Partial<OrderSummaryResponse> & {
    detail?: string;
  };

  if (!response.ok) {
    throw new Error(data.detail ?? `request failed (${response.status})`);
  }

  return {
    subtotal: Number(data.subtotal ?? 0),
    tax_amount: Number(data.tax_amount ?? 0),
    total: Number(data.total ?? 0),
    currency: String(data.currency ?? "USD")
  };
}

export async function fetchOrderSummaryFromBrowser(
  payload: OrderSummaryRequest = DEMO_ORDER_REQUEST
): Promise<OrderSummaryResponse> {
  const response = await fetch(`${BROWSER_API_BASE_URL}/api/orders/summary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  return parseSummaryResponse(response);
}

export async function fetchOrderSummaryFromServer(
  payload: OrderSummaryRequest = DEMO_ORDER_REQUEST
): Promise<OrderSummaryResponse> {
  const response = await fetch(`${SERVER_API_BASE_URL}/api/orders/summary`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store"
  });

  return parseSummaryResponse(response);
}
