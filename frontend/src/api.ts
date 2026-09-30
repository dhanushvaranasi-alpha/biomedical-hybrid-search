export interface Hit {
  rank: number; pmid: number; score: number;
  lexical: [number, number] | null; dense: [number, number] | null;
  snippet: string; source_url: string; selected_for_context?: boolean;
}
export interface AnswerInfo {
  status: "answered" | "insufficient_evidence"; reason: string | null; answer: string;
  valid: number[]; invalid: number[]; valid_rate: number;
  usage?: { prompt_tokens?: number; completion_tokens?: number; cost_usd?: number; latency_ms?: number };
}
export interface AskResponse {
  query: string; expansions: string[]; config_hash: string; results: Hit[];
  timings_ms: Record<string, number>; answer?: AnswerInfo;
}
const BASE = (import.meta as any).env?.VITE_API_BASE_URL ?? "";
async function post(path: string, body: unknown): Promise<any> {
  const r = await fetch(`${BASE}${path}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error((await r.json().catch(() => ({ detail: r.statusText }))).detail ?? r.statusText);
  return r.json();
}
export const search = (query: string, config: string): Promise<AskResponse> => post("/api/search", { query, config });
export const ask = (query: string, config: string): Promise<AskResponse> => post("/api/ask", { query, config });
export const getConfigs = async (): Promise<string[]> => (await fetch(`${BASE}/api/configs`)).json();
