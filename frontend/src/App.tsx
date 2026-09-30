import { useEffect, useState, ReactNode } from "react";
import { ask, search, getConfigs, AskResponse } from "./api";

const SAMPLES = ["Is the protein Papilin secreted?", "What is the mechanism of action of bedaquiline?",
  "Which drugs are used for treatment of castration-resistant prostate cancer?"];

function renderAnswer(text: string, onCite: (p: number) => void): ReactNode[] {
  return text.split(/(\[PMID:[^\]]+\])/g).map((part, i) => {
    const m = part.match(/^\[PMID:([^\]]+)\]$/);
    if (!m) return <span key={i}>{part}</span>;
    return <span key={i}>{(m[1].match(/\d+/g) ?? []).map((id) =>
      <button key={id} className="cite" onClick={() => onCite(Number(id))}>PMID:{id}</button>)}</span>;
  });
}

export default function App() {
  const [q, setQ] = useState(""); const [configs, setConfigs] = useState<string[]>(["hybrid"]);
  const [config, setConfig] = useState("hybrid"); const [data, setData] = useState<AskResponse | null>(null);
  const [loading, setLoading] = useState(false); const [err, setErr] = useState(""); const [hl, setHl] = useState<number | null>(null);
  const [withAnswer, setWithAnswer] = useState(true);

  useEffect(() => { getConfigs().then((c) => { setConfigs(c); if (c.includes("best")) setConfig("best"); }).catch(() => {}); }, []);

  async function run(query = q) {
    if (query.trim().length < 3) { setErr("Enter a biomedical question (at least 3 characters)."); return; }
    setLoading(true); setErr(""); setData(null); setHl(null);
    try { setData(await (withAnswer ? ask : search)(query.trim(), config)); }
    catch (e: any) { setErr(String(e.message ?? e)); } finally { setLoading(false); }
  }
  function goTo(pmid: number) {
    setHl(pmid); document.getElementById(`hit-${pmid}`)?.scrollIntoView({ behavior: "smooth", block: "center" });
  }
  const a = data?.answer;
  return (
    <main>
      <header><h1>Biomedical Hybrid Search</h1>
        <p className="muted">Research demo over rag-mini-bioasq. Not medical advice.</p></header>
      <form onSubmit={(e) => { e.preventDefault(); run(); }} className="bar">
        <input value={q} onChange={(e) => setQ(e.target.value)} maxLength={500} placeholder="Ask a biomedical question…" aria-label="Question" />
        <select value={config} onChange={(e) => setConfig(e.target.value)} aria-label="Retrieval mode">
          {configs.map((c) => <option key={c}>{c}</option>)}</select>
        <button disabled={loading}>{loading ? "Searching…" : "Search"}</button>
      </form>
      <label className="muted"><input type="checkbox" checked={withAnswer} onChange={(e) => setWithAnswer(e.target.checked)} /> Generate answer (uses LLM)</label>
      <div className="chips">{SAMPLES.map((s) => <button key={s} className="chip" onClick={() => { setQ(s); run(s); }}>{s}</button>)}</div>
      {err && <div className="banner err">{err}</div>}
      {data && <>
        <section><h2>Query</h2><p><b>{data.query}</b></p>
          <div className="chips">{data.expansions.length ? data.expansions.map((e) => <span className="chip static" key={e}>{e}</span>)
            : <span className="muted">No expansions for this mode.</span>}</div></section>
        {a && (a.status === "insufficient_evidence"
          ? <section className="banner warn"><h2>Insufficient evidence</h2>
              <p>The retrieved passages do not contain enough information to answer this question.
              {a.reason && <span className="muted"> ({a.reason})</span>}</p>{a.answer && <p className="muted">{a.answer}</p>}</section>
          : <section><h2>Answer</h2><p className="answer">{renderAnswer(a.answer, goTo)}</p>
              <p className="muted">{a.valid.length} valid citation(s), {a.invalid.length} invalid removed.</p></section>)}
        <section><h2>Evidence ({data.results.length})</h2>
          {data.results.map((h) => (
            <article key={h.pmid} id={`hit-${h.pmid}`} className={"hit" + (hl === h.pmid ? " hl" : "")}>
              <div className="row"><b>#{h.rank}</b>
                <a href={h.source_url} target="_blank" rel="noreferrer">PMID:{h.pmid}</a>
                {a?.status === "answered" && a.valid.includes(h.pmid)
                  ? <span className="tag">cited in answer</span>
                  : a?.status === "answered" && h.selected_for_context && <span className="tag muted">sent to model, not cited</span>}
                <span className="muted">fused {h.score.toFixed(4)}
                  {h.lexical && ` · BM25 #${h.lexical[0]} (${h.lexical[1].toFixed(1)})`}
                  {h.dense && ` · dense #${h.dense[0]} (${h.dense[1].toFixed(3)})`}</span></div>
              <p>{h.snippet}</p></article>))}
        </section>
        <footer className="muted">config {data.config_hash} · {Object.entries(data.timings_ms).map(([k, v]) => `${k} ${v.toFixed(0)}ms`).join(" · ")}
          {a?.usage?.cost_usd != null && ` · answer cost $${a.usage.cost_usd.toFixed(5)}`}</footer>
      </>}
    </main>
  );
}
