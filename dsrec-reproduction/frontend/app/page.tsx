"use client";

import { FormEvent, useState } from "react";

type Recommendation = { item_id: number; score: number };
type ApiResponse = { user_id: number; recommendations: Recommendation[]; model_version: string };
type Feedback = "like" | "dislike" | "hide" | "click";

export default function Home() {
  const [userId, setUserId] = useState("1");
  const [topK, setTopK] = useState("10");
  const [data, setData] = useState<ApiResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [feedback, setFeedback] = useState<Record<number, Feedback>>({});

  async function sendInteraction(
    eventType: Feedback,
    item: Recommendation,
    rank: number,
  ) {
    try {
      const response = await fetch("/v1/interactions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          user_id: Number(userId),
          event_type: eventType,
          item_id: item.item_id,
          recommendation_rank: rank,
          recommendation_score: item.score,
          metadata: { model_version: data?.model_version ?? "unknown" },
        }),
      });

      const body = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(
          body.detail ?? `Feedback request failed (${response.status})`,
        );
      }

      setFeedback((current) => ({ ...current, [item.item_id]: eventType }));
      setError("");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Could not record feedback.",
      );
    }
  }

  async function getRecommendations(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setFeedback({});
    try {
      const response = await fetch("/v1/recommend", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ user_id: Number(userId), top_k: Number(topK) }),
      });
      const body = await response.json();
      if (!response.ok) throw new Error(body.detail ?? `Request failed (${response.status})`);
      setData(body);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Could not load recommendations.");
      setData(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="min-h-screen px-5 py-8 sm:px-8 lg:px-12">
      <div className="mx-auto max-w-6xl">
        <header className="mb-10 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="grid size-11 place-items-center rounded-2xl bg-indigo-500 text-lg font-black shadow-lg shadow-indigo-500/25">D</div>
            <div><p className="text-sm font-bold tracking-wide text-white">DSRec</p><p className="text-xs text-slate-500">Recommendation intelligence</p></div>
          </div>
          <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-3 py-1 text-xs font-semibold text-emerald-300">● API online</span>
        </header>

        <section className="mb-8 grid gap-8 lg:grid-cols-[1.25fr_.75fr] lg:items-end">
          <div>
            <p className="mb-3 text-sm font-bold uppercase tracking-[.2em] text-indigo-300">Personalization engine</p>
            <h1 className="max-w-3xl text-4xl font-black leading-tight tracking-[-.04em] sm:text-6xl">Turn a user into their next <span className="text-indigo-300">best recommendation.</span></h1>
            <p className="mt-5 max-w-2xl text-base leading-7 text-slate-400 sm:text-lg">A live interface for your DSRec model. Choose a user, request Top-K items, and give feedback so the system can learn from real interactions.</p>
          </div>
          <div className="rounded-3xl border border-white/10 bg-white/[.04] p-5 backdrop-blur">
            <p className="text-xs font-bold uppercase tracking-widest text-slate-500">System</p>
            <div className="mt-4 grid grid-cols-2 gap-3 text-sm"><div className="rounded-2xl bg-black/20 p-3"><span className="text-slate-500">Backend</span><p className="mt-1 font-bold">FastAPI</p></div><div className="rounded-2xl bg-black/20 p-3"><span className="text-slate-500">Storage</span><p className="mt-1 font-bold">PostgreSQL</p></div></div>
          </div>
        </section>

        <section className="rounded-3xl border border-white/10 bg-white/[.045] p-5 shadow-2xl shadow-black/20 backdrop-blur sm:p-7">
          <form onSubmit={getRecommendations} className="grid gap-4 md:grid-cols-[1fr_180px_auto] md:items-end">
            <label className="block"><span className="mb-2 block text-sm font-semibold text-slate-300">User ID</span><input value={userId} onChange={(e) => setUserId(e.target.value)} type="number" min="1" required className="w-full rounded-2xl border border-white/10 bg-black/25 px-4 py-3.5 text-white outline-none ring-indigo-400 transition focus:ring-2" /></label>
            <label className="block"><span className="mb-2 block text-sm font-semibold text-slate-300">Top-K</span><input value={topK} onChange={(e) => setTopK(e.target.value)} type="number" min="1" max="100" required className="w-full rounded-2xl border border-white/10 bg-black/25 px-4 py-3.5 text-white outline-none ring-indigo-400 transition focus:ring-2" /></label>
            <button disabled={loading} className="rounded-2xl bg-indigo-500 px-7 py-3.5 font-bold text-white shadow-lg shadow-indigo-500/20 transition hover:bg-indigo-400 disabled:cursor-wait disabled:opacity-60">{loading ? "Running model…" : "Get recommendations →"}</button>
          </form>
          {error && <p className="mt-4 rounded-2xl border border-red-400/20 bg-red-400/10 px-4 py-3 text-sm text-red-300">{error}</p>}
        </section>

        <section className="mt-8">
          <div className="mb-4 flex items-end justify-between gap-4"><div><p className="text-xs font-bold uppercase tracking-widest text-slate-500">Inference results</p><h2 className="mt-1 text-2xl font-bold">{data ? `For user ${data.user_id}` : "Ready for a user"}</h2></div>{data && <span className="rounded-full border border-white/10 bg-white/[.04] px-3 py-1 text-xs text-slate-400">{data.model_version}</span>}</div>
          {!data ? <div className="rounded-3xl border border-dashed border-white/10 px-6 py-16 text-center text-slate-500">Enter a user ID above to run the real DSRec inference pipeline.</div> : <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-5">{data.recommendations.map((item, index) => { const state = feedback[item.item_id]; return <article key={`${item.item_id}-${index}`} className="group rounded-3xl border border-white/10 bg-white/[.035] p-5 transition hover:-translate-y-1 hover:border-indigo-400/30 hover:bg-indigo-400/[.06]"><div className="flex items-center justify-between"><span className="text-xs font-bold text-indigo-300">#{String(index + 1).padStart(2, "0")}</span><span className="text-xs text-slate-600">ITEM</span></div><p className="mt-7 text-2xl font-black">{item.item_id}</p><div className="mt-5 h-1.5 overflow-hidden rounded-full bg-white/5"><div className="h-full rounded-full bg-indigo-400" style={{ width: `${Math.min(100, Math.max(8, item.score * 15))}%` }} /></div><p className="mt-3 text-sm text-slate-400">Score <span className="font-bold text-slate-200">{item.score.toFixed(4)}</span></p><div className="mt-4 grid grid-cols-3 gap-2"><button onClick={() => sendInteraction("like", item, index + 1)} className={`rounded-xl px-2 py-2 text-xs font-bold ${state === "like" ? "bg-emerald-400/20 text-emerald-300" : "bg-white/5 text-slate-400 hover:bg-emerald-400/10 hover:text-emerald-300"}`}>Like</button><button onClick={() => sendInteraction("dislike", item, index + 1)} className={`rounded-xl px-2 py-2 text-xs font-bold ${state === "dislike" ? "bg-amber-400/20 text-amber-300" : "bg-white/5 text-slate-400 hover:bg-amber-400/10 hover:text-amber-300"}`}>Skip</button><button onClick={() => sendInteraction("click", item, index + 1)} className={`rounded-xl px-2 py-2 text-xs font-bold ${state === "click" ? "bg-indigo-400/20 text-indigo-300" : "bg-white/5 text-slate-400 hover:bg-indigo-400/10 hover:text-indigo-300"}`}>Open</button></div></article>; })}</div>}
        </section>

        <footer className="mt-14 border-t border-white/5 py-6 text-center text-xs text-slate-600">DSRec · Next.js + React · FastAPI · PostgreSQL interaction tracking</footer>
      </div>
    </main>
  );
}
