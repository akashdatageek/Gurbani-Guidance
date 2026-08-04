> **Maintainer status notes (2026-08)** — this external plan was reviewed
> against the codebase; Stage-0 items implementable in-repo are DONE:
> ✅ streaming SSE (was listed missing) · ✅ /health with passage counts +
> /ready · ✅ threadpool endpoints (event loop unblocked) · ✅ question-text
> log redaction (hash-only) · ✅ crisis-resource signposting · ✅ Redis-optional
> rate-limit/budget state (src/state.py) · ✅ semantic response cache
> (src/cache.py) · ✅ LLM concurrency semaphore + friendly 503 · ✅ security
> headers · ✅ pip-audit + Dependabot · ✅ backup + k6 load-test scripts ·
> ✅ privacy/ToS/runbook drafts (docs/).
> Licensing note: we hold direct written BaniDB permission (see NOTICE), which
> supersedes this plan's NPOSL-3.0 assumptions; the non-commercial posture is
> unchanged. Remaining items need maintainer accounts: Hetzner/Coolify,
> Cloudflare, Supabase auth, Sentry/Langfuse, Open Collective, paid Gemini
> tier, real load-test run, nonprofit registration.

# Gurbani Guidance: Production-Launch & Infrastructure Plan

## TL;DR

- **The current build is a strong prototype but not yet production-grade**; the core RAG logic, quote-verification, Docker, and CI are solid, but it is missing observability, real user auth, distributed rate-limiting, backups, a privacy/crisis-safety layer, and a horizontally scalable architecture — roughly a dozen gaps to close before public launch.
- **Launch as a PWA-first website on a cheap self-managed stack (Hetzner + Coolify + Cloudflare free tier), use Google Gemini Flash-Lite (paid tier) as the primary LLM with Claude Haiku 4.5 fallback, add Supabase Auth with anonymous-access-plus-optional-accounts, and fund via an Open Collective fiscal host** — this keeps 100K registered users affordable (~$1,050–1,350/month all-in) while preserving the non-commercial NPOSL-3.0 constraint.
- **The single-worker FastAPI process will NOT survive 5,000–10,000 concurrent users as-is**; the fixes are: offload BGE-M3 embedding off the event loop (or move query-embedding to a hosted API), make LLM calls fully async/streaming, move rate-limit and budget state to Redis, add a shared semantic response cache, run multiple stateless replicas behind a load balancer, and put Chroma/Qdrant into client-server mode.

-----

## Key Findings

1. **The dominant cost at scale is the LLM API, not infrastructure.** Cloudflare’s free tier absorbs DDoS and bandwidth; a €25–30/month Hetzner box runs the whole backend at launch. But 500,000 questions/month (100K users × 5) costs ~$3,875 on Claude Haiku 4.5 vs ~$1,075 on Gemini 3.1 Flash-Lite — an ~3.6× spread that decides whether donations can keep the service free.
1. **Semantic caching is the highest-leverage cost lever for this specific product.** Spiritual questions repeat heavily (“how do I deal with grief”, “what does Gurbani say about ego”). The peer-reviewed GPT Semantic Cache paper (Regmi & Pun, arXiv:2411.05276) reports it “reduces API calls by up to 68.8% across various query categories, with cache hit rates ranging from 61.6% to 68.8%” and “positive hit rates exceeding 97%”; a VentureBeat case study (Jan 2026) documented a jump from 18% exact-match to a 67% semantic hit rate, cutting a firm’s monthly LLM bill from $47,000 to $12,700 (a 73% reduction). AWS research on 63,796 real chatbot queries found up to 86% LLM-inference cost reduction at optimal thresholds. This roughly halves the LLM bill and dramatically improves peak survivability.
1. **Requiring login conflicts with the seva mission.** The right pattern is anonymous access with strict per-IP daily quotas plus an optional free account (via Supabase) that raises the quota and adds history — protecting the paid API key without gatekeeping access to Gurbani.
1. **PWA beats native for this use case.** Reach, SEO discoverability, offline Bani reading via service worker, instant updates, and zero app-store review all favour web; native (via a Capacitor wrapper) is a phase-2 option if push notifications and store presence become priorities.
1. **Privacy and crisis-safety are launch blockers, not nice-to-haves.** Users disclose grief, depression, and family crises. The service must minimise/avoid logging question text, use only the *paid* Gemini tier (the free tier may use data to improve Google’s products), rely on Anthropic’s no-training commercial terms, and add crisis-resource signposting.

-----

## Details

### A. Production-Readiness Scorecard (current build)

|Capability                                 |Status           |Notes                                                                                                        |
|-------------------------------------------|-----------------|-------------------------------------------------------------------------------------------------------------|
|Containerized deploy (Docker)              |✅ Have           |Corpus baked into image (60K lines, 15MB gz) — fine for a static corpus                                      |
|CI/CD (GitHub Actions)                     |✅ Have           |Unit tests + corpus audit + mock pipeline benchmark — good                                                   |
|Hallucination control                      |✅ Have           |Mandatory quote-verification layer is a genuine strength for scripture fidelity                              |
|Basic abuse controls                       |🟡 Partial        |In-memory per-IP rate limit + daily budget cap work only on a single pinned worker; break under multi-replica|
|Auth                                       |🟡 Partial        |Bearer-token option only; no real user accounts/management                                                   |
|Licensing/attribution                      |✅ Have           |MIT code + documented BaniDB permission; NPOSL-3.0 non-commercial understood                                 |
|Observability/monitoring                   |❌ Missing        |No metrics, tracing, or LLM-eval telemetry                                                                   |
|Error tracking                             |❌ Missing        |No Sentry-equivalent                                                                                         |
|Structured logging + retention policy      |❌ Missing        |Critical given sensitive question content                                                                    |
|Health checks + external uptime monitor    |❌ Missing        |Need `/health` + `/ready` + third-party uptime ping                                                          |
|Autoscaling / replicas / load balancer     |❌ Missing        |Single uvicorn worker pinned — hard scaling ceiling                                                          |
|Backups                                    |❌ Missing        |Chroma persistent volume + (future) user DB need scheduled backups                                           |
|Secrets management                         |🟡 Partial        |Move API keys from env files to a secret store                                                               |
|Load testing (real)                        |❌ Missing        |Only a *mock* full-pipeline benchmark; no k6/Locust test against real embedding+LLM path                     |
|Incident-response runbook                  |❌ Missing        |No documented on-call / rollback procedure                                                                   |
|Security headers (CSP/HSTS)                |❌ Missing/unknown|Add via Cloudflare + framework                                                                               |
|Dependency scanning                        |❌ Missing        |Enable Dependabot/Snyk + `pip-audit` in CI                                                                   |
|Privacy policy + Terms of Service          |❌ Missing        |Legally required, especially GDPR for diaspora users                                                         |
|Distributed rate-limit/budget state (Redis)|❌ Missing        |Prerequisite for multiple replicas                                                                           |
|Semantic/response cache                    |❌ Missing        |Biggest cost + peak-survivability win                                                                        |
|Crisis-resource signposting                |❌ Missing        |Duty-of-care for distressed users                                                                            |

**Verdict:** ~7 of 20 items solid, ~4 partial, ~9 missing. The prototype is production-*shaped* but needs the operational and safety layer before public traffic.

### B. Concurrency reality check

A single uvicorn worker with an **in-process 2.3GB BGE-M3 model**, ChromaDB, and **synchronous** LLM calls has two distinct ceilings:

- **I/O-bound LLM streaming is cheap to hold concurrently.** FastAPI/async can keep hundreds-to-thousands of streaming connections open per worker while waiting on the LLM (one practitioner benchmarked ~10,000 requests/minute on a single worker for a pure streaming proxy on 1 CPU/2GB RAM).  The catch: *any* synchronous call inside an `async` route blocks the whole event loop, and *every* user on that worker stalls. 
- **CPU-bound embedding is the real bottleneck.** Each query must be embedded by BGE-M3 (BAAI/bge-m3, per IONOS model docs and the SentenceTransformer registry: 568M parameters, XLM-RoBERTa architecture, 1024-dimensional output vectors, up to 8,192-token inputs, 100+ languages; Chen et al., arXiv:2402.03216, released January 2024; MIT-licensed). If embedding runs synchronously in the event loop, throughput collapses to a handful of requests/sec. It must be offloaded to a thread pool (`run_in_executor`) so it scales with CPU cores, or moved out of process entirely.

**Realistic single-worker capacity today:** on the order of a few requests/second sustained once real embedding + verification are in the path — nowhere near 5,000–10,000 concurrent users.

**Interpreting “5,000–10,000 at the same time”:** concurrent *users* ≠ concurrent *in-flight LLM requests*. If each active user asks a question every 1–2 minutes and each answer streams for ~5–15s, true in-flight concurrency is more like 500–2,000 — and a large share can be served from cache. That distinction is what makes the target affordable.

**Required architecture changes to survive the peak:**

1. **Offload embedding.** Either wrap BGE-M3 in a thread pool, run it as a separate Hugging Face TEI service, or (simplest for stateless scaling) switch *query* embedding to a hosted embedding API and keep the baked corpus vectors — this removes the 2.3GB model from each replica and lets them scale horizontally.
1. **Fully async LLM + streaming** via `httpx`/SSE; throttle outbound calls with an `asyncio.Semaphore` to stay under provider rate limits. 
1. **Move rate-limit + daily-budget state to Redis** so replicas share one truth.
1. **Add a Redis semantic response cache** (see cost section) — returns milliseconds-latency answers for repeated/paraphrased questions (cache hits ~27ms vs ~7s for a fresh Gemini call, per Percona 2026) and sheds LLM load during spikes.
1. **Run N stateless FastAPI replicas behind a load balancer** with autoscaling on CPU/connection count.
1. **Put the vector store in client-server mode.** Chroma is fine at ~150K vectors, but a shared server (Chroma client-server or Qdrant) lets all replicas query one store. Qdrant self-hosted on a small VPS handles millions of vectors for ~$30–50/month;  Qdrant Cloud has a free tier for small workloads. For a 150K-vector, 1024-dim corpus, every managed option is cheap (pgvector on your existing Postgres is effectively $0 incremental). 
1. **Graceful degradation:** when the budget cap or provider rate limit is hit, queue or return a friendly “high demand — please try again shortly” and, where possible, serve the closest cached answer.

### C. Target architecture

**Launch architecture (up to ~100 concurrent / ~1K–10K MAU):**

- **Edge:** Cloudflare free plan — DNS, CDN, unmetered DDoS, SSL, 5 custom WAF rules, 1 rate-limiting rule (10-second window, IP-based).
- **Frontend:** Next.js as a PWA on Cloudflare Pages (free) with a service worker for offline Bani reading.
- **Backend:** one Hetzner dedicated-vCPU box (CCX23-class, 4 vCPU/16GB) managed by Coolify; FastAPI with embedding offloaded to a thread pool; ChromaDB persistent; BM25 pickle; Redis (rate limits + semantic cache) as a sidecar container.
- **LLM:** Gemini 3.1 Flash-Lite (paid tier) primary, Claude Haiku 4.5 fallback, async streaming.
- **Auth + user DB:** Supabase (free tier) — email/password + Google sign-in; anonymous access allowed.
- **Observability:** Sentry (free), self-hosted Langfuse (MIT) for LLM traces/evals, UptimeRobot/Better Stack uptime monitor, `/health` endpoint.
- **Donations:** Open Collective (fiscal-hosted).

**Peak architecture (5–10K concurrent / ~100K MAU):**

- Same Cloudflare edge.
- **4–10 stateless FastAPI replicas** behind a load balancer (Hetzner LB or GCP Cloud Run autoscaling). Note: Cloud Run’s scale-to-zero causes multi-second cold starts with a 2.3GB in-image model (Google’s own guidance flags model-loading/VRAM transfer as the bottleneck for large images) — mitigate with min-instances ≥1 and startup CPU boost, or (better) remove the in-process model per fix #1 so cold starts are cheap.
- **Embedding** as a dedicated TEI service (or hosted embedding API) shared by all replicas.
- **Qdrant** (self-hosted on a dedicated node, or Qdrant Cloud) in client-server mode.
- **Central managed Redis** for shared rate-limit/budget state and the semantic cache (the shock absorber for spikes).
- **Queue + degradation** path in front of the LLM.
- **Supabase Pro** for the user/quota store.

### D. Monthly cost model (infra + LLM)

**LLM assumptions:** 5 questions/user/month; ~3,000 input tokens (RAG context) + ~500 classifier input; ~800 output + ~50 classifier output → ≈3,500 input / 850 output tokens per question.

Per-question cost:

- **Gemini 2.5 Flash-Lite** ($0.10/$0.40 per MTok):  **~$0.0007** *(cheapest, but retiring Oct 16, 2026  — migrate off)*
- **Gemini 3.1 Flash-Lite** ($0.25/$1.50):  **~$0.00215** *(go-forward cheap model)*
- **Claude Haiku 4.5** ($1/$5):  **~$0.008** — Anthropic’s official pricing page (anthropic.com/claude/haiku) states Haiku 4.5 “starts at $1 per million input tokens and $5 per million output tokens, with up to 90% cost savings with prompt caching and 50% cost savings with batch processing” (released Oct 15, 2025)

Monthly LLM cost (before caching):

|MAU    |Questions/mo|Gemini 3.1 Flash-Lite|Claude Haiku 4.5|
|-------|------------|---------------------|----------------|
|1,000  |5,000       |~$11                 |~$39            |
|10,000 |50,000      |~$107                |~$388           |
|100,000|500,000     |~$1,075              |~$3,875         |

With a ~40% semantic-cache hit rate (conservative — the research above shows 60–86% is achievable for repetitive-query apps), the 100K-MAU LLM bill drops to roughly **~$645 (Gemini 3.1 Flash-Lite)** or **~$2,325 (Haiku)**.

**Prompt caching mechanics (from Anthropic’s docs):** cache reads cost 0.1× base input (90% off) and 5-minute cache writes cost 1.25× base input (1-hour writes are 2×); for Haiku 4.5 that is “$1.25 per million tokens for writes and $0.10 per million tokens for reads” (per the Anthropic/Caylent breakdown). **Critical caveat: Claude Haiku 4.5 requires a 4,096-token minimum to cache** — a static system prompt shorter than that will silently not cache on Haiku (no error is returned). This is a change from the older, widely-cited “1,024-token minimum” (which still applies to Sonnet 5/4.6/4.5). Gemini’s context caching offers similar ~90% savings on cached input.  For this product the bigger win is *semantic response* caching (whole-answer reuse), not prompt caching.

Infra cost by tier:

|Tier                      |Infra components                                                                     |Est. infra/mo|
|--------------------------|-------------------------------------------------------------------------------------|-------------|
|Launch (~100 concurrent)  |1 Hetzner CCX23 + Cloudflare free + Pages free + Supabase free + Sentry/Langfuse free|~$30         |
|Growth (~1,000 concurrent)|2 Hetzner nodes + Qdrant self-host + Supabase Pro ($25)                              |~$145        |
|Peak (5–10K concurrent)   |6–10-node autoscaling fleet + LB + managed Redis + Qdrant + Supabase Pro             |~$400–700    |

**Cheapest viable total (recommended config: Gemini 3.1 Flash-Lite + semantic caching + quotas):**

|Tier                |Infra    |LLM (cached)|Total/mo         |
|--------------------|---------|------------|-----------------|
|Launch (1K–10K MAU) |~$30     |~$7–65      |**~$40–100**     |
|Growth (10K–50K MAU)|~$145    |~$65–325    |**~$210–470**    |
|Peak (100K MAU)     |~$400–700|~$645       |**~$1,050–1,350**|

DDoS mitigation and CDN bandwidth are free on Cloudflare regardless of tier (this is a text/JSON workload, so Cloudflare’s non-HTML/video ToS caveat on the free CDN does not apply).

**Nonprofit/credit programs to pursue** (can materially cut or zero the infra line): Google for Nonprofits → Google Cloud credits (requires nonprofit validation; note Google Ad Grants are NOT cloud credits), the AWS IMAGINE Grant for nonprofits (+ AWS Activate for startups), Microsoft for Nonprofits Azure credits, Cloudflare Project Galileo (public-interest/at-risk orgs), and GitHub for Nonprofits. Most require formal nonprofit status, which an Open Collective fiscal host can supply.

### E. Auth recommendation

**Provider: Supabase Auth.** Free to 50K MAU; per Supabase’s official docs, paid plans “include 100,000 MAUs. After that, you are charged $0.00325 per MAU,” with the Pro plan at $25/mo (8 GB database, 250 GB bandwidth, daily backups with 7-day retention). It supports email/password, Google OAuth, and phone OTP, integrates cleanly with Next.js and with FastAPI (verify the Supabase JWT server-side), and bundles the Postgres user/quota store. At 100K MAU it costs ~$25/month (auth included in Pro) vs Clerk’s ~$1,800 ($0.02/MAU after 10K free) and Auth0’s ~$2,400 — decisive for a donation-funded project. Firebase Auth (50K free, ~$275 at 100K) is the main alternative and is strong for phone auth, but ties you to Google’s ecosystem and has weaker Next.js DX. (Note: at very high scale — e.g., 500K MAU — Supabase auth overage would reach ~$1,300/mo, so revisit if the user base explodes.)

**Login vs anonymous — the seva tradeoff:** requiring login on a service whose mission is *maximum access to Gurbani* imposes friction that works against the mission and deters exactly the vulnerable users who most need it. **Recommendation: hybrid.**

- **Anonymous by default**, gated by strict per-IP daily quotas (e.g., a handful of questions/day) enforced in Redis + Cloudflare rate-limiting.
- **Optional free account** that raises the quota (e.g., 20/day), enables saved history/bookmarks, and gives you a stable identity for abuse control.
- **Social/phone strategy for the Punjab/diaspora audience:** lead with **Google sign-in** (near-ubiquitous on Android in India) and email/password; **defer phone OTP** because SMS costs ~$0.01+/message and is a bot-abuse cost vector. Add phone OTP only if data shows a real need.
- **Abuse prevention:** disposable-email blocking, Cloudflare Turnstile on signup, per-account and per-IP quotas, and the daily global budget cap as a backstop for the paid key.

### F. Website vs app verdict

**Verdict: PWA-first web app; native later only if needed.** Precedent Sikh apps (SikhiToTheMax by Khalis Foundation/SHARE Charity UK; iGurbani by Akal Design) are native, but they are *search/display* tools; Gurbani Guidance is a *Q&A guidance* service where discoverability and low friction matter more. A PWA gives:

- **Reach + SEO:** Gurbani-guidance queries are discovered via search; web pages are indexable, native apps are not. Industry data puts PWAs at 30–40% faster to production and 3–5× cheaper than dual-native builds.
- **Offline Bani reading** via service worker (cache the corpus text).
- **Push notifications** for hukamnama-style daily content — full on Android, and on iOS for *installed* PWAs since iOS 16.4 (March 2023).
- **Instant updates**, one codebase, no app-store review cycle.

Religious/scripture apps are permitted on both stores (no policy barrier for Sikh scripture content — the restrictions target hate/violence, sexual content, and unauthorized data collection), but native adds review latency and platform maintenance for little benefit at launch. **Phased roadmap:** (1) PWA at launch; (2) add installability + push + offline; (3) if store presence is desired, wrap the same Next.js app with Capacitor for iOS/Android rather than rebuilding — noting Apple’s rules on external donation/payment links and the 2025 requirement to clearly disclose when data is shared with third parties (including AI services).

### G. Launch-readiness specifics

- **Privacy / sensitive data:** Treat question text as highly sensitive (grief, depression, family). Default to **not logging question bodies**, or log only anonymized/aggregated metadata with short retention. Publish a plain-language privacy policy + ToS. **Use only the *paid* Gemini tier** — Google’s free tier may use data to improve its products, whereas the paid tier does not.  Anthropic’s commercial API terms do not train on your data; Anthropic’s official docs state conversation content “is not retained by default” for standard models and that retained data “is never used for model training without your express permission,” with a default retention window (docs cite 30 days; some 2026 reporting cites a reduction toward 7 days for standard API traffic), and a Zero Data Retention (ZDR) arrangement is available for eligible customers via a commercial agreement. Consider a ZDR/DPA for the strongest posture.
- **Crisis handling:** Add a classifier-triggered **crisis-resource signpost** (regional helplines) when distress is detected, plus a standing disclaimer that the service offers spiritual reflection, not medical/mental-health treatment.
- **Gurmukhi + accessibility:** Bundle a proper Gurmukhi Unicode font (e.g., Noto Sans Gurmukhi), offer font-size/line-height controls, ensure high contrast and screen-reader labels, and test rendering of lagamatras/padh chhedh (BaniDB is standardized for both, so preserve that fidelity).
- **SEO:** Server-render shabad/ang pages with structured data and canonical URLs so Gurbani content is discoverable — a web/PWA advantage over native.
- **Community trust:** A transparency page modeled on peers (e.g., readsggs.com, which states it “charges nothing, displays no advertising, collects no payments, and earns no revenue of any kind”) stating “free, ad-free, no revenue,” explicit **BaniDB attribution under NPOSL-3.0**, a link to the open-source repo, and sevadar credits. This is essential for legitimacy in the Sikh community.
- **Licensing constraint:** BaniDB data is under **NPOSL-3.0**, which (per its own terms and the readsggs.com attribution precedent) “requires that anyone distributing this work represents they are a not-for-profit organisation that derives no revenue whatsoever from it.” Donations that only cover infrastructure are compatible, but must be structured as nonprofit funding (fiscal host), never as revenue from the data or service.

-----

## Recommendations

**Stage 0 — Close the launch-blocking gaps (pre-launch, 2–4 weeks):**

1. Offload BGE-M3 embedding off the event loop; make all LLM calls async/streaming with a concurrency semaphore.
1. Move rate-limit + budget state to Redis; add the Redis semantic response cache.
1. Add Sentry, structured logging (with question-text redaction), Langfuse traces, `/health` + external uptime monitor.
1. Write privacy policy + ToS; switch Gemini to the paid tier; enable Anthropic no-train commercial terms (consider ZDR).
1. Add crisis-resource signposting + disclaimer.
1. Add Dependabot/`pip-audit`, security headers, scheduled Chroma/DB backups, and a one-page incident runbook.
1. Run a real load test (k6/Locust) against the full embedding+LLM path — this, not the estimates here, should size the fleet.

**Stage 1 — Launch config:** Hetzner + Coolify backend, Cloudflare free edge, Next.js PWA on Cloudflare Pages, Supabase Auth (anonymous + optional Google/email accounts), Gemini 3.1 Flash-Lite primary + Haiku fallback, Open Collective donations. Budget ~$40–100/month.

**Stage 2 — Growth (trigger at sustained >100 concurrent or LLM bill >$300/mo):** add a second replica + load balancer, move vectors to Qdrant client-server, upgrade to Supabase Pro, tune the semantic-cache hit rate. Budget ~$210–470/month.

**Stage 3 — Peak readiness (trigger at sustained >1,000 concurrent):** autoscaling replica fleet, dedicated embedding service (or hosted embedding API), central managed Redis, queue + graceful degradation, min-instances to avoid cold starts. Budget ~$1,050–1,350/month at 100K MAU.

**Launch week checklist:** freeze the corpus + prompt version tags; verify semantic-cache warm-up on the top ~200 expected questions; confirm quota + budget-cap kill-switch works across replicas; publish transparency/privacy/ToS pages; set uptime + budget alerts; soft-launch to a gurdwara/community beta before broad promotion.

**Post-launch:** monitor LLM cost vs donations weekly; watch cache hit-rate and retrieval-quality (Precision@k / groundedness) for drift after any corpus refresh; collect sevadar and community feedback on answer tone and scripture fidelity.

**Thresholds that change the plan:** if the LLM bill exceeds donations for two consecutive months, tighten quotas and raise semantic-cache aggressiveness before adding capacity. If Gemini quality proves insufficient for pastoral tone, route only sensitive/complex questions to Claude Sonnet and keep Flash-Lite for the rest (model routing). If iOS push/store presence becomes a top user request, trigger the Capacitor wrapper.

-----

## Caveats

- **LLM pricing and model names move fast.** Gemini 2.5 Flash-Lite (the current price floor) retires Oct 16, 2026;  plan on Gemini 3.1 Flash-Lite. Claude Sonnet 5 carries introductory pricing ($2/$10) only through Aug 31, 2026, reverting to $3/$15 on Sept 1. Re-verify all rates at build time.
- **Anthropic retention terms are reported inconsistently** across sources (30-day default in official docs vs 7-day claims elsewhere, plus special 30-day rules for “covered” Mythos/Fable-class models effective June 9, 2026). Confirm the exact DPA/ZDR terms directly with Anthropic before handling sensitive user text.
- **Concurrency estimates depend on request mix.** Actual capacity hinges on cache hit rate, embedding latency, and streaming duration; the real load test in Stage 0 is what should size the fleet, not these estimates.
- **Several infra/auth figures came from third-party pricing analyses** rather than the vendors’ own dashboards; treat them as directional and confirm on official pricing pages (Supabase auth pricing and Anthropic caching mechanics here are from primary docs).
- **Nonprofit credit eligibility varies** and generally requires formal nonprofit status; timelines and approval are not guaranteed.