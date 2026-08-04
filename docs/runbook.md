# Incident Runbook (one page)

## Contacts & dashboards
- Maintainer: akashdatageek · Uptime: (external monitor URL) · Errors: (Sentry URL)

## Golden signals
- `GET /health` — liveness (+ index passage counts)
- `GET /ready`  — readiness (503 until index is loaded — LB gate)
- Logs: `Response-cache hit`, `Daily request budget reached`, `LLM concurrency limit saturated`

## Common incidents
| Symptom | Likely cause | Action |
|---|---|---|
| 503 on /ready at boot | Chroma index missing on volume | `python -m src.embed` on the volume; boot fails LOUD by design |
| Every /ask 429 | Daily budget cap hit | Verify spend; raise `DAILY_REQUEST_CAP` or wait for UTC midnight |
| 503 "high demand" | LLM semaphore saturated | Check provider status/limits; raise `LLM_MAX_CONCURRENCY` only if provider quota allows |
| Answers slow, no cache hits | Redis down (falls back to per-process) | Restore Redis; check `REDIS_URL`; restart replicas |
| Quotes being stripped for real Gurbani | Corpus/prompt drift | `python -m src.audit`; check `PROMPT_VERSION` bump; roll back last deploy |

## Rollback
Deploys are stateless images: redeploy the previous tag. Data volume is
independent; restore from `scripts/backup_data.sh` archives if corrupted.

## After any incident
Write 3 lines in the log: what broke, why, what prevents recurrence.
