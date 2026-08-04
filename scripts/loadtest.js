// Real-path load test (k6) — sizes the fleet; run against a staging deploy.
//   k6 run -e BASE=https://staging.example.com -e VUS=50 -e DUR=2m scripts/loadtest.js
// NOTE: every /ask hits the real embedding+retrieval path and (unless cached)
// the LLM — run against staging with a test API key and a low DAILY_REQUEST_CAP.
import http from "k6/http";
import { check, sleep } from "k6";

const BASE = __ENV.BASE || "http://localhost:8000";
const QUESTIONS = [
  "What does Gurbani say about haumai?",
  "What is Naam Simran?",
  "I feel lost and alone, what does Gurbani teach?",
  "What does Sukhmani Sahib say about peace?",
  "ਹਉਮੈ ਬਾਰੇ ਗੁਰਬਾਣੀ ਕੀ ਸਿਖਾਉਂਦੀ ਹੈ?",
];

export const options = {
  vus: Number(__ENV.VUS || 20),
  duration: __ENV.DUR || "1m",
  thresholds: {
    http_req_failed: ["rate<0.05"],
    http_req_duration: ["p(95)<30000"],
  },
};

export default function () {
  // health/ready are cheap liveness signals under load
  check(http.get(`${BASE}/health`), { "health 200": (r) => r.status === 200 });
  const q = QUESTIONS[Math.floor(Math.random() * QUESTIONS.length)];
  const res = http.post(`${BASE}/ask`, JSON.stringify({ question: q }), {
    headers: { "Content-Type": "application/json" },
    timeout: "60s",
  });
  check(res, {
    "ask ok or throttled": (r) => r.status === 200 || r.status === 429 || r.status === 503,
  });
  sleep(Math.random() * 3 + 1);
}
