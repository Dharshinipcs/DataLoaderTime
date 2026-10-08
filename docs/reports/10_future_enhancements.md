# Report 10 — Future Enhancements

> ⚠️ **Status: ideas backlog — none of these are implemented.** Each entry notes where it would attach.

## Reliability & data

1. **Database-backed store** — swap JSONL stores (Report 6) for SQLite/Postgres behind the same repository functions; JSONL stays as export format. Attach in `store/` accessors only.
2. **Automated delivery pipeline** — implement the DataZap design (Report 8): atomic workbook delivery + post-delivery validation probe + optional candidate trigger. No lifecycle changes needed.
3. **Wider conformal calibration** — rolling or per-slice residual windows to replace the single 85-residual window (attack the ≥12× width-factor LOW-reliability false alarms).

## ML

4. **Per-stage drift alerts** — per-feature PSI/KS on the drift endpoints surfaced as per-stage badges in the Stage Model Error Comparison table.
5. **Champion/challenger auto-evaluation** — nightly candidate training on new outcomes with the candidate parked in `REGISTERED` (still human-promoted); recommendation card links to it.
6. **Approvals ingestion** — real approval timestamps feed the two contextual estimates (today they are user-supplied estimates) and could become ML features behind the locked leakage rules.

## Product / UI

7. **Export & audit pack** — one-click download of registry + lifecycle + evaluations for compliance review.
8. **Multi-user roles** — promote/rollback restricted to an approver role; decision log already records reasons.
9. **Loader onboarding wizard** — guided first-prediction flow for new loaders (INSUFFICIENT_HISTORY state made actionable).

## Ops / delivery

10. **Deployment packaging** — Dockerfile(s) + compose for backend/frontend, or a single FastAPI serving the built `frontend/dist` for one-process hosting; enables the "shareable link" path (hosting + tunnel or a PaaS deploy).
11. **CI** — GitHub Actions running backend pytest + frontend vitest/tsc/build on PR; the targeted suites already run in minutes.
12. **Config hardening** — `DLTIME_DATA_DIR` + ports + thresholds via a single documented `.env` (currently env var + CLI flags).

## Deliberately NOT on the roadmap

- Automatic promotion of candidates (human gate is a product invariant).
- Summation "corrections" that would make the served total non-transparent (the EXP-037/038 decision is recorded and reversible, not forgotten).
