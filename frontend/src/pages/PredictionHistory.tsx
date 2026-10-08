import { useEffect, useMemo, useState } from 'react'
import {
  ApiError,
  type FeedbackRecord,
  type PredictionEventRecord,
  fetchMonitoringEvents,
  fetchMonitoringFeedback,
} from '../api'
import { formatSeconds } from '../lib/format'
import { Badge, Card, EmptyState, PageHeader } from '../components/ui'

type Phase =
  | { kind: 'loading' }
  | { kind: 'done'; events: PredictionEventRecord[]; feedback: FeedbackRecord[] }
  | { kind: 'error'; message: string }

interface Filters {
  search: string
  loader: string
  dataset: string
  connection: string
  sprint: string
  status: string
  outcome: string
  risk: string
  from: string
  to: string
}

const EMPTY_FILTERS: Filters = {
  search: '',
  loader: '',
  dataset: '',
  connection: '',
  sprint: '',
  status: '',
  outcome: '',
  risk: '',
  from: '',
  to: '',
}

/** Table status badges: the Generated/Unavailable distinction first, then the
 * honest state label. No internal jargon anywhere. */
const STATE_LABELS: Record<string, string> = {
  PREDICTED: 'Prediction generated',
  INSUFFICIENT_HISTORY: 'Insufficient historical evidence',
  UNRESOLVED_IDENTITY: 'Required identity could not be resolved',
  AMBIGUOUS_IDENTITY: 'Identity matched multiple configurations — refused to guess',
  INVALID_INPUT: 'Request was rejected (invalid input)',
}

const STATE_SHORT: Record<string, string> = {
  PREDICTED: 'Predicted',
  INSUFFICIENT_HISTORY: 'Insufficient history',
  UNRESOLVED_IDENTITY: 'Could not resolve',
  AMBIGUOUS_IDENTITY: 'Ambiguous identity',
  INVALID_INPUT: 'Rejected',
}

function isGenerated(e: PredictionEventRecord): boolean {
  return e.event_state === 'PREDICTED' && !!e.prediction
}

// 2026-10-03: Status/Risk table columns (and their tone helpers) were
// removed — those dimensions live in the filters and the detail drawer.

export default function PredictionHistory() {
  const [phase, setPhase] = useState<Phase>({ kind: 'loading' })
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS)
  const [applied, setApplied] = useState<Filters>(EMPTY_FILTERS)

  useEffect(() => {
    // 2026-10-05 (user spec): the list must stay live — silently refetch when
    // the user returns to the tab so new predictions and saved outcomes appear
    // without a manual reload.
    const load = () =>
      Promise.all([fetchMonitoringEvents(300), fetchMonitoringFeedback(300)])
        .then(([events, feedback]) =>
          setPhase({ kind: 'done', events: events.events ?? [], feedback: feedback.feedback ?? [] }),
        )
        .catch((err: unknown) =>
          setPhase({
            kind: 'error',
            message:
              err instanceof ApiError
                ? err.message
                : 'Unexpected error while contacting the backend.',
          }),
        )
    load()
    window.addEventListener('focus', load)
    document.addEventListener('visibilitychange', load)
    return () => {
      window.removeEventListener('focus', load)
      document.removeEventListener('visibilitychange', load)
    }
  }, [])

  function apply(e: React.FormEvent) {
    e.preventDefault()
    setApplied({ ...filters })
  }

  return (
    <div>
      <PageHeader
        eyebrow=""
        title="Prediction History"
      />

      {phase.kind === 'loading' && (
        <div
          role="status"
          aria-live="polite"
          className="flex min-h-40 items-center justify-center rounded-md border border-linen bg-parchment p-8"
        >
          <div className="text-center">
            <div className="mx-auto h-7 w-7 animate-spin rounded-full border-2 border-bronze border-t-transparent" />
            <p className="mt-3 text-sm text-warmgray">Loading prediction history…</p>
          </div>
        </div>
      )}
      {phase.kind === 'error' && (
        <div role="alert" className="rounded-md border border-[#e3c6c1] bg-[#f6e7e4] p-4 text-sm text-ember">
          {phase.message}
        </div>
      )}
      {phase.kind === 'done' && (
        <HistoryBody
          events={phase.events}
          feedback={phase.feedback}
          filters={applied}
          apply={apply}
          filtersValue={filters}
          setFilters={setFilters}
          setApplied={setApplied}
        />
      )}
    </div>
  )
}

function HistoryBody({
  events,
  feedback,
  filters,
  apply,
  filtersValue,
  setFilters,
  setApplied,
}: {
  events: PredictionEventRecord[]
  feedback: FeedbackRecord[]
  filters: Filters
  apply: (e: React.FormEvent) => void
  filtersValue: Filters
  setFilters: (f: Filters) => void
  setApplied: (f: Filters) => void
}) {
  const [moreOpen, setMoreOpen] = useState(false)
  const [detail, setDetail] = useState<PredictionEventRecord | null>(null)
  // 2026-10-04 (user spec): 10 predictions per view with prev/next paging.
  const [page, setPage] = useState(0)
  const PER_VIEW = 10

  // event_id -> matched feedback (actual/error source of truth)
  const byEvent = useMemo(() => {
    const m = new Map<string, FeedbackRecord>()
    for (const f of feedback) {
      if (f.prediction_event_id && f.linkage.status === 'MATCHED') {
        m.set(f.prediction_event_id, f)
      }
    }
    return m
  }, [feedback])

  const filtered = useMemo(() => {
    const q = (s: string | null | undefined) => (s ?? '').toLowerCase()
    return events.filter((e) => {
      const req = e.request ?? {}
      if (filters.search) {
        const hay = q(
          [req.loader_name, req.dataset, req.connection, req.sprint].join(' '),
        )
        if (!hay.includes(filters.search.toLowerCase())) return false
      }
      if (filters.loader && !q(req.loader_name).includes(filters.loader.toLowerCase())) return false
      if (filters.dataset && !q(req.dataset).includes(filters.dataset.toLowerCase())) return false
      if (filters.sprint && !q(req.sprint).includes(filters.sprint.toLowerCase())) return false
      if (filters.connection && !q(req.connection).includes(filters.connection.toLowerCase())) return false
      if (filters.status === 'generated' && !isGenerated(e)) return false
      if (filters.status === 'unavailable' && isGenerated(e)) return false
      if (filters.status && filters.status !== 'generated' && filters.status !== 'unavailable' && e.event_state !== filters.status) return false
      const fb = e.event_id ? byEvent.get(e.event_id) : undefined
      const outcome: 'available' | 'awaiting' | 'na' = isGenerated(e) ? (fb ? 'available' : 'awaiting') : 'na'
      if (filters.outcome && outcome !== filters.outcome) return false
      if (filters.risk === 'raised' && !(e.prediction?.risk_label && e.prediction.risk_label !== 'NORMAL')) return false
      if (filters.risk === 'normal' && e.prediction?.risk_label !== 'NORMAL') return false
      if (filters.from) {
        const d = new Date(e.created_at)
        if (!Number.isNaN(d.getTime()) && d < new Date(filters.from)) return false
      }
      if (filters.to) {
        const d = new Date(e.created_at)
        if (!Number.isNaN(d.getTime()) && d > new Date(filters.to + 'T23:59:59')) return false
      }
      return true
    })
    // 2026-10-05 (user spec): newest first — the most recent
    // prediction is at the top, no searching through pages.
    .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
  }, [events, filters, byEvent])

  // Matched chart rows from the FILTERED population only.
  const chartRows = useMemo(
    () =>
      filtered
        .filter((e) => e.prediction && e.event_id && byEvent.has(e.event_id))
        .map((e) => {
          const fb = byEvent.get(e.event_id)!
          const predicted = e.prediction!.predicted_total_seconds
          const signed = fb.total_error?.signed_error_seconds ?? null
          const actual = signed !== null ? predicted - signed : null
          const absError = fb.total_error?.absolute_error_seconds ?? null
          const relError = actual !== null && actual !== 0 && signed !== null ? (-signed / actual) * 100 : null
          return {
            event: e,
            predicted,
            actual: actual!,
            signedError: signed,
            absError,
            relError,
            status: fb.identity?.loader_name !== undefined ? (fb.total_error ? 'recorded' : null) : null,
          }
        })
        .filter((r) => Number.isFinite(r.actual))
        // 2026-10-05 (user spec): newest first — the most recent
        // run is at the top, no searching.
        .sort((a, b) => new Date(b.event.created_at).getTime() - new Date(a.event.created_at).getTime()),
    [filtered, byEvent],
  )

  // Compact summary + accuracy — computed from the filtered population.
  const summary = useMemo(() => {
    const attempts = filtered.length
    const generated = filtered.filter(isGenerated)
    const withActual = chartRows
    const absErrors = withActual.map((r) => r.absError ?? 0).filter((v) => v > 0 || v === 0)
    const meanAbs = absErrors.length ? absErrors.reduce((a, b) => a + b, 0) / absErrors.length : null
    const medAbs = absErrors.length ? median(absErrors) : null
    const within10 = withActual.filter((r) => r.relError !== null && Math.abs(r.relError) <= 10).length
    const within20 = withActual.filter((r) => r.relError !== null && Math.abs(r.relError) <= 20).length
    return {
      attempts,
      generated: generated.length,
      actuals: withActual.length,
      meanAbs,
      medAbs,
      within10Pct: withActual.length ? (100 * within10) / withActual.length : null,
      within20Pct: withActual.length ? (100 * within20) / withActual.length : null,
    }
  }, [filtered, chartRows])

  const activeChips = useMemo(
    () =>
      Object.entries(filters)
        .filter(([, v]) => v)
        .map(([k, v]) => ({ key: k, value: v })),
    [filters],
  )

  return (
    <div className="space-y-5">
      {/* Compact filter bar + More Filters + active chips. */}
      <Card testId="history-filters">
        <form onSubmit={apply} className="flex flex-wrap items-center gap-2">
          <input
            className="field-input min-w-56 !w-auto flex-1 basis-56"
            placeholder="Search loader, dataset, connection…"
            aria-label="Search filter"
            value={filtersValue.search}
            onChange={(e) => setFilters({ ...filtersValue, search: e.target.value })}
          />
          <input className="field-input !w-40" placeholder="Loader" aria-label="Loader filter" value={filtersValue.loader} onChange={(e) => setFilters({ ...filtersValue, loader: e.target.value })} />
          <select className="field-input !w-auto" aria-label="Sprint filter" value={filtersValue.sprint} onChange={(e) => setFilters({ ...filtersValue, sprint: e.target.value })}>
            <option value="">Sprint: any</option>
            {distinct(feedback.length ? [] : [], filtersValue.sprint, filtered, (e) => e.request?.sprint).map((s) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <input type="date" className="field-input !w-36" aria-label="From date" value={filtersValue.from} onChange={(e) => setFilters({ ...filtersValue, from: e.target.value })} />
          <button type="button" onClick={() => setMoreOpen((o) => !o)} aria-expanded={moreOpen} className="btn-secondary whitespace-nowrap" data-testid="more-filters-toggle">
            More Filters
          </button>
          <button type="submit" className="btn-primary whitespace-nowrap">Apply filters</button>
        </form>
        {moreOpen && (
          <form onSubmit={apply} className="mt-2 flex flex-wrap items-center gap-2 border-t border-linen pt-2" data-testid="more-filters">
            <input className="field-input !w-40" placeholder="Dataset" aria-label="Dataset filter" value={filtersValue.dataset} onChange={(e) => setFilters({ ...filtersValue, dataset: e.target.value })} />
            <input className="field-input !w-40" placeholder="Connection" aria-label="Connection filter" value={filtersValue.connection} onChange={(e) => setFilters({ ...filtersValue, connection: e.target.value })} />
            <select className="field-input !w-auto" aria-label="Prediction status filter" value={filtersValue.status} onChange={(e) => setFilters({ ...filtersValue, status: e.target.value })}>
              <option value="">Prediction: any</option>
              <option value="generated">Prediction generated</option>
              <option value="unavailable">Prediction unavailable</option>
              {Object.entries(STATE_SHORT).map(([k, v]) => (
                <option key={k} value={k}>{v}</option>
              ))}
            </select>
            <select className="field-input !w-auto" aria-label="Outcome status filter" value={filtersValue.outcome} onChange={(e) => setFilters({ ...filtersValue, outcome: e.target.value })}>
              <option value="">Outcome: any</option>
              <option value="available">Actual available</option>
              <option value="awaiting">Awaiting actual</option>
              <option value="na">Not applicable</option>
            </select>
            <select className="field-input !w-auto" aria-label="Risk filter" value={filtersValue.risk} onChange={(e) => setFilters({ ...filtersValue, risk: e.target.value })}>
              <option value="">Risk: all</option>
              <option value="normal">Normal only</option>
              <option value="raised">Raised a risk signal</option>
            </select>
            <input type="date" className="field-input !w-36" aria-label="To date" value={filtersValue.to} onChange={(e) => setFilters({ ...filtersValue, to: e.target.value })} />
          </form>
        )}
        {activeChips.length > 0 && (
          <div className="mt-2 flex flex-wrap items-center gap-2 text-xs" data-testid="active-filters">
            <span className="text-warmgray">Filters:</span>
            {activeChips.map((c) => (
              <button
                key={c.key}
                type="button"
                className="rounded-full bg-[#f3ead9] px-2 py-0.5 font-medium text-ochre hover:bg-[#eadfc8]"
                onClick={() => {
                  const next = { ...filtersValue, [c.key]: '' }
                  setFilters(next)
                  setApplied(next)
                }}
                title="Click to clear this filter"
              >
                {c.key}: {c.value} ✕
              </button>
            ))}
            <button
              type="button"
              className="text-warmgray underline hover:text-ink"
              onClick={() => {
                setFilters(EMPTY_FILTERS)
                setApplied(EMPTY_FILTERS)
              }}
            >
              clear all
            </button>
          </div>
        )}
      </Card>

      {/* 2026-10-03 compact single-page layout: filters on top, then a
          two-column workspace — LEFT: the prediction-vs-actual graph plus the
          live accuracy cards (recomputed from the current system state on
          every load, so the error shrinks as the models improve); RIGHT: the
          predictions list. Status and Risk stay in the filters — they are not
          table columns. */}
      <div className="grid gap-4 lg:grid-cols-2" data-testid="history-workspace">
        <div className="space-y-4">
          {/* Live accuracy counts — filtered population, no model information. */}
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4" data-testid="history-summary">
            <SummaryFact label="Attempts" value={String(summary.attempts)} />
            <SummaryFact label="Generated" value={String(summary.generated)} />
            <SummaryFact label="With actual" value={String(summary.actuals)} />
            <SummaryFact label="Mean abs error" value={summary.meanAbs !== null ? formatSeconds(summary.meanAbs) : '—'} />
          </div>

          {/* Hero: prediction vs actual over time. */}
          <Card title="Prediction vs actual" testId="history-chart">
        {chartRows.length >= 2 ? (
          <HistoryChart rows={chartRows} />
        ) : (
          <EmptyState title="Not enough matched predictions yet">
            The timeline appears once at least two predictions have matched actual outcomes.{' '}
            {chartRows.length === 1 ? 'One matched prediction so far.' : 'No matched predictions in this view.'}
          </EmptyState>
        )}
          </Card>

          {/* Accuracy — only over predictions with available actual outcomes. */}
          <Card title="Prediction accuracy" testId="history-accuracy">
        {summary.actuals > 0 ? (
          <div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
              <SummaryFact label="Median absolute error" value={summary.medAbs !== null ? formatSeconds(summary.medAbs) : '—'} />
              <SummaryFact label="Mean absolute error" value={summary.meanAbs !== null ? formatSeconds(summary.meanAbs) : '—'} />
              <SummaryFact label="Within ±10%" value={summary.within10Pct !== null ? `${summary.within10Pct.toFixed(1)}%` : '—'} />
              <SummaryFact label="Within ±20%" value={summary.within20Pct !== null ? `${summary.within20Pct.toFixed(1)}%` : '—'} />
            </div>
          </div>
        ) : (
          <p className="text-sm text-warmgray">
            No matched actual outcomes in this view yet — accuracy is never shown without real
            outcomes to compare against.
          </p>
        )}
          </Card>
        </div>

        {/* Individual prediction events (right column). */}
        <Card title="Predictions" testId="history-table">
        {filtered.length === 0 ? (
          <EmptyState title="No predictions match these filters" />
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm" data-testid="history-table-body">
              <thead>
                <tr>
                  {/* 2026-10-03: Status and Risk are filter controls, not
                      columns — removed from the table per the user spec. */}
                  {['When', 'Loader', 'Dataset', 'Predicted', 'Actual', 'Error'].map((h) => (
                    <th key={h} className="border-b border-linen py-2 pr-3 text-[11px] font-semibold uppercase tracking-wide text-warmgray">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.slice(page * PER_VIEW, (page + 1) * PER_VIEW).map((e) => {
                  const fb = e.event_id ? byEvent.get(e.event_id) : undefined
                  const predicted = e.prediction?.predicted_total_seconds ?? null
                  const signed = fb?.total_error?.signed_error_seconds ?? null
                  const actual = predicted !== null && signed !== null ? predicted - signed : null
                  const rel = actual !== null && actual !== 0 && signed !== null ? (-signed / actual) * 100 : null
                  return (
                    <tr
                      key={e.event_id}
                      data-testid="history-row"
                      className="cursor-pointer hover:bg-[#fbf8f1]"
                      onClick={() => setDetail(e)}
                    >
                      <td className="border-b border-[#efece4] py-2 pr-3 text-warmgray">{fmtWhen(e.created_at)}</td>
                      <td className="border-b border-[#efece4] py-2 pr-3 font-medium text-ink">{e.request?.loader_name ?? '—'}</td>
                      <td className="max-w-40 truncate border-b border-[#efece4] py-2 pr-3 text-warmgray" title={e.request?.dataset ?? ''}>{e.request?.dataset ?? '—'}</td>
                      <td className="border-b border-[#efece4] py-2 pr-3">{predicted !== null ? formatSeconds(predicted) : '—'}</td>
                      <td className="border-b border-[#efece4] py-2 pr-3">{actual !== null ? formatSeconds(actual) : fb ? '—' : '—'}</td>
                      <td className="border-b border-[#efece4] py-2 pr-3">{rel !== null ? `${rel.toFixed(1)}%` : '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
            {(() => {
              const pageCount = Math.max(1, Math.ceil(filtered.length / PER_VIEW))
              const safePage = Math.min(page, pageCount - 1)
              return pageCount > 1 ? (
                <div className="mt-2 flex items-center gap-3 text-xs" data-testid="history-pager">
                  <button
                    type="button"
                    className="btn-secondary !py-1 disabled:opacity-40"
                    disabled={safePage === 0}
                    onClick={() => setPage(Math.max(0, safePage - 1))}
                  >
                    ← Prev 10
                  </button>
                  <span className="text-warmgray">
                    Page {safePage + 1} of {pageCount} · {filtered.length} predictions
                  </span>
                  <button
                    type="button"
                    className="btn-secondary !py-1 disabled:opacity-40"
                    disabled={safePage >= pageCount - 1}
                    onClick={() => setPage(Math.min(pageCount - 1, safePage + 1))}
                  >
                    Next 10 →
                  </button>
                </div>
              ) : null
            })()}
          </div>
        )}
        </Card>
      </div>

      {detail && <EventDetail event={detail} feedback={detail.event_id ? byEvent.get(detail.event_id) : undefined} onClose={() => setDetail(null)} />}
    </div>
  )
}

function distinct(
  _seed: string[],
  _current: string,
  events: PredictionEventRecord[],
  pick: (e: PredictionEventRecord) => string | null | undefined,
): string[] {
  void _seed
  void _current
  return [...new Set(events.map(pick).filter((s): s is string => !!s))].sort().slice(0, 20)
}

function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b)
  const mid = Math.floor(s.length / 2)
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2
}

function SummaryFact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md border border-linen bg-[#fbf8f1] px-3 py-2">
      <dt className="text-[11px] uppercase tracking-wide text-warmgray">{label}</dt>
      <dd className="mt-0.5 font-serif text-lg font-bold text-ink">{value}</dd>
    </div>
  )
}

/* ------------------------------ detail drawer ------------------------------ */

function EventDetail({
  event,
  feedback,
  onClose,
}: {
  event: PredictionEventRecord
  feedback?: FeedbackRecord
  onClose: () => void
}) {
  const p = event.prediction
  const generated = isGenerated(event)
  const predicted = p?.predicted_total_seconds ?? null
  const signed = feedback?.total_error?.signed_error_seconds ?? null
  const actual = predicted !== null && signed !== null ? predicted - signed : null
  const absError = feedback?.total_error?.absolute_error_seconds ?? null
  const rel = actual !== null && actual !== 0 && signed !== null ? (-signed / actual) * 100 : null
  const stages = (event.stage_intelligence?.stages ?? []).filter(
    (s) => !s.stage.endsWith('approval'),
  )
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/25" onClick={onClose} data-testid="event-detail-overlay">
      <div
        className="h-full w-full max-w-md overflow-y-auto border-l border-linen bg-parchment p-5 shadow-xl"
        onClick={(e) => e.stopPropagation()}
        data-testid="event-detail"
        role="dialog"
        aria-label="Prediction details"
      >
        <div className="flex items-start justify-between gap-2">
          <div>
            <h3 className="font-serif text-lg font-bold text-ink">
              {event.request?.loader_name ?? 'Unknown loader'}
            </h3>
            <p className="text-xs text-warmgray">{fmtWhen(event.created_at)}</p>
          </div>
          <button type="button" onClick={onClose} className="btn-secondary !px-2 !py-1 text-xs" aria-label="Close details">✕</button>
        </div>

        <div className="mt-3 flex flex-wrap gap-2">
          <Badge tone={generated ? 'success' : 'warning'}>{generated ? 'Prediction generated' : 'Prediction unavailable'}</Badge>
          {feedback && <Badge tone="sage">Actual available</Badge>}
          {generated && !feedback && <Badge tone="bronze">Awaiting actual</Badge>}
          {!generated && <Badge tone="neutral">Outcome not applicable</Badge>}
        </div>
        {!generated && (
          <p className="mt-2 rounded-md border border-[#e0cda6] bg-[#f3ead9] p-3 text-sm text-ochre" data-testid="detail-unavailable-reason">
            {STATE_LABELS[event.event_state] ?? event.event_state}
          </p>
        )}

        {generated && p && (
          <>
            <Section title="Prediction">
              <p className="font-serif text-3xl font-bold text-ink">{formatSeconds(p.predicted_total_seconds)}</p>
              {p.model_version && (
                <p className="mt-1 text-xs text-warmgray">Model: {p.model_version}</p>
              )}
            </Section>

            <Section title="Stage breakdown">
              {stages.length > 0 ? (
                <ul className="space-y-1 text-sm">
                  {stages.map((s) => (
                    <li key={s.stage} className="flex justify-between gap-3">
                      <span className="capitalize text-warmgray">{s.stage.replace(/_/g, ' ')}</span>
                      <span className="font-medium text-ink">
                        {s.available && s.estimate_seconds != null ? formatSeconds(s.estimate_seconds) : 'not recorded'}
                      </span>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-warmgray">Stage predictions were not recorded for this event.</p>
              )}
            </Section>
          </>
        )}

        <Section title="Actual outcome">
          {feedback ? (
            actual !== null ? (
              <div className="space-y-1 text-sm" data-testid="detail-outcome">
                <p>
                  Actual total: <strong className="text-ink">{formatSeconds(actual)}</strong>
                </p>
                {absError !== null && <p>Absolute error: {formatSeconds(absError)}</p>}
                {rel !== null && (
                  <p>
                    Relative error: {rel.toFixed(1)}% —{' '}
                    {rel < 0 ? `prediction was ${Math.abs(rel).toFixed(1)}% higher than actual` : rel > 0 ? `prediction was ${rel.toFixed(1)}% lower than actual` : 'prediction matched actual'}
                  </p>
                )}
                <p className="text-xs text-warmgray">
                  Training eligible: {feedback.training_eligible ? 'Yes' : `No (${feedback.training_eligibility_reason})`}
                </p>
              </div>
            ) : (
              <p className="text-sm text-warmgray">Actual recorded without a comparable duration.</p>
            )
          ) : generated ? (
            <p className="text-sm text-warmgray">
              No actual outcome recorded yet — it appears here once the real execution finishes and
              is saved.
            </p>
          ) : (
            <p className="text-sm text-warmgray">Not applicable — no prediction was produced.</p>
          )}
        </Section>
      </div>
    </div>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-4 border-t border-linen pt-3">
      <h4 className="text-xs font-semibold uppercase tracking-[0.12em] text-warmgray">{title}</h4>
      <div className="mt-2">{children}</div>
    </div>
  )
}

/* --------------------------------- charts --------------------------------- */

type ChartRow = {
  event: PredictionEventRecord
  predicted: number
  actual: number
  signedError: number | null
  absError: number | null
  relError: number | null
  status: string | null
}

/** Two compact SVG views over the same filtered rows: predicted-vs-actual and
 * signed error with a zero baseline. Hover reveals full per-point detail. */
function HistoryChart({ rows }: { rows: ChartRow[] }) {
  const [mode, setMode] = useState<'compare' | 'error'>('compare')
  const [hover, setHover] = useState<number | null>(null)
  const W = 760
  const H = 230
  const PAD = 44
  const maxVal =
    mode === 'compare'
      ? Math.max(...rows.flatMap((r) => [r.predicted, r.actual]), 1)
      : Math.max(...rows.map((r) => Math.abs(r.signedError ?? 0)), 1e-9)
  const x = (i: number) => PAD + (i * (W - 2 * PAD)) / Math.max(rows.length - 1, 1)
  const y = (v: number) => H - PAD - (v / maxVal) * (H - 2 * PAD - (mode === 'error' ? H / 4 : 0))
  const yErr = (v: number) => {
    // signed error: zero line centered in the plot area
    const top = PAD - 10
    const bottom = H - PAD + 6
    const half = (bottom - top) / 2
    return top + half - (v / maxVal) * half
  }
  const yOf = mode === 'compare' ? y : yErr
  const line = (vals: (number | null)[]) =>
    vals
      .map((v, i) => (v === null ? null : `${x(i)},${yOf(v)}`))
      .filter(Boolean)
      .join(' ')

  return (
    <div data-testid="timeline-chart">
      <div className="mb-2 flex items-center gap-3 text-xs">
        <button
          type="button"
          className={(mode === 'compare' ? 'font-semibold text-bronze' : 'text-warmgray') + ' hover:underline'}
          onClick={() => setMode('compare')}
        >
          Prediction vs actual
        </button>
        <button
          type="button"
          className={(mode === 'error' ? 'font-semibold text-bronze' : 'text-warmgray') + ' hover:underline'}
          onClick={() => setMode('error')}
          data-testid="error-toggle"
        >
          Prediction error
        </button>
        <span className="ml-auto text-warmgray">
          {mode === 'compare' ? (
            <>
              <span className="mr-2 inline-block h-1.5 w-4 rounded bg-bronze" /> prediction
              <span className="ml-3 mr-2 inline-block h-1.5 w-4 rounded bg-sage" /> actual
            </>
          ) : (
            'signed deviation from actual (above zero = over-prediction)'
          )}
        </span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Predicted versus actual execution time over time">
        {mode === 'compare'
          ? [0, 0.5, 1].map((f) => (
              <g key={f}>
                <line x1={PAD} x2={W - PAD} y1={y(maxVal * f)} y2={y(maxVal * f)} stroke="#efece4" />
                <text x={4} y={y(maxVal * f) + 3} fontSize="9" fill="#8a8676">
                  {fmtAxis(maxVal * f)}
                </text>
              </g>
            ))
          : (
            <g>
              <line x1={PAD} x2={W - PAD} y1={yErr(0)} y2={yErr(0)} stroke="#8a8676" strokeDasharray="3 3" />
              <text x={4} y={yErr(0) + 3} fontSize="9" fill="#8a8676">0</text>
              <text x={4} y={yErr(maxVal) + 3} fontSize="9" fill="#8a8676">+{fmtAxis(maxVal)}</text>
              <text x={4} y={yErr(-maxVal) + 3} fontSize="9" fill="#8a8676">−{fmtAxis(maxVal)}</text>
            </g>
          )}
        {mode === 'compare' ? (
          <>
            <polyline points={line(rows.map((r) => r.predicted))} fill="none" stroke="#c6a15b" strokeWidth="1.5" />
            <polyline points={line(rows.map((r) => r.actual))} fill="none" stroke="#7d9b76" strokeWidth="1.5" />
            {rows.map((r, i) => (
              <g key={r.event.event_id}>
                <circle cx={x(i)} cy={y(r.predicted)} r={hover === i ? 4 : 2.5} fill="#c6a15b" />
                <circle cx={x(i)} cy={y(r.actual)} r={hover === i ? 4 : 2.5} fill="#7d9b76" />
                <rect
                  x={x(i) - 8} y={0} width={16} height={H} fill="transparent"
                  onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)}
                />
              </g>
            ))}
          </>
        ) : (
          <>
            <polyline points={line(rows.map((r) => r.signedError))} fill="none" stroke="#b0705f" strokeWidth="1.5" />
            {rows.map((r, i) => (
              <g key={r.event.event_id}>
                <circle
                  cx={x(i)} cy={yErr(r.signedError ?? 0)} r={hover === i ? 4 : 2.5}
                  fill={(r.signedError ?? 0) >= 0 ? '#b0705f' : '#7d9b76'}
                />
                <rect x={x(i) - 8} y={0} width={16} height={H} fill="transparent" onMouseEnter={() => setHover(i)} onMouseLeave={() => setHover(null)} />
              </g>
            ))}
          </>
        )}
        <text x={PAD} y={H - 6} fontSize="9" fill="#8a8676">{fmtWhen(rows[0].event.created_at)}</text>
        <text x={W - PAD} y={H - 6} fontSize="9" fill="#8a8676" textAnchor="end">{fmtWhen(rows[rows.length - 1].event.created_at)}</text>
      </svg>
      {hover !== null && rows[hover] && (
        <div className="mt-2 rounded-md border border-linen bg-[#fbf8f1] p-2 text-xs" data-testid="chart-hover-detail">
          <p className="font-semibold text-ink">{rows[hover].event.request?.loader_name}</p>
          <p className="text-warmgray">{fmtWhen(rows[hover].event.created_at)}</p>
          <p>Prediction: {formatSeconds(rows[hover].predicted)}</p>
          <p>Actual: {formatSeconds(rows[hover].actual)}</p>
          {rows[hover].signedError !== null && <p>Error: {fmtSigned(rows[hover].signedError!)}</p>}
          {rows[hover].absError !== null && <p>Absolute error: {formatSeconds(rows[hover].absError)}</p>}
          {rows[hover].relError !== null && (
            <p>
              Relative error: {rows[hover].relError!.toFixed(1)}% — prediction was{' '}
              {Math.abs(rows[hover].relError!).toFixed(1)}%{' '}
              {rows[hover].relError! > 0 ? 'lower' : 'higher'} than actual
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function fmtAxis(v: number): string {
  if (v >= 86400) return `${(v / 86400).toFixed(0)}d`
  if (v >= 3600) return `${(v / 3600).toFixed(0)}h`
  if (v >= 60) return `${(v / 60).toFixed(0)}m`
  return `${v.toFixed(0)}s`
}

function fmtSigned(v: number): string {
  return `${v >= 0 ? '+' : '−'}${formatSeconds(Math.abs(v))}`
}

function fmtWhen(iso: string | null): string {
  if (!iso) return '—'
  const d = new Date(iso)
  if (Number.isNaN(d.getTime())) return String(iso)
  return d.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' })
}
