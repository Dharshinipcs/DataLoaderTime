import { useEffect, useMemo, useState } from 'react'
import {
  ApiError,
  type LoaderDirectoryRow,
  type LoaderIntelResponse,
  fetchLoaderDirectory,
  fetchLoaderIntelligence,
} from '../api'
import { formatSeconds } from '../lib/format'

type Phase =
  | { kind: 'idle' }
  | { kind: 'loading'; name: string }
  | { kind: 'done'; data: LoaderIntelResponse }
  | { kind: 'error'; message: string }

type Filters = {
  name: string
  dataset: string
  connection: string
  sprint: string
  status: string
  intelligence: string
}

const EMPTY_FILTERS: Filters = { name: '', dataset: '', connection: '', sprint: '', status: '', intelligence: '' }

export default function LoaderIntelligence() {
  const [query, setQuery] = useState('')
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' })
  const [directory, setDirectory] = useState<LoaderDirectoryRow[] | null>(null)
  const [dirError, setDirError] = useState<string | null>(null)
  const [filters, setFilters] = useState<Filters>(EMPTY_FILTERS)
  const [moreOpen, setMoreOpen] = useState(false)

  useEffect(() => {
    // 2026-10-05 (user spec): refresh the directory when the tab regains focus
    // so newly saved executions update loader stats immediately.
    const load = () =>
      fetchLoaderDirectory()
        .then((d) => setDirectory(d.loaders))
        .catch(() => setDirError('The loader directory is unavailable right now.'))
    load()
    window.addEventListener('focus', load)
    document.addEventListener('visibilitychange', load)
    return () => {
      window.removeEventListener('focus', load)
      document.removeEventListener('visibilitychange', load)
    }
  }, [])

  async function lookupWith(name: string) {
    if (!name.trim()) return
    setPhase({ kind: 'loading', name })
    try {
      const data = await fetchLoaderIntelligence(name)
      setPhase({ kind: 'done', data })
    } catch (err: unknown) {
      setPhase({
        kind: 'error',
        message:
          err instanceof ApiError
            ? err.message
            : 'Unexpected error while contacting the backend.',
      })
    }
  }

  const activeChips = useMemo(
    () =>
      Object.entries(filters)
        .filter(([, v]) => v)
        .map(([k, v]) => ({ key: k, value: v })),
    [filters],
  )

  return (
    <main className="mx-auto max-w-6xl">
      <h1 className="font-serif text-2xl font-bold tracking-tight text-ink">
        Loader Intelligence
      </h1>

      {/* Compact filter bar: one search box (name/dataset/connection/sprint)
          + two high-value dropdowns + More Filters. */}
      <form
        className="mt-4 flex flex-wrap items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault()
          const name = query.trim()
          if (name) {
            setFilters({ ...filters, name })
            void lookupWith(name)
          }
        }}
      >
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Search loader, dataset, connection or sprint…"
          aria-label="Loader name"
          className="field-input min-w-56 !w-auto flex-1 basis-56"
        />
        <FilterSelect
          label="Sprint"
          value={filters.sprint}
          onChange={(v) => setFilters({ ...filters, sprint: v })}
          options={distinctValues(directory, (l) => l.sprints ?? [])}
        />
        <FilterSelect
          label="Connection"
          value={filters.connection}
          onChange={(v) => setFilters({ ...filters, connection: v })}
          options={distinctValues(directory, (l) => l.connections ?? [])}
        />
        <button
          type="button"
          onClick={() => setMoreOpen((o) => !o)}
          aria-expanded={moreOpen}
          className="btn-secondary whitespace-nowrap"
          data-testid="more-filters-toggle"
        >
          More Filters
        </button>
        <button
          type="submit"
          disabled={phase.kind === 'loading' || !query.trim()}
          className="btn-primary whitespace-nowrap"
        >
          {phase.kind === 'loading' ? 'Looking up…' : 'Look up loader'}
        </button>
      </form>

      {moreOpen && (
        <div className="mt-2 flex flex-wrap items-center gap-2 rounded-md border border-linen bg-[#fbf8f1] p-2" data-testid="more-filters">
          <FilterSelect
            label="Dataset"
            value={filters.dataset}
            onChange={(v) => setFilters({ ...filters, dataset: v })}
            options={distinctValues(directory, (l) => l.datasets ?? [])}
          />
          <FilterSelect
            label="Outcome"
            value={filters.status}
            onChange={(v) => setFilters({ ...filters, status: v })}
            options={[
              ['mostly successful', 'Mostly successful'],
              ['has failures', 'Has failures'],
              ['new', 'New (<3 executions)'],
              ['established', 'Established (3+)'],
            ]}
          />
          <FilterSelect
            label="Intelligence"
            value={filters.intelligence}
            onChange={(v) => setFilters({ ...filters, intelligence: v })}
            options={[
              ['available', 'Intelligence available'],
              ['limited', 'Limited intelligence'],
              ['no history', 'No history'],
            ]}
          />
        </div>
      )}

      {activeChips.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-2 text-xs" data-testid="active-filters">
          <span className="text-warmgray">Filters:</span>
          {activeChips.map((c) => (
            <button
              key={c.key}
              type="button"
              className="rounded-full bg-[#f3ead9] px-2 py-0.5 font-medium text-ochre hover:bg-[#eadfc8]"
              onClick={() => setFilters({ ...filters, [c.key]: '' })}
              title="Click to clear this filter"
            >
              {c.key}: {c.value} ✕
            </button>
          ))}
          <button
            type="button"
            className="text-warmgray underline hover:text-ink"
            onClick={() => setFilters(EMPTY_FILTERS)}
          >
            clear all
          </button>
        </div>
      )}

      {phase.kind === 'error' && (
        <div role="alert" className="mt-4 rounded-md border border-[#e3c6c1] bg-[#f6e7e4] p-4 text-sm text-ember">
          {phase.message}
        </div>
      )}

      {phase.kind === 'loading' && (
        <p className="mt-4 text-sm text-warmgray" role="status">
          Assembling intelligence for “{phase.name}”…
        </p>
      )}
      {phase.kind === 'done' && <SimpleIntelPanel data={phase.data} />}

      {/* Loader directory — loader-level rollups, never prediction events. */}
      <div className="mt-8">
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-warmgray">
          Loader directory
        </h2>
        {dirError && <p className="mt-2 text-sm text-ember" data-testid="directory-error">{dirError}</p>}
        {directory && (
          <DirectoryTable
            loaders={directory}
            filters={filters}
            onPick={(n) => {
              setQuery(n)
              void lookupWith(n)
            }}
          />
        )}
      </div>
    </main>
  )
}

function FilterSelect({
  label,
  value,
  onChange,
  options,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  options: (string | [string, string])[]
}) {
  return (
    <select
      className="field-input !w-auto"
      aria-label={`${label} filter`}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    >
      <option value="">{label}: any</option>
      {options.map((o) => {
        const [val, lab] = Array.isArray(o) ? o : [o, o]
        return (
          <option key={val} value={val}>
            {lab}
          </option>
        )
      })}
    </select>
  )
}

function distinctValues(
  directory: LoaderDirectoryRow[] | null,
  pick: (l: LoaderDirectoryRow) => string[],
): string[] {
  if (!directory) return []
  return [...new Set(directory.flatMap(pick))].filter(Boolean).sort().slice(0, 25)
}

/* ------------------------------------------------------------------ */

/* ------------------------- simple inline panel ---------------------- */

/** 2026-10-03 (user final spec): clicking a loader opens a SIMPLE
 * intelligence panel right here, directly after the filters — execution-time
 * trend, execution counts by outcome (completed successfully / with warnings
 * / killed), typical and recent execution time, and the typical min-max
 * total range. Stage intelligence, historical medians and historical
 * configurations were removed from this view on user request. */
function SimpleIntelPanel({ data }: { data: LoaderIntelResponse }) {
  const b = data.behavior
  const series = b?.series ?? []
  const dist = b?.distribution
  const ok = series.filter((p) => p.status === 'SUCCEED').length
  const warned = series.filter((p) => p.status && p.status.includes('WARNING')).length
  const killed = series.filter(
    (p) => p.status && p.status !== 'SUCCEED' && !p.status.includes('WARNING'),
  ).length
  const rangeMin = dist?.min_sec ?? (series.length ? Math.min(...series.map((p) => p.seconds)) : null)
  const rangeMax = dist?.max_sec ?? (series.length ? Math.max(...series.map((p) => p.seconds)) : null)
  return (
    <div className="mt-5 space-y-4" data-testid="loader-intel-card">
      <div className="card p-5">
        <div className="flex flex-wrap items-center gap-2">
          <h2 className="font-serif text-lg font-bold text-ink" data-testid="intel-loader-name">
            {data.loader_name}
          </h2>
          {data.known ? (
            <span className="rounded bg-[#efece4] px-2 py-0.5 text-xs text-warmgray">known loader</span>
          ) : (
            <span className="rounded bg-[#f1e0de] px-2 py-0.5 text-xs text-ember">unknown to definitions</span>
          )}
          {data.eta_eligible ? (
            <span className="rounded bg-[#e7ebe6] px-2 py-0.5 text-xs font-medium text-forest" data-testid="intel-eta-eligible">
              Prediction available
            </span>
          ) : (
            <span className="rounded bg-[#f3ead9] px-2 py-0.5 text-xs font-medium text-ochre" data-testid="intel-eta-eligible">
              Insufficient historical evidence
            </span>
          )}
        </div>

        <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
          <Fact label="Historical executions" value={String(data.history.eligible_execution_count)} testId="intel-exec-count" />
          <Fact label="Completed successfully" value={String(ok)} testId="intel-success-count" />
          <Fact label="Completed with warnings" value={String(warned)} testId="intel-warn-count" />
          <Fact label="Killed / failed" value={String(killed)} testId="intel-killed-count" />
          <Fact label="Typical execution time" value={formatSeconds(b?.median_sec ?? null)} testId="intel-typical" />
          <Fact label="Recent execution time" value={formatSeconds(b?.recent_median_sec ?? null)} testId="intel-recent" />
          <Fact
            label="Typical total range (min-max)"
            value={rangeMin != null && rangeMax != null ? `${formatSeconds(rangeMin)} - ${formatSeconds(rangeMax)}` : '—'}
            testId="intel-range"
          />
        </div>

        <div className="mt-4">
          <h3 className="text-xs font-semibold uppercase tracking-[0.12em] text-warmgray">
            Execution time trend
          </h3>
          <div className="mt-2">
            {b?.series && b.series.length >= 2 ? (
              <TrendChart series={b.series.slice(-30)} />
            ) : (
              <p className="text-sm text-warmgray">
                {b && b.n === 1
                  ? 'One recorded execution — a trend needs at least two.'
                  : 'No recorded execution durations for this loader yet.'}
              </p>
            )}
          </div>
        </div>

        {data.limitations.length > 0 && (
          <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-ochre">
            {data.limitations.map((lim) => (
              <li key={lim}>{lim}</li>
            ))}
          </ul>
        )}
      </div>
    </div>
  )
}

/* ------------------------------ shared ----------------------------- */

/** Execution-time trend for the simple panel — one point per recorded
 * execution, colored by outcome (success sage, warning ochre, killed red). */
function TrendChart({ series }: { series: { start_time: string; seconds: number; status: string | null }[] }) {
  const W = 640
  const H = 170
  const pad = { l: 44, r: 8, t: 10, b: 22 }
  const max = Math.max(...series.map((p) => p.seconds)) * 1.1 || 1
  const x = (i: number) => pad.l + (i * (W - pad.l - pad.r)) / Math.max(1, series.length - 1)
  const y = (v: number) => pad.t + (1 - v / max) * (H - pad.t - pad.b)
  const color = (s: string | null) =>
    s === 'SUCCEED' ? '#7a9471' : s && s.includes('WARNING') ? '#c08a3e' : '#b3574d'
  const pts = series.map((p, i) => `${x(i)},${y(p.seconds)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" role="img" aria-label="Execution time trend">
      <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="#e4ddcf" />
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="#e4ddcf" />
      <text x={4} y={pad.t + 8} fontSize="10" fill="#8a8378">{Math.round(max)}s</text>
      <text x={4} y={H - pad.b} fontSize="10" fill="#8a8378">0s</text>
      <polyline points={pts} fill="none" stroke="#9a6b3f" strokeWidth="2" />
      {series.map((p, i) => (
        <circle key={i} cx={x(i)} cy={y(p.seconds)} r="3" fill={color(p.status)}>
          <title>{`${p.start_time} — ${formatSeconds(p.seconds)} (${p.status ?? 'status unknown'})`}</title>
        </circle>
      ))}
      <text x={pad.l} y={H - 4} fontSize="10" fill="#8a8378">{series[0].start_time.slice(0, 16)}</text>
      <text x={W - pad.r} y={H - 4} fontSize="10" fill="#8a8378" textAnchor="end">
        {series[series.length - 1].start_time.slice(0, 16)}
      </text>
    </svg>
  )
}

function Fact({ label, value, testId }: { label: string; value: string; testId?: string }) {
  return (
    <div>
      <dt className="text-[11px] uppercase tracking-wide text-warmgray">{label}</dt>
      <dd className="text-sm font-medium text-ink" data-testid={testId}>
        {value}
      </dd>
    </div>
  )
}

/* --------------------------- directory ----------------------------- */

function intelligenceLabel(l: LoaderDirectoryRow): { label: string; cls: string } {
  if (l.executions === 0 || l.median_duration_sec == null)
    return { label: 'No history', cls: 'bg-[#efece4] text-warmgray' }
  if (l.executions >= 3) return { label: 'Available', cls: 'bg-[#e7ebe6] text-forest' }
  return { label: 'Limited', cls: 'bg-[#f3ead9] text-ochre' }
}

function DirectoryTable({
  loaders,
  filters,
  onPick,
}: {
  loaders: LoaderDirectoryRow[]
  filters: Filters
  onPick: (name: string) => void
}) {
  const [page, setPage] = useState(0)
  // 2026-10-04 (user spec): 10 rows per view with prev/next paging instead of
  // a long scroll.
  const PER_VIEW = 10
  const filtered = useMemo(() => {
    const q = (s: string) => s.toLowerCase()
    return loaders.filter((l) => {
      const haystack = q(
        [l.loader_name, ...(l.datasets ?? []), ...(l.connections ?? []), ...(l.sprints ?? [])].join(' '),
      )
      if (filters.name && !haystack.includes(q(filters.name))) return false
      if (filters.dataset && !(l.datasets ?? []).some((d) => q(d).includes(q(filters.dataset)))) return false
      if (filters.connection && !(l.connections ?? []).some((c) => q(c).includes(q(filters.connection)))) return false
      if (filters.sprint && !(l.sprints ?? []).some((s) => q(s).includes(q(filters.sprint)))) return false
      if (filters.status === 'mostly successful' && l.failed > 0) return false
      if (filters.status === 'has failures' && l.failed === 0) return false
      if (filters.status === 'new' && l.executions >= 3) return false
      if (filters.status === 'established' && l.executions < 3) return false
      const intel = intelligenceLabel(l).label.toLowerCase()
      if (filters.intelligence && intel !== filters.intelligence) return false
      return true
    })
  }, [loaders, filters])

  const pageCount = Math.max(1, Math.ceil(filtered.length / PER_VIEW))
  const safePage = Math.min(page, pageCount - 1)
  const rows = filtered.slice(safePage * PER_VIEW, (safePage + 1) * PER_VIEW)

  return (
    <div data-testid="loader-directory">
      {filtered.length === 0 ? (
        <p className="mt-3 text-sm text-warmgray">No loaders match these filters.</p>
      ) : (
        <>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr>
                  {/* 2026-10-03: the Intelligence COLUMN was removed (the
                      intelligence filter stays). */}
                  {['Loader', 'Dataset', 'Connection', 'Sprint', 'Executions', 'Typical duration'].map((h) => (
                    <th key={h} className="border-b border-linen py-2 pr-3 text-[11px] font-semibold uppercase tracking-wide text-warmgray">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((l) => {
                  return (
                    <tr key={l.loader_name} data-testid="directory-row">
                      <td className="border-b border-[#efece4] py-2 pr-3">
                        <button
                          type="button"
                          className="font-medium text-ink hover:text-bronze hover:underline"
                          onClick={() => onPick(l.loader_name)}
                        >
                          {l.loader_name}
                        </button>
                      </td>
                      <td className="max-w-48 truncate border-b border-[#efece4] py-2 pr-3 text-warmgray" title={(l.datasets ?? []).join(', ')}>
                        {(l.datasets ?? []).join(', ') || '—'}
                      </td>
                      <td className="border-b border-[#efece4] py-2 pr-3 text-warmgray" title={(l.connections ?? []).join(', ')}>
                        {(l.connections ?? []).join(', ') || '—'}
                      </td>
                      <td className="border-b border-[#efece4] py-2 pr-3 text-warmgray" title={(l.sprints ?? []).join(', ')}>
                        {(l.sprints ?? []).join(', ') || '—'}
                      </td>
                      <td className="border-b border-[#efece4] py-2 pr-3">{l.executions}</td>
                      <td className="border-b border-[#efece4] py-2 pr-3">{formatSeconds(l.median_duration_sec)}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
          {pageCount > 1 && (
            <div className="mt-2 flex items-center gap-3 text-xs" data-testid="directory-pager">
              <button
                type="button"
                className="btn-secondary !py-1 disabled:opacity-40"
                disabled={safePage === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
              >
                ← Prev 10
              </button>
              <span className="text-warmgray">
                Page {safePage + 1} of {pageCount} · {filtered.length} loaders
              </span>
              <button
                type="button"
                className="btn-secondary !py-1 disabled:opacity-40"
                disabled={safePage >= pageCount - 1}
                onClick={() => setPage((p) => Math.min(pageCount - 1, p + 1))}
              >
                Next 10 →
              </button>
            </div>
          )}
        </>
      )}
    </div>
  )
}
