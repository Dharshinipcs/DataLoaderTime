import { useEffect, useState } from 'react'
import {
  ApiError,
  type CandidateEvaluation,
  type DriftOverview,
  type LatestCandidate,
  type ModelRecord,
  type ModelsStatus,
  type RetrainingDecision,
  type ServingStatus,
  type StageLineageResponse,
  type TrainCandidateResponse,
  fetchDriftOverview,
  fetchModels,
  fetchModelsLifecycle,
  fetchModelsStatus,
  fetchRetrainingDecisions,
  fetchServingStatus,
  fetchStageLineage,
  postModelPromote,
  postModelRollback,
  postRetrainingDecision,
  postTrainCandidate,
} from '../api'
import { formatSeconds } from '../lib/format'
import { Badge, Card, PageHeader, type BadgeTone } from '../components/ui'

type Phase =
  | { kind: 'loading' }
  | { kind: 'done'; data: PageData }
  | { kind: 'error'; message: string }

interface PageData {
  overview: DriftOverview
  models: ModelRecord[]
  status: ModelsStatus
  lifecycle: LifecycleEvent[]
  decisions: RetrainingDecision[]
  serving: ServingStatus
  lineage: StageLineageResponse | null
  evaluation: CandidateEvaluation | null
}

interface LifecycleEvent {
  at: string
  model_id: string
  event_type: string
  kind: string
  label: string
  reason?: string
}

/* ---------- human vocabulary (locked plan §4/§8/§17) --------------------- */

/* 2026-10-05 (user final spec): every status is written in plain
 * language a non-ML operator understands. Raw codes never reach the
 * UI; the statistical detail behind each claim stays one click away
 * (drill-through) for the ML expert. */
function healthLabel(state: string): { label: string; tone: BadgeTone } {
  switch (state) {
    case 'STABLE':
      return { label: 'Stable', tone: 'success' }
    case 'WATCH':
      return { label: 'Keep an eye', tone: 'warning' }
    case 'DRIFT_DETECTED':
      return { label: 'Data has drifted', tone: 'risk' }
    case 'PERFORMING_NORMALLY':
      return { label: 'Performing normally', tone: 'success' }
    case 'PERFORMANCE_CHANGING':
      return { label: 'Performance is changing', tone: 'warning' }
    case 'DEGRADATION_DETECTED':
      return { label: 'Performance is degrading', tone: 'risk' }
    case 'INSUFFICIENT_DATA':
      return { label: 'Not enough data yet', tone: 'neutral' }
    case 'INSUFFICIENT_EVIDENCE':
      return { label: 'Not enough evidence yet', tone: 'neutral' }
    case 'INSUFFICIENT_ACTUAL_OUTCOMES':
      return { label: 'Not enough actual outcomes yet', tone: 'neutral' }
    default:
      return { label: state.toLowerCase().replace(/_/g, ' '), tone: 'neutral' }
  }
}
void healthLabel

function driftStatusLabel(status: string): { label: string; tone: BadgeTone } {
  if (status.includes('SIGNIFICANT')) return { label: 'Data has drifted', tone: 'risk' }
  if (status.includes('WARNING')) return { label: 'Keep an eye', tone: 'warning' }
  if (status.includes('INSUFFICIENT')) return { label: 'Not enough data yet', tone: 'neutral' }
  return { label: 'Stable', tone: 'success' }
}
void driftStatusLabel

/** Plain-language one-line summaries for the top of the page —
 * what a non-ML operator needs to know at a glance. The evidence
 * (tables, statistics) is kept one drill-through below. */
/* ---------- component ---------------------------------------------------- */

export default function DriftRetraining() {
  const [phase, setPhase] = useState<Phase>({ kind: 'loading' })

  async function load() {
    setPhase({ kind: 'loading' })
    try {
      // 2026-10-04 (user spec): system scope only — per-loader drill-down lives
      // in Loader Intelligence / Prediction History.
      const overview = await fetchDriftOverview(undefined)
      const [models, status, lifecycle, decisions, serving, lineage] = await Promise.all([
        fetchModels(),
        fetchModelsStatus(),
        fetchModelsLifecycle(),
        fetchRetrainingDecisions().catch(() => []),
        fetchServingStatus().catch(() => ({ serving_version: '—', candidate_override_active: false, candidate_version: null })),
        fetchStageLineage().catch(() => null),
      ])
      setPhase({
        kind: 'done',
        data: {
          overview,
          models: models.models,
          status,
          lifecycle: lifecycle.events,
          decisions,
          serving,
          lineage,
          evaluation: null,
        },
      })
    } catch (err: unknown) {
      setPhase({
        kind: 'error',
        message:
          err instanceof ApiError ? err.message : 'Unexpected error while contacting the backend.',
      })
    }
  }

  useEffect(() => {
    // 2026-10-06 (user rework spec): NO 30-second timer — full retraining can
    // run for minutes and an automatic page refresh mid-retrain is unnecessary
    // and disruptive. Live updates remain: refetch on tab focus/visibility
    // plus React state updates after every action (retrain/promote/rollback).
    void load()
    const onFocus = () => void load()
    window.addEventListener('focus', onFocus)
    document.addEventListener('visibilitychange', onFocus)
    return () => {
      window.removeEventListener('focus', onFocus)
      document.removeEventListener('visibilitychange', onFocus)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 2026-10-05 (user spec): the candidate produced by "Retrain now"
  // feeds the left-side stage comparison table. The state lives here
  // (above Body) so it survives the post-decision refresh.
  const [candidate, setCandidate] = useState<TrainCandidateResponse | null>(null)

  return (
    <div>
      <PageHeader
        eyebrow=""
        title="Monitoring"
      />
      {phase.kind === 'loading' && (
        <div
          role="status"
          aria-live="polite"
          className="flex min-h-40 items-center justify-center rounded-md border border-linen bg-parchment p-8"
        >
          <div className="text-center">
            <div className="mx-auto h-7 w-7 animate-spin rounded-full border-2 border-bronze border-t-transparent" />
            <p className="mt-3 text-sm text-warmgray">Assembling drift and retraining evidence…</p>
          </div>
        </div>
      )}
      {phase.kind === 'error' && (
        <div role="alert" className="rounded-md border border-[#e3c6c1] bg-[#f6e7e4] p-4 text-sm text-ember">
          {phase.message}
        </div>
      )}
      {phase.kind === 'done' && (
        <Body data={phase.data} onReload={load} candidate={candidate} onCandidate={setCandidate} />
      )}
    </div>
  )
}

function Body({
  data,
  onReload,
  candidate,
  onCandidate,
}: {
  data: PageData
  onReload: () => void
  candidate: TrainCandidateResponse | null
  onCandidate: (r: TrainCandidateResponse) => void
}) {
  const { overview, status } = data
  const a = overview.assessment
  const evaluation = data.evaluation ?? (candidate ? unwrapEval(candidate.evaluation, '') : null)
  const [stageNotice, setStageNotice] = useState<string | null>(null)

  // ---- per-stage retraining / promotion / rollback (2026-10-05).
  // A single-stage candidate trains with the identical protocol
  // scoped to ONE stage; promoting it activates ONLY that stage's
  // serving override; rolling back resets only that stage. Every
  // action is human-initiated — nothing is automatic.
  async function retrainStage(stage: string) {
    setStageNotice(null)
    try {
      const result = await postTrainCandidate({
        approved_by: 'local-operator',
        reason: `Single-stage retraining for ${stage} (human-approved from Drift & Retraining)`,
        stages: [stage],
      })
      onCandidate(result)
      const cid = String(
        (Array.isArray(result.registered) ? result.registered[result.registered.length - 1] : result.registered)
          ?.model_id ?? '',
      )
      setStageNotice(
        `Candidate ${cid || 'trained'} created for ${stageLabel(stage)} — see "Model versions". Serving is untouched; promote it explicitly.`,
      )
      onReload()
    } catch (err: unknown) {
      setStageNotice(err instanceof ApiError ? err.message : 'Stage retraining failed.')
    }
  }

  async function promoteStage(stage: string, modelId: string) {
    setStageNotice(null)
    try {
      await postModelPromote({
        model_id: modelId,
        reason: `Promote stage candidate for ${stage} (human-approved from Drift & Retraining)`,
        stages: [stage],
      })
      setStageNotice(`Promoted ${modelId} — only ${stage.replace('_', ' ')} switched; other stages keep serving their current models.`)
      onReload()
    } catch (err: unknown) {
      setStageNotice(err instanceof ApiError ? err.message : 'Stage promotion failed.')
    }
  }

  async function rollbackStage(stage: string) {
    setStageNotice(null)
    try {
      await postModelRollback({
        reason: `Roll back stage ${stage} (human-approved from Drift & Retraining)`,
        stages: [stage],
      })
      setStageNotice(`Rolled back ${stage.replace('_', ' ')} — the previous stage model serves again.`)
      onReload()
    } catch (err: unknown) {
      setStageNotice(err instanceof ApiError ? err.message : 'Stage rollback failed.')
    }
  }

  return (
    <div className="grid gap-4 lg:grid-cols-5">
      {/* LEFT (3/5): recommendation → stage model error comparison → human
          decision → view timeline (2026-10-08 user spec). */}
      <div className="space-y-4 lg:col-span-3">
        <RecommendationCard
          overview={overview}
          evaluation={evaluation}
          latestCandidate={overview.retraining.latest_candidate ?? null}
        />
        {stageNotice && (
          <p className="text-xs text-bronze" role="status" data-testid="stage-notice">
            {stageNotice}
          </p>
        )}
        <Card title="Stage model error comparison" testId="stage-error-comparison">
          <StageComparison
            stageHealth={overview.model_performance.stage_health}
            models={data.models}
            lineage={data.lineage}
            onRetrainStage={retrainStage}
            onPromoteStage={promoteStage}
            onRollbackStage={rollbackStage}
          />
        </Card>
        <Card testId="human-decision-card">
          <GovernanceActions onDone={onReload} />
        </Card>
        <ViewTimelineCard events={data.lifecycle} decisions={data.decisions} />
      </div>

      {/* RIGHT (2/5): retraining actions + model versions (scrollable). */}
      <div className="space-y-4 lg:col-span-2">
        <Card testId="retraining-assessment">
          <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-warmgray">Retraining</p>
          <ul className="mt-2 space-y-1 text-sm text-ink" data-testid="retraining-signals">
            <li>
              New outcomes: <strong>{overview.retraining.signals.eligible_new_actuals.count} of {overview.retraining.signals.eligible_new_actuals.threshold} needed</strong>
            </li>
            <li>
              Performance check:{' '}
              {overview.retraining.signals.performance_degradation.assessed
                ? 'assessed against the current model version'
                : 'needs more matched outcomes'}
            </li>
          </ul>
          <RetrainingActions
            verdict={a.verdict}
            onDone={onReload}
            evaluation={data.evaluation}
            onCandidate={onCandidate}
            degradedStages={recommendationFor(overview, data.evaluation).degradedStages}
          />
        </Card>

        <Card testId="model-history">
          <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-warmgray">Model versions</p>
          <p className="mt-1 text-sm" data-testid="serving-model-version">
            Serving: {status.serving_model_version}
            {data.serving.candidate_override_active ? ` → candidate ${data.serving.candidate_version}` : ''}
          </p>
          {(() => {
            const newestCandidate = data.models
              .filter((m) => String(m.lifecycle_state ?? m.status ?? '').toUpperCase() === 'CANDIDATE_EVALUATED')
              .sort((x, y) => String(y.created_at ?? '').localeCompare(String(x.created_at ?? '')))[0]
            return newestCandidate ? (
              <p className="mt-1 text-xs text-bronze" data-testid="latest-candidate-summary">
                Candidate awaiting decision:{' '}
                <strong>{String(newestCandidate.model_id ?? '')}</strong>
                {newestCandidate.stage ? ` (${stageLabel(String(newestCandidate.stage))})` : ''} — use "Use for
                promote" on its row below.
              </p>
            ) : null
          })()}
          <ModelVersionTable models={data.models} />
        </Card>
      </div>
    </div>
  )
}

/** Collapsed "View Timeline" section (2026-10-08 user spec): the existing
 * history/timeline, hidden until the header is clicked. Content and data are
 * unchanged — presentation only. */
function ViewTimelineCard({
  events,
  decisions,
}: {
  events: LifecycleEvent[]
  decisions: RetrainingDecision[]
}) {
  const [open, setOpen] = useState(false)
  return (
    <Card testId="view-timeline-card">
      <button
        type="button"
        data-testid="view-timeline-toggle"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between text-left"
      >
        <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-warmgray">
          View Timeline
        </span>
        <span className="text-xs font-semibold text-bronze">{open ? 'Hide timeline' : 'Show timeline'}</span>
      </button>
      <div className={open ? 'mt-2' : 'mt-2 hidden'}>
        <CombinedTimeline events={events} decisions={decisions} />
      </div>
    </Card>
  )
}
/* ---------- system recommendation (2026-10-06 left-side rework) ---------- */

interface Recommendation {
  headline: string
  scope: 'NONE' | 'STAGES' | 'ALL'
  badge: { label: string; tone: BadgeTone }
  reasons: string[]
  actionsLine: string
  /** Stage keys recommended for individual retraining (empty = none/full). */
  degradedStages: string[]
}

/** Human stage labels keyed by the backend stage key (§ stage naming). */
const STAGE_LABELS: Record<string, string> = {
  staging: 'Staging',
  prevalidation: 'Prevalidation',
  transformation: 'Transformation',
  pre_load: 'Pre-load',
  data_loading: 'Data Loading',
}

function stageLabel(key: string): string {
  return STAGE_LABELS[key] ?? key.replace(/_/g, ' ')
}

/** The system's own retraining recommendation, built from the SAME evidence
 * the right-side assessment uses — degradation/watch stage health and the
 * retraining signals. Evidence-gated: with too few outcomes the honest
 * answer is "continue monitoring", never a fabricated alarm. */
function recommendationFor(
  overview: DriftOverview,
  evaluation: CandidateEvaluation | null,
): Recommendation {
  const a = overview.assessment
  const stages = overview.model_performance.stage_health.stages
  const degraded = Object.entries(stages)
    .filter(([, s]) => s.status === 'DEGRADATION_DETECTED' || s.status === 'PERFORMANCE_CHANGING')
    .map(([k]) => k)
  const perfOk = a.verdict === 'NO_RETRAINING_NEEDED'
  void evaluation
  if (perfOk && degraded.length === 0) {
    return {
      headline: 'Continue Monitoring — no stage model currently requires retraining.',
      scope: 'NONE',
      badge: { label: 'Continue Monitoring', tone: 'success' },
      reasons: [
        'Neither full retraining nor individual stage retraining is currently required: the serving models match the recorded outcomes within their normal error band, and no stage shows degradation.',
        a.narrative ? `Basis: ${a.narrative}` : 'Basis: the latest outcomes and drift checks show no change that a new model would fix.',
      ],
      actionsLine: 'Keep using the current models. The table below is watched automatically — this page updates as new matched outcomes arrive.',
      degradedStages: [],
    }
  }
  if (degraded.length >= 5) {
    return {
      headline: 'Full Retraining Recommended — all 5 stage models require retraining.',
      scope: 'ALL',
      badge: { label: 'Retrain Now', tone: 'risk' },
      reasons: [
        'All five stage models show performance change/degradation, so the whole set should be retrained together.',
        ...(a.narrative ? [`Basis: ${a.narrative}`] : []),
      ],
      actionsLine: 'Use "Retrain all 5 stage models" to retrain the whole bundle; promote after human review.',
      degradedStages: [],
    }
  }
  const names = degraded.map((k) => stageLabel(k)).join(', ')
  const scope = degraded.length > 0 ? 'STAGES' : 'ALL'
  return {
    headline:
      degraded.length > 0
        ? `Retraining Recommended — the ${names} model${degraded.length === 1 ? '' : 's'} require${degraded.length === 1 ? 's' : ''} retraining.`
        : 'Retraining Recommended — full retraining is the safer choice.',
    scope,
    badge: { label: 'Retrain Now', tone: 'warning' },
    reasons: [
      degraded.length > 0
        ? `Individual stage retraining is currently required for: ${names}. The other stage models are performing within their normal band and can keep serving.`
        : 'Neither the stage signals nor the aggregate assessment is clean enough to keep serving unchanged; full retraining of all five stage models is recommended.',
      ...(a.narrative ? [`Basis: ${a.narrative}`] : []),
    ],
    actionsLine:
      degraded.length > 0
        ? `Use "Retrain ${names}" below to retrain only ${degraded.length === 1 ? 'that stage model' : 'those stage models'}; use "Retrain all 5 stage models" if several degrade together.`
        : 'Use "Retrain all 5 stage models" below to retrain the whole bundle; promote after human review.',
    degradedStages: degraded.length > 0 ? degraded : [],
  }
}

function RecommendationCard({
  overview,
  evaluation,
  latestCandidate,
}: {
  overview: DriftOverview
  evaluation: CandidateEvaluation | null
  latestCandidate: LatestCandidate | null
}) {
  const rec = recommendationFor(overview, evaluation)
  void latestCandidate
  return (
    <Card testId="system-recommendation">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={rec.badge.tone} testId="recommendation-badge">
          {rec.badge.label}
        </Badge>
        <p className="text-base font-serif font-bold text-ink" data-testid="recommendation-headline">
          {rec.headline}
        </p>
      </div>
      <ul className="mt-2 space-y-1 text-sm text-ink" data-testid="recommendation-reasons">
        {rec.reasons.map((r, i) => (
          <li key={i}>{r}</li>
        ))}
      </ul>
      <p className="mt-2 text-xs text-warmgray" data-testid="recommendation-actions">
        {rec.actionsLine}
      </p>
    </Card>
  )
}


/* ---------- sub-components ------------------------------------------------ */

function StageComparison({
  stageHealth,
  models,
  lineage,
  onRetrainStage,
  onPromoteStage,
  onRollbackStage,
}: {
  stageHealth: DriftOverview['model_performance']['stage_health']
  models: ModelRecord[]
  lineage: StageLineageResponse | null
  onRetrainStage: (stage: string) => Promise<void>
  onPromoteStage: (stage: string, modelId: string) => Promise<void>
  onRollbackStage: (stage: string) => Promise<void>
}) {
  const [openStage, setOpenStage] = useState<string | null>(null)
  const order = ['staging', 'prevalidation', 'transformation', 'pre_load', 'data_loading'] as const
  // Which registry entry is PROMOTED per stage: full-bundle entries are
  // stage-scoped via their recorded stage field; single-stage candidates
  // name their stage explicitly. Only promoted stages show the badge.
  const promotedModels = models.filter(
    (m) =>
      String(
        (m as Record<string, unknown>).lifecycle_state ?? m.status ?? '',
      ).toUpperCase() === 'PROMOTED',
  )
  const promotedFor = (stage: string) =>
    promotedModels.filter((m) => {
      const st = m.stage ? String(m.stage) : null
      if (st) return st === stage
      const cands = (m as Record<string, unknown>).candidate_versions as
        | Record<string, string>
        | undefined
      return cands ? Object.keys(cands).includes(stage) : false
    })
  return (
    <div data-testid="stage-comparison">
      <table className="mt-1 w-full text-left text-sm">
        <thead>
          <tr>
            {['Stage model', 'Baseline error', 'Current serving error', 'Previous error'].map((h) => (
              <th
                key={h}
                className="border-b border-linen py-1.5 pr-3 text-[11px] font-semibold uppercase tracking-wide text-warmgray"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {order.map((key) => {
            const s = stageHealth.stages[key]
            if (!s) return null
            const post = s.post_promotion ?? null
            const line = lineage?.stages?.[key] ?? null
            const lineageBaseline = line?.baseline ?? null
            const prevModel = line?.previous ?? null
            const rowPromoted = promotedFor(key)
            const stageCandidate =
              (lineage as unknown as { active_candidate?: { kind?: string; candidate_versions?: Record<string, string> } | null })
                ?.active_candidate?.kind === 'stage-candidate'
                ? (lineage as unknown as { active_candidate: { candidate_versions: Record<string, string> } }).active_candidate
                    .candidate_versions[key] ?? null
                : null
            const warrantsRetrain =
              s.status === 'DEGRADATION_DETECTED' || s.status === 'PERFORMANCE_CHANGING'
            const baselineSeconds = s.baseline_medae_seconds ?? lineageBaseline?.holdout_medae ?? null
            return (
              <tr key={key} data-testid={`stage-row-${key}`}>
                <td className="border-b border-[#efece4] py-2 pr-3 font-medium text-ink">
                  <div className="flex flex-wrap items-center gap-1.5">
                    <button
                      type="button"
                      className="hover:underline"
                      onClick={() => setOpenStage(openStage === key ? null : key)}
                    >
                      {s.stage}
                    </button>
                    {rowPromoted.length > 0 && (
                      <span
                        className="rounded-sm bg-[#e4ecdd] px-1 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-sage"
                        title={rowPromoted.map((m) => String(m.model_id ?? '?')).join(', ')}
                      >
                        promoted
                      </span>
                    )}
                  </div>
                  <div className="mt-1 flex flex-wrap gap-1.5">
                    {warrantsRetrain && (
                      <button
                        type="button"
                        className="text-[11px] font-semibold text-bronze hover:underline"
                        onClick={() => void onRetrainStage(key)}
                        data-testid={`retrain-stage-${key}`}
                      >
                        Retrain this stage
                      </button>
                    )}
                    {stageCandidate && (
                      <>
                        <button
                          type="button"
                          className="text-[11px] font-semibold text-bronze hover:underline"
                          onClick={() => void onPromoteStage(key, stageCandidate)}
                          data-testid={`promote-stage-${key}`}
                        >
                          Promote stage candidate
                        </button>
                        <button
                          type="button"
                          className="text-[11px] font-semibold text-warmgray hover:underline"
                          onClick={() => void onRollbackStage(key)}
                          data-testid={`rollback-stage-${key}`}
                        >
                          Roll back stage
                        </button>
                      </>
                    )}
                  </div>
                  {openStage === key && (
                    <span className="mt-1 block text-xs text-warmgray" data-testid="stage-detail">
                      Recent median absolute error{' '}
                      {s.recent_medae_seconds != null ? formatSeconds(s.recent_medae_seconds) : '—'}{' '}
                      vs baseline{' '}
                      {baselineSeconds != null ? formatSeconds(baselineSeconds) : '—'} (chronological holdout)
                      over {s.recent_outcomes} linked outcome{s.recent_outcomes === 1 ? '' : 's'}.
                      {post
                        ? ` Since promotion (${String(post.promoted_at ?? '')}, model ${String(post.model_id ?? '?')}): median ${formatSeconds(post.median_absolute_error_seconds)} on ${post.outcomes} matched outcome${post.outcomes === 1 ? '' : 's'} — updates as new matched outcomes arrive.`
                        : ' No matched outcomes recorded since the current serving model was activated.'}
                      {prevModel
                        ? ` Previous model ${String(prevModel.model_version ?? '?')} last held ${formatSeconds(prevModel.holdout_medae)} (held before the current promotion).`
                        : ''}
                    </span>
                  )}
                </td>
                <td className="border-b border-[#efece4] py-2 pr-3">
                  {baselineSeconds != null ? formatSeconds(baselineSeconds) : '—'}
                </td>
                <td className="border-b border-[#efece4] py-2 pr-3" data-testid={`current-serving-${key}`}>
                  {post ? (
                    <>
                      {formatSeconds(post.median_absolute_error_seconds)}
                      <span className="block text-[10px] font-normal text-warmgray" data-testid={`post-promotion-n-${key}`}>
                        ({post.outcomes} outcome{post.outcomes === 1 ? '' : 's'})
                      </span>
                    </>
                  ) : (
                    '—'
                  )}
                </td>
                <td className="border-b border-[#efece4] py-2 pr-3" data-testid={`previous-${key}`}>
                  {line?.previous?.holdout_medae != null ? formatSeconds(line.previous.holdout_medae) : '—'}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}
function ModelVersionTable({ models }: { models: ModelRecord[] }) {
  // Newest first — a freshly retrained candidate is immediately visible at
  // the top without digging through the lifecycle timeline.
  const sorted = [...models].sort((x, y) =>
    String(y.created_at ?? '').localeCompare(String(x.created_at ?? '')),
  )
  return (
    <div className="mt-3 max-h-80 overflow-y-auto overflow-x-auto" data-testid="model-versions">
      <table className="w-full text-left text-sm">
        <thead>
          <tr>
            {['Model', 'State', 'Registered'].map((h) => (
              <th
                key={h}
                className="border-b border-linen py-2 pr-3 text-[11px] font-semibold uppercase tracking-wide text-warmgray"
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {sorted.map((m) => {
            const id = String(m.model_id ?? m.model_version ?? '?')
            const state = String(m.lifecycle_state ?? m.status ?? '—').toUpperCase()
            const isCandidate = state === 'CANDIDATE_EVALUATED'
            const isPromoted = state === 'PROMOTED'
            return (
              <tr key={id} data-testid={`model-row-${id}`}>
                <td className="border-b border-[#efece4] py-2 pr-3 font-medium text-ink">
                  {id}
                  {m.stage ? (
                    <span className="block text-[10px] text-warmgray">stage: {m.stage.replace(/_/g, ' ')}</span>
                  ) : null}
                  {isCandidate && (
                    <button
                      type="button"
                      className="mt-1 block text-[11px] font-semibold text-bronze hover:underline"
                      data-testid={`promote-quick-${id}`}
                      onClick={() => PromoteQuickFill(id)}
                    >
                      Use for promote
                    </button>
                  )}
                </td>
                <td className="border-b border-[#efece4] py-2 pr-3">
                  <Badge
                    tone={isPromoted ? 'sage' : isCandidate ? 'warning' : state === 'RETIRED' ? 'neutral' : 'bronze'}
                    testId={isCandidate ? 'current-candidate-flag' : undefined}
                  >
                    {state.toLowerCase().replace(/_/g, ' ')}
                  </Badge>
                </td>
                <td className="border-b border-[#efece4] py-2 pr-3 text-warmgray">
                  {formatDate(m.created_at ?? null)}
                </td>
              </tr>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

/** One-click fill of the "Model to promote" field from a candidate row. */
function PromoteQuickFill(id: string) {
  const input = document.querySelector<HTMLInputElement>('input[aria-label="Model to promote"]')
  if (input) {
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, 'value')?.set
    setter?.call(input, id)
    input.dispatchEvent(new Event('input', { bubbles: true }))
    input.focus()
  }
  window.dispatchEvent(new CustomEvent('drift:quick-fill', { detail: { modelId: id } }))
}

/** Combined history timeline: model promotions / rollbacks / retraining
 * events AND recorded human decisions, newest first, with timestamps. */
function CombinedTimeline({
  events,
  decisions,
}: {
  events: LifecycleEvent[]
  decisions: RetrainingDecision[]
}) {
  const [open, setOpen] = useState(false)
  type Item = { at: string; text: string; reason?: string; kind: 'model' | 'decision' }
  const items: Item[] = [
    ...events.map<Item>((e) => ({
      at: e.at,
      text: e.label,
      reason: e.reason,
      kind: 'model',
    })),
    ...decisions.map<Item>((d) => ({
      at: d.decided_at,
      text: d.decision_type === 'START_RETRAINING' ? 'Retraining started' : 'Continue monitoring',
      reason: d.reason,
      kind: 'decision',
    })),
  ].sort((x, y) => (x.at < y.at ? 1 : -1))
  const shown = open ? items.slice(0, 20) : items.slice(0, 4)
  return (
    <div className="mt-2" data-testid="lifecycle-timeline">
      <ul className="space-y-1.5 text-sm">
        {shown.map((e, i) => (
          <li key={`${e.at}-${i}`} className="flex gap-2">
            <span className="w-24 shrink-0 text-xs text-warmgray">{formatDate(e.at)}</span>
            <span className="text-ink">
              {e.text}
              {e.reason ? <span className="block text-xs text-warmgray">{e.reason}</span> : null}
            </span>
          </li>
        ))}
        {items.length === 0 && (
          <li className="text-warmgray" data-testid="lifecycle-empty">No events recorded yet.</li>
        )}
      </ul>
      {items.length > 4 && (
        <button
          type="button"
          data-testid="lifecycle-toggle"
          onClick={() => setOpen((o) => !o)}
          className="mt-2 text-xs font-semibold text-bronze hover:underline"
        >
          {open ? 'Show less' : `Show all ${items.length} events`}
        </button>
      )}
    </div>
  )
}

/** Human decision controls (§20-§24). Promotion is compatibility-gated by the
 * backend; rollback restores the previous promoted model. Nothing is automatic. */
function GovernanceActions({ onDone }: { onDone: () => void }) {
  const [busy, setBusy] = useState<'promote' | 'rollback' | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const [candidate, setCandidate] = useState('')
  const [reason, setReason] = useState('')

  // "Use for promote" in the Model Versions table prefills the candidate id here.
  useEffect(() => {
    const onQuickFill = (e: Event) => {
      const detail = (e as CustomEvent<{ modelId?: string }>).detail
      if (detail?.modelId) setCandidate(detail.modelId)
    }
    window.addEventListener('drift:quick-fill', onQuickFill)
    return () => window.removeEventListener('drift:quick-fill', onQuickFill)
  }, [])

  async function promote() {
    if (!candidate.trim()) return
    setBusy('promote')
    setNotice(null)
    try {
      await postModelPromote({
        model_id: candidate.trim(),
        reason: reason.trim() || 'Human-approved promotion from Drift & Retraining',
      })
      setNotice(`Promoted ${candidate.trim()}.`)
      onDone()
    } catch (err: unknown) {
      setNotice(err instanceof ApiError ? err.message : 'Promotion failed.')
    } finally {
      setBusy(null)
    }
  }

  async function rollback() {
    setBusy('rollback')
    setNotice(null)
    try {
      await postModelRollback({
        reason: reason.trim() || 'Human-approved rollback from Drift & Retraining',
      })
      setNotice('Rollback recorded. The previous promoted model is serving again.')
      onDone()
    } catch (err: unknown) {
      setNotice(err instanceof ApiError ? err.message : 'Rollback failed.')
    } finally {
      setBusy(null)
    }
  }

  return (
    <div data-testid="governance-actions">
      <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-warmgray">
        Human decision
      </p>
      <div className="mt-2 grid grid-cols-1 gap-2 sm:grid-cols-2">
        <input
          className="field-input"
          placeholder="Model id to promote"
          aria-label="Model to promote"
          value={candidate}
          onChange={(e) => setCandidate(e.target.value)}
        />
        <input
          className="field-input"
          placeholder="Reason"
          aria-label="Reason"
          value={reason}
          onChange={(e) => setReason(e.target.value)}
        />
      </div>
      <div className="mt-2 flex flex-wrap gap-2">
        <button
          type="button"
          className="btn-primary"
          disabled={busy !== null || !candidate.trim()}
          onClick={() => void promote()}
          data-testid="promote-button"
        >
          {busy === 'promote' ? 'Promoting…' : 'Promote candidate'}
        </button>
        <button
          type="button"
          className="btn-secondary"
          disabled={busy !== null}
          onClick={() => void rollback()}
          data-testid="rollback-button"
        >
          {busy === 'rollback' ? 'Rolling back…' : 'Roll back'}
        </button>
      </div>
      {notice && (
        <p className="mt-2 text-xs text-bronze" role="status" data-testid="governance-notice">
          {notice}
        </p>
      )}
    </div>
  )
}

/** Human decision controls (§9-§16). Decisions are persisted; "Retrain now"
 * only trains + evaluates a CANDIDATE with the same approved model families —
 * the serving model is untouched until a separate human promotion. */
function RetrainingActions({
  verdict,
  onDone,
  evaluation,
  onCandidate,
  degradedStages,
}: {
  verdict: string
  onDone: () => void
  evaluation: CandidateEvaluation | null
  onCandidate: (r: TrainCandidateResponse) => void
  degradedStages: string[]
}) {
  const [busy, setBusy] = useState<'monitor' | 'retrain' | 'stage' | null>(null)
  const [reason, setReason] = useState('')
  const [notice, setNotice] = useState<string | null>(null)
  const [stageNotice, setStageNotice] = useState<string | null>(null)
  const [lastEval, setLastEval] = useState<TrainCandidateResponse | null>(null)

  async function decide(decision_type: 'CONTINUE_MONITORING' | 'START_RETRAINING') {
    setBusy(decision_type === 'CONTINUE_MONITORING' ? 'monitor' : 'retrain')
    setNotice(null)
    try {
      await postRetrainingDecision({
        decision_type,
        reason: reason.trim() || `Human decision from Drift & Retraining (assessment: ${verdict})`,
        decided_by: 'local-operator',
      })
      if (decision_type === 'CONTINUE_MONITORING') {
        setNotice('Decision recorded: continue monitoring. The current model stays in production.')
        onDone()
      } else {
        setNotice('Decision recorded: retraining started. Training + evaluating a candidate with the same approved model families — this can take a few minutes; the current model keeps serving.')
        const result = await postTrainCandidate({
          approved_by: 'local-operator',
          reason: reason.trim() || 'Candidate training after human retrain decision',
        })
        // Feed the left-side stage comparison table (survives the
        // refresh triggered by onDone below).
        onCandidate(result)
        setLastEval(result)
        onDone()
      }
    } catch (err: unknown) {
      setNotice(err instanceof ApiError ? err.message : 'The action could not be completed.')
    } finally {
      setBusy(null)
    }
  }

  // Individual-stage retraining (same backend path with ``stages``):
  // retrains and evaluates ONLY the selected stage; the other four
  // serving stage models remain untouched until per-stage promotion.
  async function retrainStage(stage: string) {
    setBusy('stage')
    setNotice(null)
    try {
      const result = await postTrainCandidate({
        approved_by: 'local-operator',
        reason:
          reason.trim() ||
          `Individual-stage retraining for ${stageLabel(stage)} (human-approved from Drift & Retraining)`,
        stages: [stage],
      })
      onCandidate(result)
      setLastEval(result)
      setStageNotice(`Candidate for ${stageLabel(stage)} trained and evaluated — see "Candidate" in Model versions. Serving is untouched.`)
      onDone()
    } catch (err: unknown) {
      setNotice(err instanceof ApiError ? err.message : 'Stage retraining failed.')
    } finally {
      setBusy(null)
    }
  }

  const ev = lastEval?.evaluation ?? evaluation
  return (
    <div className="mt-4 border-t border-linen pt-3" data-testid="retraining-actions">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-warmgray">
        Your decision (recorded)
      </p>
      <input
        className="field-input"
        placeholder="Reason"
        aria-label="Decision reason"
        value={reason}
        onChange={(e) => setReason(e.target.value)}
      />
      <div className="mt-2 flex flex-wrap gap-2">
        <button
          type="button"
          className="btn-secondary"
          disabled={busy !== null}
          onClick={() => void decide('CONTINUE_MONITORING')}
          data-testid="continue-monitoring-button"
        >
          {busy === 'monitor' ? 'Recording…' : 'Continue monitoring'}
        </button>
        <button
          type="button"
          className="btn-primary"
          disabled={busy !== null}
          onClick={() => void decide('START_RETRAINING')}
          data-testid="retrain-now-button"
        >
          {busy === 'retrain' ? 'Training candidate… (minutes)' : 'Retrain all 5 stage models'}
        </button>

      {degradedStages.length > 0 && (
        <div className="mt-2 flex flex-wrap items-center gap-2" data-testid="stage-retrain-actions">
          <span className="text-[11px] font-semibold uppercase tracking-[0.12em] text-warmgray">
            Retrain individual stage:
          </span>
          {degradedStages.map((s) => (
            <button
              key={s}
              type="button"
              className="btn-secondary"
              disabled={busy !== null}
              onClick={() => void retrainStage(s)}
              data-testid={`retrain-stage-now-${s}`}
            >
              {busy === 'stage' ? 'Training…' : `Retrain ${stageLabel(s)}`}
            </button>
          ))}
        </div>
      )}
      {stageNotice && (
        <p className="mt-2 text-xs text-bronze" role="status" data-testid="stage-retrain-notice">
          {stageNotice}
        </p>
      )}
      </div>

      {notice && (
        <p className="mt-2 text-xs text-bronze" role="status" data-testid="retraining-notice">
          {notice}
        </p>
      )}
      {lastEval && (
        <CandidateEvaluationView
          evaluation={unwrapEval(lastEval.evaluation, String(unwrapRegister(lastEval.registered)?.model_id ?? ''))}
          registeredId={String(unwrapRegister(lastEval.registered)?.model_id ?? '')}
          servingUntouched={lastEval.serving_untouched}
        />
      )}
      {ev && !lastEval && (
        <CandidateEvaluationView evaluation={unwrapEval(ev, '?')} registeredId="" servingUntouched />
      )}
    </div>
  )
}

// 2026-10-05 (user spec): policy-explainer sentences are hidden from the UI —
// statuses and numbers stay, the prose goes.
const EVAL_VERDICTS: Record<string, { title: string; tone: BadgeTone }> = {
  ACCEPTABLE_FOR_REVIEW: { title: 'Candidate ready for human review', tone: 'sage' },
  NOT_SUFFICIENTLY_BETTER: { title: 'Candidate not sufficiently better', tone: 'warning' },
  INSUFFICIENT_EVIDENCE: { title: 'Candidate cannot be compared', tone: 'neutral' },
}

function CandidateEvaluationView({
  evaluation,
  registeredId,
  servingUntouched,
}: {
  evaluation: CandidateEvaluation
  registeredId: string
  servingUntouched: boolean
}) {
  const v = EVAL_VERDICTS[evaluation.verdict] ?? { title: evaluation.verdict, tone: 'neutral' as BadgeTone }
  return (
    <div className="mt-3 rounded-md border border-linen bg-parchment p-3" data-testid="candidate-evaluation">
      <div className="flex flex-wrap items-center gap-2">
        <Badge tone={v.tone} testId="candidate-verdict">{v.title}</Badge>
        {registeredId && <Badge tone="neutral">Candidate {registeredId}</Badge>}
        {servingUntouched && <Badge tone="neutral">Production model unchanged</Badge>}
      </div>
      <table className="mt-2 w-full text-left text-sm">
        <thead>
          <tr>
            {['Stage', 'Candidate median error', 'Current baseline', 'Candidate outcomes'].map((hd) => (
              <th
                key={hd}
                className="border-b border-linen py-1.5 pr-3 text-[11px] font-semibold uppercase tracking-wide text-warmgray"
              >
                {hd}
              </th>
            ))}
          </tr>
        </thead>
        <tbody data-testid="candidate-stage-table">
          {evaluation.per_stage.map((r) => (
            <tr key={r.stage}>
              <td className="border-b border-[#efece4] py-1.5 pr-3 font-medium text-ink">{r.stage.replace(/_/g, ' ')}</td>
              <td className="border-b border-[#efece4] py-1.5 pr-3">{formatSeconds(r.candidate_median_ae)}</td>
              <td className="border-b border-[#efece4] py-1.5 pr-3">{formatSeconds(r.baseline_median_ae)}</td>
              <td className="border-b border-[#efece4] py-1.5 pr-3 text-warmgray">{r.candidate_n ?? '—'}</td>
            </tr>
          ))}
          <tr>
            <td className="border-b border-[#efece4] py-1.5 pr-3 font-semibold text-ink">Total (sum of stages)</td>
            <td className="border-b border-[#efece4] py-1.5 pr-3 font-semibold">{formatSeconds(evaluation.total.candidate_median_ae)}</td>
            <td className="border-b border-[#efece4] py-1.5 pr-3 font-semibold">{formatSeconds(evaluation.total.baseline_median_ae)}</td>
            <td className="border-b border-[#efece4] py-1.5 pr-3 text-warmgray">{evaluation.total.candidate_n ?? '—'}</td>
          </tr>
        </tbody>
      </table>
      <p className="mt-2 text-xs text-warmgray" data-testid="candidate-rationale">{evaluation.rationale}</p>
      {registeredId && (
        <p className="mt-1 text-xs text-bronze">
          To make this candidate production: copy its id into "Model to promote" in the section below
          and promote it explicitly. Nothing promotes automatically.
        </p>
      )}
    </div>
  )
}

/** Single-stage train responses carry ONE record; multi-stage carry lists. */
function unwrapRegister(r: TrainCandidateResponse['registered']): ModelRecord | null {
  if (Array.isArray(r)) return r[r.length - 1] ?? null
  return r ?? null
}

function unwrapEval(e: TrainCandidateResponse['evaluation'], candidateVersion: string): CandidateEvaluation {
  if (Array.isArray(e)) {
    return {
      ...(e[0] ?? ({} as CandidateEvaluation)),
      candidate_version: e[0]?.candidate_version ?? candidateVersion,
    }
  }
  return { ...e, candidate_version: e.candidate_version ?? candidateVersion }
}

function formatDate(v: string | null | undefined): string {
  if (!v) return '—'
  const d = new Date(v)
  return Number.isNaN(d.getTime()) ? String(v) : d.toLocaleDateString()
}
