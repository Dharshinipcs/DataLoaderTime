import { useState } from 'react'
import {
  ApiError,
  type PredictionResponse,
  type PredictionRequestBody,
  type StageIntelligenceEntry,
  type ShapExplanation,
  fetchPredictionShap,
  predict,
} from '../api'
import ActualOutcomeForm, {
  outcomeStateFromPrediction,
} from '../components/ActualOutcomeForm'
import { formatNumber, formatSeconds } from '../lib/format'
import { Card, PageHeader } from '../components/ui'

type FormState = {
  loader_name: string
  sprint: string
  dataset: string
  connection: string
  // 2026-10-02: per-component pre-execution rows matching the real detail
  // source — each row is ONE component of this execution: its display name
  // (the source's single preval_trans_display_name column), its calling type
  // (PREVALIDATION or TRANSFORMATION), its component type (crossreference,
  // dataobject, dataextract, dataset, query or table) and its execution
  // order. Prevalidation alone, transformation alone, both, or none (rows
  // removed) are all valid.
  components: ComponentRow[]
}

export type ComponentRow = {
  display_name: string
  calling_type: string
  type: string
  execution_order: string
}

/** The six component types of the real pre-execution source (2026-10-03). */
const COMPONENT_TYPES = [
  'crossreference',
  'dataobject',
  'dataextract',
  'dataset',
  'query',
  'table',
] as const

const EMPTY_COMPONENT_ROW: ComponentRow = {
  display_name: '',
  calling_type: 'PREVALIDATION',
  type: 'crossreference',
  execution_order: '',
}

const EMPTY_FORM: FormState = {
  loader_name: '',
  sprint: '',
  dataset: '',
  connection: '',
  components: [],
}

const STATE_TITLES: Record<PredictionResponse['state'], string> = {
  PREDICTED: 'Prediction Result',
  INSUFFICIENT_HISTORY: 'Insufficient history to predict',
  UNRESOLVED_IDENTITY: 'Identity could not be resolved',
  AMBIGUOUS_IDENTITY: 'Ambiguous identity — system refuses to guess',
  INVALID_INPUT: 'Input rejected',
}

const STATE_EXPLANATIONS: Record<PredictionResponse['state'], string> = {
  PREDICTED:
    'The applicable operational stage models were run and their predictions, plus the approval estimates when available, are summed into the authoritative total execution-time estimate (legitimate pre-execution information and eligible loader history only).',
  INSUFFICIENT_HISTORY:
    'This loader does not have enough eligible historical executions. No authoritative ETA was produced — the prediction is intentionally withheld (no fallback estimate is invented).',
  UNRESOLVED_IDENTITY:
    'The supplied pre-execution identity (Dataset + Sprint + Loader) could not be resolved against the registered loader definitions. No identity was selected on its behalf.',
  AMBIGUOUS_IDENTITY:
    'This identity matches more than one distinct definition connection. The system refuses to silently choose between them.',
  INVALID_INPUT:
    'The request violated the pre-execution-only contract or failed validation. Fix the issue and predict again.',
}

type RequestPhase =
  | { kind: 'idle' }
  | { kind: 'loading' }
  | { kind: 'done'; result: PredictionResponse }
  | { kind: 'error'; message: string }

export default function PredictionStudio() {
  const [form, setForm] = useState<FormState>(EMPTY_FORM)
  const [phase, setPhase] = useState<RequestPhase>({ kind: 'idle' })
  const [fieldErrors, setFieldErrors] = useState<Record<string, string>>({})

  const loading = phase.kind === 'loading'

  function update<K extends keyof FormState>(key: K, value: FormState[K]) {
    setForm((f) => ({ ...f, [key]: value }))
  }

  function updateComponent(i: number, key: keyof ComponentRow, value: string) {
    setForm((f) => ({
      ...f,
      components: f.components.map((c, idx) => (idx === i ? { ...c, [key]: value } : c)),
    }))
  }

  function addComponent() {
    setForm((f) => ({ ...f, components: [...f.components, { ...EMPTY_COMPONENT_ROW }] }))
  }

  function removeComponent(i: number) {
    setForm((f) => ({ ...f, components: f.components.filter((_, idx) => idx !== i) }))
  }

  function validate(): boolean {
    const errors: Record<string, string> = {}
    if (!form.loader_name.trim()) errors.loader_name = 'Loader Name is required.'
    if (!form.sprint.trim()) errors.sprint = 'Sprint is required.'
    if (!form.dataset.trim()) errors.dataset = 'Dataset is required.'
    if (!form.connection.trim()) errors.connection = 'Connection is required.'
    setFieldErrors(errors)
    return Object.keys(errors).length === 0
  }

  async function onSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (loading) return // duplicate submission guard
    if (!validate()) {
      // A new prediction was attempted: drop the previous result so the UI
      // never shows a stale ETA alongside the new request's field errors.
      setPhase({ kind: 'idle' })
      return
    }

    setPhase({ kind: 'loading' }) // clears the previous result for this request
    const body: PredictionRequestBody = {
      loader_name: form.loader_name.trim(),
      sprint: form.sprint.trim(),
      dataset: form.dataset.trim(),
      connection: form.connection.trim(),
    }
    // 2026-10-02: per-component rows — every filled row (a display name is
    // the identity of a component) is sent exactly as entered; empty rows are
    // ignored. No components -> the definition-derived configuration stands.
    const comps = form.components
      .map((c) => ({
        display_name: c.display_name.trim(),
        calling_type: c.calling_type === 'TRANSFORMATION' ? 'TRANSFORMATION' : 'PREVALIDATION',
        type: c.type.trim().toLowerCase(),
        execution_order: c.execution_order.trim(),
      }))
      .filter((c) => c.display_name.length > 0 || c.execution_order.length > 0)
      .map((c) => ({
        display_name: c.display_name,
        calling_type: c.calling_type,
        type: COMPONENT_TYPES.includes(c.type as (typeof COMPONENT_TYPES)[number]) ? c.type : null,
        execution_order: c.execution_order ? Number(c.execution_order) : null,
      }))
      .filter((c) => c.display_name.length > 0)
    for (const c of comps) {
      if (c.execution_order !== null && (!Number.isFinite(c.execution_order) || c.execution_order < 0)) {
        setPhase({ kind: 'error', message: `Execution order for component "${c.display_name}" must be a non-negative number.` })
        return
      }
    }
    if (comps.length > 0) {
      body.preval_trans_components = comps
    }

    try {
      const result = await predict(body)
      setPhase({ kind: 'done', result })
    } catch (err) {
      setPhase({
        kind: 'error',
        message:
          err instanceof ApiError
            ? err.message
            : 'Unexpected error while contacting the prediction backend.',
      })
    }
  }

  const inputClass = 'field-input !py-1.5'

  return (
    <div>
      <PageHeader
        eyebrow=""
        title="Pre-Execution Features"
      />

      <div className="grid gap-6 lg:grid-cols-5">
        {/* 1. Execution Context form — wider than half so Calling Type / Type
            etc. render without cramping (2026-10-08 user spec). */}
        <section className="lg:col-span-3">
          <Card testId="studio-form" className="p-4">
            <form onSubmit={onSubmit} noValidate>
              <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-warmgray">
                Execution Context
              </h2>
              <div className="mt-4 grid grid-cols-1 gap-x-4 gap-y-3 sm:grid-cols-2">
                <div>
                  <label htmlFor="loader_name" className="field-label">
                    Loader Name <span className="text-ember">*</span>
                  </label>
                  <input
                    id="loader_name"
                    className={inputClass}
                    value={form.loader_name}
                    onChange={(e) => update('loader_name', e.target.value)}
                    disabled={loading}
                  />
                  {fieldErrors.loader_name && (
                    <p className="mt-1 text-xs text-ember">{fieldErrors.loader_name}</p>
                  )}
                </div>
                <div>
                  <label htmlFor="sprint" className="field-label">
                    Sprint <span className="text-ember">*</span>
                  </label>
                  <input
                    id="sprint"
                    className={inputClass}
                    value={form.sprint}
                    onChange={(e) => update('sprint', e.target.value)}
                    disabled={loading}
                  />
                  {fieldErrors.sprint && (
                    <p className="mt-1 text-xs text-ember">{fieldErrors.sprint}</p>
                  )}
                </div>
                <div>
                  <label htmlFor="dataset" className="field-label">
                    Dataset Name <span className="text-ember">*</span>
                  </label>
                  <input
                    id="dataset"
                    className={inputClass}
                    value={form.dataset}
                    onChange={(e) => update('dataset', e.target.value)}
                    disabled={loading}
                  />
                  {fieldErrors.dataset && (
                    <p className="mt-1 text-xs text-ember">{fieldErrors.dataset}</p>
                  )}
                </div>
                <div>
                  <label htmlFor="connection" className="field-label">
                    Connection <span className="text-ember">*</span>
                  </label>
                  <input
                    id="connection"
                    className={inputClass}
                    value={form.connection}
                    onChange={(e) => update('connection', e.target.value)}
                    disabled={loading}
                  />
                  {fieldErrors.connection && (
                    <p className="mt-1 text-xs text-ember">{fieldErrors.connection}</p>
                  )}
                </div>
              </div>

              {/* 2026-10-02: per-component pre-execution configuration — the
                  real source stores one row per component (single
                  preval_trans_display_name column + type/calling_type/
                  execution_order). The user enters exactly those rows:
                  prevalidation alone, transformation alone, both, or none. */}
              <fieldset className="mt-4 border-t border-linen pt-3">
                <legend className="text-xs font-semibold uppercase tracking-[0.12em] text-warmgray">
                  Prevalidation / Transformation Components
                </legend>
                {/* 2026-10-03: live component counts — computed directly from
                    the rows the user entered, BEFORE Predict, so the user can
                    verify the system sees exactly what they configured. */}
                {form.components.some((c) => c.display_name.trim() || c.execution_order.trim()) ? (
                  <p
                    data-testid="component-counts"
                    className="mt-2 rounded-md border border-linen bg-white px-2.5 py-1.5 text-xs font-medium text-ink"
                  >
                    Components detected:{' '}
                    <span className="text-bronze">{form.components.filter((c) => c.display_name.trim() && c.calling_type === 'PREVALIDATION').length} prevalidation</span>
                    {' · '}
                    <span className="text-bronze">{form.components.filter((c) => c.display_name.trim() && c.calling_type === 'TRANSFORMATION').length} transformation</span>
                    {' — both counts are passed to the models with the prediction.'}
                  </p>
                ) : null}
                <div className="mt-2 space-y-2">
                  {form.components.map((row, i) => (
                    <div
                      key={i}
                      data-testid={`component-row-${i}`}
                      className="grid grid-cols-12 items-end gap-2 rounded-md border border-linen bg-[#fbf8f1] p-2"
                    >
                      <div className="col-span-12 sm:col-span-4">
                        <label htmlFor={`component_display_name_${i}`} className="field-label">
                          Display Name
                        </label>
                        <input
                          id={`component_display_name_${i}`}
                          type="text"
                          className={inputClass}
                          placeholder="e.g. Preval_1_employeeName or tf"
                          value={row.display_name}
                          data-testid={`component-display-name-${i}`}
                          onChange={(e) => updateComponent(i, 'display_name', e.target.value)}
                          disabled={loading}
                        />
                      </div>
                      <div className="col-span-5 sm:col-span-2">
                        <label htmlFor={`component_calling_type_${i}`} className="field-label">
                          Calling Type
                        </label>
                        <select
                          id={`component_calling_type_${i}`}
                          className={inputClass}
                          value={row.calling_type}
                          data-testid={`component-calling-type-${i}`}
                          onChange={(e) => updateComponent(i, 'calling_type', e.target.value)}
                          disabled={loading}
                        >
                          <option value="PREVALIDATION">PREVALIDATION</option>
                          <option value="TRANSFORMATION">TRANSFORMATION</option>
                        </select>
                      </div>
                      <div className="col-span-5 sm:col-span-3">
                        <label htmlFor={`component_type_${i}`} className="field-label">
                          Type
                        </label>
                        <select
                          id={`component_type_${i}`}
                          className={inputClass}
                          value={row.type}
                          data-testid={`component-type-${i}`}
                          onChange={(e) => updateComponent(i, 'type', e.target.value)}
                          disabled={loading}
                        >
                          {COMPONENT_TYPES.map((t) => (
                            <option key={t} value={t}>
                              {t}
                            </option>
                          ))}
                        </select>
                      </div>
                      <div className="col-span-4 sm:col-span-2">
                        <label htmlFor={`component_execution_order_${i}`} className="field-label">
                          Order
                        </label>
                        <input
                          id={`component_execution_order_${i}`}
                          type="number" min="0" step="1"
                          className={inputClass}
                          value={row.execution_order}
                          data-testid={`component-execution-order-${i}`}
                          onChange={(e) => updateComponent(i, 'execution_order', e.target.value)}
                          disabled={loading}
                        />
                      </div>
                      <div className="col-span-3 sm:col-span-1">
                        <button
                          type="button"
                          aria-label={`Remove component row ${i + 1}`}
                          title="Remove this component row"
                          className="w-full rounded border border-linen bg-white py-1.5 text-xs text-warmgray hover:text-ember"
                          data-testid={`component-remove-${i}`}
                          onClick={() => removeComponent(i)}
                          disabled={loading}
                        >
                          ✕
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
                <button
                  type="button"
                  className="btn-secondary mt-2 !py-1 text-xs"
                  data-testid="component-add"
                  onClick={addComponent}
                  disabled={loading}
                >
                  + Add Component
                </button>
              </fieldset>

              {/* 2. Predict action — placed directly under the inputs so the
                  whole workflow fits one desktop viewport. */}
              <button type="submit" disabled={loading} className="btn-primary mt-4 w-full">
                {loading ? 'Predicting…' : 'Predict Execution Time'}
              </button>

              {/* AD-24/AD-31 + auto-derivation note, one compact line. The
                  workflow reference belongs to outcome recording only; ML
                  features are derived internally and never entered here. */}
            </form>
          </Card>
        </section>

        {/* 3. Result area — remaining width (form takes 3 of 5 columns). */}
        <section className="lg:col-span-2">
          {phase.kind === 'idle' && (
            <div className="flex h-full min-h-64 items-center justify-center rounded-md border border-dashed border-linen bg-[#fbf8f1] p-8 text-center">
              <p className="text-sm text-warmgray">
                <strong>Loader Execution Time</strong>.
              </p>
            </div>
          )}

          {phase.kind === 'loading' && (
            <div
              role="status"
              aria-live="polite"
              className="flex h-full min-h-64 items-center justify-center rounded-md border border-linen bg-parchment p-8"
            >
              <div className="text-center">
                <div className="mx-auto h-8 w-8 animate-spin rounded-full border-2 border-bronze border-t-transparent" />
              </div>
            </div>
          )}

          {phase.kind === 'error' && (
            <div
              role="alert"
              className="rounded-md border border-[#e3c6c1] bg-[#f6e7e4] p-5"
            >
              <h3 className="text-sm font-semibold text-ember">Request failed</h3>
              <p className="mt-1 text-sm text-ember">{phase.message}</p>
              <button
                type="button"
                onClick={() => onSubmit({ preventDefault() {} } as React.FormEvent)}
                className="btn-secondary mt-3"
              >
                Retry prediction
              </button>
            </div>
          )}

          {phase.kind === 'done' && <ResultCard result={phase.result} />}

          {/* AD-32 §25: post-execution outcome entry for the just-predicted
              execution (prefilled from the prediction; manual ingestion is the
              temporary source until DataZap provides the same fields). The form
              stays one click away so the result remains the primary object. */}
          {/* 2026-10-02: post-execution entry is available after BOTH the
              prediction AND the withheld-history states — recording actuals
              is how a new/below-threshold loader accumulates history until it
              reaches the trust threshold and predictions begin. */}
          {phase.kind === 'done' &&
            (phase.result.state === 'PREDICTED' || phase.result.state === 'INSUFFICIENT_HISTORY') && (
              <ActualOutcomeSection result={phase.result} />
            )}
        </section>
      </div>
    </div>
  )
}

function ResultCard({ result }: { result: PredictionResponse }) {
  const state = result.state
  const predicted = state === 'PREDICTED'
  return (
    <div
      role="region"
      aria-label="prediction result"
      data-testid="prediction-result"
      className={
        'rounded-md border p-5 ' +
        (predicted
          ? 'border-linen bg-parchment'
          : state === 'INVALID_INPUT'
            ? 'border-[#e3c6c1] bg-[#f6e7e4]'
            : 'border-linen bg-[#fbf8f1]')
      }
    >
      <div className="flex flex-wrap items-center gap-2">
        <h2 className="font-serif text-lg font-bold text-ink">{STATE_TITLES[state]}</h2>
      </div>

      {/* 2026-10-02: INSUFFICIENT_HISTORY gets the short user-specified
          message only — no long explanation, no "Reason:" block, no counts
          line. New loader: it is a new loader with no prior executions.
          Below-threshold: only N priors vs the trust threshold. */}
      {state === 'INSUFFICIENT_HISTORY' ? (
        <p data-testid="insufficient-history-note" className="mt-2 text-sm leading-relaxed text-ink">
          {insufficientHistoryMessage(result)}
        </p>
      ) : (
        <>
          {state !== 'PREDICTED' && (
            <p className="mt-1 text-sm leading-relaxed text-warmgray">{STATE_EXPLANATIONS[state]}</p>
          )}
          {result.reason && (
            <p className="mt-3 text-sm text-warmgray">
              <span className="font-semibold text-ink">Reason: </span>
              {result.reason}
            </p>
          )}
        </>
      )}

      {predicted && (
        <>
          {/* B. Total ETA — the strongest visual element on the page.
              2026-10-02: the total is displayed in SECONDS (exact), with the
              minutes equivalent on the line below. */}
          <div className="mt-4 rounded-md border border-linen bg-[#fbf8f1] p-4">
            <div className="flex flex-wrap items-baseline justify-between gap-2">
              <p className="text-xs font-semibold uppercase tracking-[0.12em] text-warmgray">
                Total Execution Time
              </p>
            </div>
            <p
              data-testid="predicted-eta"
              className="mt-1 font-serif text-4xl font-bold tracking-tight text-ink"
            >
              {formatSecondsExact(result.predicted_seconds)}
            </p>
            <p data-testid="predicted-human" className="mt-0.5 text-sm font-semibold text-bronze">
              ≈ {formatNumber((result.predicted_seconds ?? 0) / 60, 2)} min
            </p>
          </div>

          {/* Five operational stage predictions — compact table. */}
          {result.stage_intelligence && (
            <StageIntelligenceSection si={result.stage_intelligence} total={result.predicted_seconds} />
          )}

          {/* Prediction Insights — ⓘ icon (§5): range, confidence, and the
              real per-stage contribution explanation only. Risk and technical
              metadata are NOT in this panel. */}
          <InsightsPanel result={result} />
        </>
      )}      {state === 'INSUFFICIENT_HISTORY' && result.history && (
        <p data-testid="insufficient-history-counts" className="mt-2 text-xs text-warmgray">
          Eligible historical executions found: <strong>{String(result.history.prior_loader_executions ?? '—')}</strong>
          {' '}(required minimum: {String(result.history.eligibility_policy?.min_prior_loader_executions ?? '—')}).
        </p>
      )}

      {/* 2026-10-05 (user spec): per-prediction risk pop-ups/toasts are fully
          removed. Risk stays on the dedicated risk surfaces; the Alert Center
          carries only the three fixed scenarios. */}

      {/* Explanations are user-facing (why the prediction moved). Only
          internal feature-engineering notices are suppressed; the backend
          still logs everything for auditability. */}
      {result.warnings.filter((w) => !/feature engineering|stage_hist|connection-aware|historical features/i.test(w)).length > 0 && (
        <ul className="mt-3 list-disc space-y-1 pl-5 text-xs text-ochre">
          {result.warnings
            .filter((w) => !/feature engineering|stage_hist|connection-aware|historical features/i.test(w))
            .map((w) => (
              <li key={w}>{w}</li>
            ))}
        </ul>
      )}
    </div>
  )
}

/** "Why this ETA?" (2026-10-06 user spec, refined):
 * PREFER the backend plain-explain-v2 explanation — the essence of why this
 * ETA (biggest driver vs the loader's own typical time, the calibrated range
 * from similar past runs, and steps that do not apply) — assembled from the
 * same explainability layers that run for every prediction. The exact-sum
 * equation of the applicable stage predictions remains the fallback when the
 * backend layer is unavailable (e.g. INSUFFICIENT_HISTORY), so the line is
 * never empty. Nothing is hardcoded. */
function whyEtaSummary(result: PredictionResponse): string {
  if (result.plain_explanation && result.plain_explanation.trim()) {
    return result.plain_explanation
  }
  const si = result.stage_intelligence
  if (!si || !si.stages.length) {
    return 'The total is the exact sum of the applicable stage predictions for this run.'
  }
  const operational = si.stages.filter((s) => !isApprovalStage(s.stage))
  const used = operational.filter((s) => s.available && s.estimate_seconds != null)
  const notApplicable = operational.filter((s) => !s.available)
  const approvals = si.stages.filter(
    (s) =>
      isApprovalStage(s.stage) &&
      s.available &&
      s.estimate_seconds != null &&
      s.included_in_total !== false,
  )
  const lines: string[] = []
  if (used.length > 0) {
    const sum = used.reduce((acc, s) => acc + (s.estimate_seconds ?? 0), 0)
    const total = result.predicted_seconds
    lines.push(
      `The total is the exact sum of the ${used.length} stage prediction${used.length === 1 ? '' : 's'} that apply to this run: ` +
        used.map((s) => `${stageDisplay(s.stage)} ${formatSeconds(s.estimate_seconds)}`).join(' + ') +
        ` = ${formatSeconds(total != null && Math.abs(total - sum) < 0.01 ? total : sum)}.`,
    )
  }
  if (notApplicable.length > 0) {
    lines.push(
      `${notApplicable.map((s) => stageDisplay(s.stage)).join(' and ')} do${notApplicable.length === 1 ? 'es' : ''} not apply to this run, so they add nothing — that is why this ETA can look shorter than a full run.`,
    )
  }
  if (approvals.length > 0) {
    lines.push(
      `Approval timings (${approvals.map((s) => approvalLabel(s.stage)).join(', ')}) are historical medians from recorded runs — they are not part of the ETA.`,
    )
  }
  return lines.join(' ')
}

/** F/G. Applicable-Stage Execution View (user final spec, 2026-09-30): only the
 * stages THIS execution will actually run are shown — staging, pre-load and
 * data loading always, prevalidation/transformation only when the user
 * supplied the matching pre-execution entries — plus the contextual
 * statistical approval estimates. The total is the exact sum of the
 * applicable stage predictions plus the approval estimates when available.
 * Method and evidence come from the backend; no stage is ever called an actual. */
function StageIntelligenceSection({
  si,
  total,
}: {
  si: PredictionResponse['stage_intelligence']
  total: PredictionResponse['predicted_seconds']
}) {
  if (!si) return null
  const operational = si.stages.filter((s) => !isApprovalStage(s.stage))
  const predicted = operational.filter((s) => s.available)
  const unavailable = operational.filter((s) => !s.available)
  // 2026-10-03 (user final spec): approval timings appear ONLY when they come
  // from THIS loader's own recorded approval history (backend flags each
  // entry included_in_total). A loader with no own approval history shows NO
  // approval section at all — nothing is displayed from broader-context or
  // unavailable estimates.
  const approvals = si.stages.filter(
    (s) => isApprovalStage(s.stage) && s.available && s.estimate_seconds != null && s.included_in_total !== false,
  )

  return (
    <div>

      {si.available && si.stages.length > 0 ? (
        <>
          <div className="mt-3 space-y-2" data-testid="stage-rows">
            {predicted.map((s) => (
              <StageRow key={s.stage} entry={s} total={total ?? null} />
            ))}
          </div>

          {unavailable.length > 0 && (
            <p
              data-testid="stage-omitted-note"
              className="mt-2 rounded-md border border-linen bg-[#fbf8f1] px-3 py-2 text-[11px] leading-relaxed text-warmgray"
            >
              {unavailable.length} applicable stage(s) could not be predicted for this context —
              {' '}
              {unavailable
                .map(
                  (s) =>
                    `${stageDisplay(s.stage)} (${s.availability_reason ?? 'insufficient stage evidence'})`,
                )
                .join('; ')}
              . No estimate is fabricated for them.
            </p>
          )}
        </>
      ) : (
        <p className="mt-3 rounded-md border border-linen bg-[#fbf8f1] p-3 text-sm text-warmgray">
          Stage view unavailable for this context
          {si.reason ? `: ${si.reason}` : '.'}
        </p>
      )}

      {/* G. Contextual approval estimates — clearly separated. 2026-10-02:
          ONLY estimates backed by the loader's OWN approval history (LOADER
          tier) are added to the total; broader-context estimates (e.g. global
          medians) are shown as analytics only, never summed. */}
      {approvals.length > 0 && (
        <div className="mt-3 rounded-md border border-dashed border-linen bg-[#fbf8f1] p-3" data-testid="approval-analytics">
          <div className="flex flex-wrap items-baseline justify-between gap-2">
            <h4 className="text-xs font-semibold uppercase tracking-[0.12em] text-warmgray">
              Approval Timings
            </h4>
          </div>
          <div className="mt-2 space-y-1.5">
            {approvals.map((s) => (
              <div key={s.stage} className="flex items-baseline justify-between gap-3 text-sm">
                <span className="font-medium text-ink">{approvalLabel(s.stage)}</span>
                <span className="text-warmgray">
                  {formatSeconds(s.estimate_seconds)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}

function isApprovalStage(stage: string): boolean {
  return stage === 'datamart_approval' || stage === 'dataloading_approval'
}

function approvalLabel(stage: string): string {
  return stage === 'datamart_approval' ? 'Datamart Approval' : 'Dataloading Approval'
}

function StageRow({ entry, total }: { entry: StageIntelligenceEntry; total: number | null }) {
  // Compact product row (final UI spec §3/§4): stage name + predicted seconds
  // + proportional bar. Model/method/evidence details stay in the backend —
  // the normal result shows execution-time predictions only.
  return (
    <div className="flex items-center gap-3 text-sm">
      <span className="w-28 shrink-0 font-medium text-ink" data-testid={`stage-name-${entry.stage}`}>
        {stageDisplay(entry.stage)}
      </span>
      <span className="w-24 shrink-0 whitespace-nowrap text-right text-ink">
        {entry.available ? formatSeconds(entry.estimate_seconds) : '—'}
      </span>
      <div className="relative h-2.5 min-w-0 flex-1 rounded-full bg-[#efece4]">
        {entry.available && entry.estimate_seconds != null && (
          <div
            className="absolute left-0 top-0 h-2.5 rounded-full bg-bronze"
            style={{ width: `${barWidth(entry.estimate_seconds, total)}%` }}
          />
        )}
      </div>
    </div>
  )
}

function stageDisplay(stage: string): string {
  const names: Record<string, string> = {
    staging: 'Staging',
    prevalidation: 'Prevalidation',
    transformation: 'Transformation',
    pre_load: 'Pre-load',
    data_loading: 'Data Loading',
    datamart_approval: 'Datamart Approval',
    dataloading_approval: 'Dataloading Approval',
  }
  return names[stage] ?? stage
}

function barWidth(seconds: number, total: number | null): number {
  if (total && total > 0) return Math.max(2, Math.min(100, (seconds / total) * 100))
  return 60
}

/** Prediction Insights (2026-10-05 user spec): hidden by default; when opened
 * shows ONLY the plain 2–3-line explanation ("Why this ETA?") and the calibrated
 * conformal prediction interval (90%). No evidence tables, no SHAP, no technical
 * collapsible. All numbers come from the backend source of truth — nothing is
 * computed or invented in the frontend. */
function InsightsPanel({ result }: { result: PredictionResponse }) {
  const [open, setOpen] = useState(false)
  const u = result.uncertainty
  return (
    <div className="mt-3">
      <button
        type="button"
        data-testid="prediction-insights-toggle"
        aria-expanded={open}
        aria-label="Prediction insights"
        title="Prediction insights — why this prediction"
        onClick={() => setOpen((o) => !o)}
        className="inline-flex items-center gap-1.5 rounded-md border border-linen bg-[#fbf8f1] px-2.5 py-1 text-xs font-semibold text-bronze hover:bg-[#f3ead9]"
      >
        <span aria-hidden="true" className="text-sm leading-none">ⓘ</span> Prediction Insights
      </button>
      {open && (
        <div
          data-testid="prediction-insights"
          className="mt-2 rounded-md border border-linen bg-[#fbf8f1] p-3"
        >
          <div className="border-t border-linen pt-3 first:border-t-0 first:pt-0">
            <h3 className="font-serif text-sm font-bold text-ink">Why this ETA?</h3>
            {/* 2026-10-05 (user spec): ONE generated 2–3-line plain
                explanation built from the stage-intelligence layers —
                the exact-sum equation, the non-applicable stages (why
                shorter), and the approval exclusion. Falls back to the
                backend plain explanation only when stage layers are
                unavailable. */}
            <p
              data-testid="plain-explanation"
              className="mt-2 rounded-sm border border-linen bg-white px-2.5 py-2 text-sm leading-relaxed text-ink"
            >
              {whyEtaSummary(result)}
            </p>
            {/* user clarification 2026-10-01: ONLY confidence was removed — the
                calibrated prediction interval is restored as a visible block. */}
            {u && u.available !== false ? (
              <div
                data-testid="prediction-range"
                className="mt-2 rounded-sm border border-linen bg-white px-2.5 py-2"
              >
                <p className="text-[11px] font-semibold uppercase tracking-wide text-bronze">
                  Prediction range
                </p>
                <p data-testid="uncertainty-range" className="mt-0.5 text-sm font-bold text-ink">
                  {formatSeconds(u.lower_seconds)} — {formatSeconds(u.upper_seconds)}
                </p>
              </div>
            ) : (
              <p data-testid="uncertainty-range" className="mt-1 text-xs leading-relaxed text-warmgray">
                Range not available for this prediction.
              </p>
            )}
            {/* 2026-10-06 (user spec): technical drill-down INSIDE the same
                section — the plain explanation stays exactly as it is; the
                dropdown reveals the real per-prediction SHAP feature
                importance on demand. */}
            <ShapDrilldown result={result} />

          </div>
        </div>
      )}
    </div>
  )
}

/** Rebuild the exact request body from the prediction's echoed inputs so the
 * SHAP recompute sees the SAME pre-execution context the user submitted. */
function shapRequestFrom(result: PredictionResponse): PredictionRequestBody {
  const inp = (result.inputs ?? {}) as Record<string, unknown>
  const s = (k: string): string | null => {
    const v = inp[k]
    return typeof v === 'string' && v.trim() !== '' ? v : null
  }
  const body: Record<string, unknown> = {
    loader_name: s('loader_name') ?? '',
    sprint: s('sprint') ?? '',
    dataset: s('dataset') ?? '',
    connection: s('connection') ?? '',
  }
  for (const k of [
    'prevalidation_display_name',
    'transformation_display_name',
    'preval_trans_display_name',
    'loader_type',
    'calling_type',
    'planned_start_time',
  ] as const) {
    const v = s(k)
    if (v) body[k] = v
  }
  const echo = inp['preval_trans_components_echo']
  if (Array.isArray(echo) && echo.length > 0) {
    body['preval_trans_components'] = echo
  }
  return body as unknown as PredictionRequestBody
}

/** Technical SHAP drill-down (2026-10-06 user spec): a collapsed dropdown
 * labeled "Show technical feature importance". Expanding it fetches
 * /explain/prediction (read-only, same inputs) and lists the real SHAP
 * contributors — which features moved this prediction, how much, and in
 * which direction. The plain "Why this ETA?" text above is untouched. */
function ShapDrilldown({ result }: { result: PredictionResponse }) {
  const [open, setOpen] = useState(false)
  const [shap, setShap] = useState<ShapExplanation | null>(null)
  const [error, setError] = useState<string | null>(null)

  async function toggle() {
    const next = !open
    setOpen(next)
    if (next && shap === null && error === null) {
      try {
        setShap(await fetchPredictionShap(shapRequestFrom(result)))
      } catch (err: unknown) {
        setError(err instanceof ApiError ? err.message : 'Feature importance unavailable.')
      }
    }
  }

  const contributors = shap?.top_contributors ?? []
  return (
    <div className="mt-2" data-testid="shap-drilldown">
      <button
        type="button"
        data-testid="shap-drilldown-toggle"
        aria-expanded={open}
        onClick={() => void toggle()}
        className="text-xs font-semibold text-bronze hover:underline"
      >
        {open ? 'Hide technical feature importance' : 'Show technical feature importance'}
      </button>
      {open && (
        <div className="mt-2 rounded-sm border border-linen bg-white px-2.5 py-2">
          {error ? (
            <p className="text-xs text-ember" data-testid="shap-error">{error}</p>
          ) : shap === null ? (
            <p className="text-xs text-warmgray">Computing feature importance…</p>
          ) : !shap.available ? (
            <p className="text-xs text-warmgray" data-testid="shap-unavailable">
              {shap.reason ?? 'Feature importance is not available for this prediction.'}
            </p>
          ) : (
            <>
              <p className="text-[11px] font-semibold uppercase tracking-wide text-bronze">
                SHAP feature importance — this prediction
              </p>
              <ul className="mt-1 space-y-1" data-testid="shap-contributors">
                {contributors.map((c) => (
                  <li key={`${c.stage}-${c.feature}-${c.contribution_rank}`} className="text-xs text-ink">
                    <span className="font-semibold">{c.display_name || c.feature}</span>
                    {c.value !== null && c.value !== undefined && c.value !== '' && (
                      <span className="text-warmgray"> ({String(c.value)})</span>
                    )}
                    {' — '}
                    <span className={c.direction === 'higher' ? 'text-ember' : 'text-sage'}>
                      {c.direction === 'higher' ? '+' : '−'}{Math.abs(c.stage_seconds_effect).toFixed(2)}s
                    </span>
                    <span className="text-warmgray"> on {c.stage_display.toLowerCase()}</span>
                  </li>
                ))}
              </ul>

            </>
          )}
        </div>
      )}
    </div>
  )
}

/** 2026-10-02 short messages for the two refused-history states (user spec):
 * no long paragraph, no "Reason:" block — one clear line. */
function insufficientHistoryMessage(result: PredictionResponse): string {
  const gateCase = result.loader_case?.case
  const loader = result.identity?.loader_name
  const priors = result.loader_case?.prior_count ?? result.history?.prior_loader_executions ?? 0
  const threshold =
    result.loader_case?.threshold ??
    result.history?.eligibility_policy?.min_prior_loader_executions ??
    3
  const named = loader ? ` '${loader}'` : ''
  if (gateCase === 'UNKNOWN_LOADER') {
    return `New loader${named} — it has no prior executions yet, so there is no history to predict from.`
  }
  return `Insufficient history to predict: loader${named} has only ${priors} prior execution(s); the trust threshold is ${threshold}.`
}

/** Total ETA in exact seconds (2026-10-02: seconds are the primary unit). */
function formatSecondsExact(s: number | null | undefined): string {
  if (s === null || s === undefined || !Number.isFinite(s)) return '—'
  return `${s.toFixed(2)} seconds`
}
function ActualOutcomeSection({ result }: { result: PredictionResponse }) {
  const [open, setOpen] = useState(false)
  // 2026-10-02: when the prediction is withheld (new loader / below the
  // trust threshold) there is no predicted total to prefill — the form starts
  // empty so the REAL executed values are captured and become history.
  const withheld = result.state === 'INSUFFICIENT_HISTORY'
  return (
    <div className="mt-3">
      <button
        type="button"
        data-testid="actual-outcome-toggle"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="btn-secondary"
      >
        Enter Actual Post-Execution Values
      </button>
      {open && (
        <div className="mt-3">
          {withheld && (
            <p className="mb-2 text-xs leading-relaxed text-warmgray">
              No prediction was served for this execution — enter what actually ran.
              Recorded executions build this loader's history ({' '}
              {result.history?.prior_loader_executions ?? 0} so far); predictions start once it reaches the trust threshold.
            </p>
          )}
          <ActualOutcomeForm initial={outcomeStateFromPrediction(result)} />
        </div>
      )}
    </div>
  )
}
