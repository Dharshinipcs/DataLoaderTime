import { useMemo, useState } from 'react'
import { postActualOutcome, type ActualIngestResponse, type PredictionResponse } from '../api'

type Phase =
  | { kind: 'idle' }
  | { kind: 'busy' }
  | { kind: 'done'; result: ActualIngestResponse }
  | { kind: 'error'; message: string }

// 2026-10-03 (user spec): ALL timing inputs are entered in SECONDS —
// predictions are seconds, so actuals are captured in seconds too. The stored
// keys keep the source's *_in_min names; values are converted to minutes at
// submission (÷60) so the historical record stays in its native unit.
const STAGE_MINUTE_FIELDS: { key: keyof ActualOutcomeFormState; label: string }[] = [
  { key: 'staging_timetaken_in_min', label: 'Staging (sec)' },
  { key: 'prevalidation_timetaken_in_min', label: 'Prevalidation (sec)' },
  { key: 'transformation_timetaken_in_min', label: 'Transformation (sec)' },
  { key: 'pre_load_timetaken_in_min', label: 'Pre-load (sec)' },
  { key: 'data_loading_timetaken_in_min', label: 'Data Loading (sec)' },
  { key: 'datamart_approval_timetaken_in_min', label: 'Datamart Approval (sec)' },
  { key: 'dataloading_approval_timetaken_in_min', label: 'Dataloading Approval (sec)' },
]

const ERROR_COUNT_FIELDS: { key: keyof ActualOutcomeFormState; label: string }[] = [
  { key: 'prevalidation_errors', label: 'Prevalidation errors' },
  { key: 'format_errors', label: 'Format errors' },
  { key: 'constraint_errors', label: 'Constraint errors' },
  { key: 'application_errors', label: 'Application errors' },
  { key: 'cascade_errors', label: 'Cascade errors' },
]

const RECORD_COUNT_FIELDS: { key: keyof ActualOutcomeFormState; label: string }[] = [
  { key: 'total_records', label: 'Total records' },
  { key: 'success_records', label: 'Success records' },
  { key: 'error_records', label: 'Error records' },
]

export interface ActualOutcomeFormState {
  loader_name: string
  dataset: string
  sprint: string
  connection: string
  workflow_execution_id: string
  ldr_execution_id: string
  start_time: string
  end_time: string
  actual_total_timetaken_in_min: string
  status: string
  total_records: string
  success_records: string
  error_records: string
  prevalidation_errors: string
  format_errors: string
  constraint_errors: string
  application_errors: string
  cascade_errors: string
  staging_timetaken_in_min: string
  prevalidation_timetaken_in_min: string
  transformation_timetaken_in_min: string
  pre_load_timetaken_in_min: string
  data_loading_timetaken_in_min: string
  datamart_approval_timetaken_in_min: string
  dataloading_approval_timetaken_in_min: string
}

export function emptyOutcomeState(): ActualOutcomeFormState {
  return {
    loader_name: '',
    dataset: '',
    sprint: '',
    connection: '',
    workflow_execution_id: '',
    ldr_execution_id: '',
    start_time: '',
    end_time: '',
    actual_total_timetaken_in_min: '',
    status: 'SUCCEED',
    total_records: '',
    success_records: '',
    error_records: '',
    prevalidation_errors: '',
    format_errors: '',
    constraint_errors: '',
    application_errors: '',
    cascade_errors: '',
    staging_timetaken_in_min: '',
    prevalidation_timetaken_in_min: '',
    transformation_timetaken_in_min: '',
    pre_load_timetaken_in_min: '',
    data_loading_timetaken_in_min: '',
    datamart_approval_timetaken_in_min: '',
    dataloading_approval_timetaken_in_min: '',
  }
}

/** Prefill identity + optional workflow reference from a prediction result.
 * 2026-10-02: when the prediction was WITHHELD (insufficient history —
 * new loader / below the trust threshold) there is no predicted total to
 * prefill; the actual-total field starts empty so only real executed values
 * are recorded as this loader's history. */
export function outcomeStateFromPrediction(result: PredictionResponse): ActualOutcomeFormState {
  const base = emptyOutcomeState()
  return {
    ...base,
    loader_name: result.identity?.loader_name ?? '',
    dataset: result.identity?.dataset ?? '',
    sprint: result.identity?.sprint ?? '',
    connection: result.identity?.resolved_connection ?? '',
    actual_total_timetaken_in_min:
      result.state === 'PREDICTED' && result.predicted_seconds != null
        ? String(Math.round(result.predicted_seconds)) // prefilled in SECONDS (2026-10-03)
        : '',
    status: 'SUCCEED',
  }
}

const inputCls =
  'w-full rounded border border-linen bg-white px-2 py-1.5 text-sm text-ink outline-none focus:border-bronze'

function Num(props: {
  label: string
  value: string
  onChange: (v: string) => void
  step?: string
  testid?: string
}) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium text-warmgray">{props.label}</span>
      <input
        type="number"
        min="0"
        step={props.step ?? 'any'}
        className={inputCls}
        value={props.value}
        data-testid={props.testid}
        onChange={(e) => props.onChange(e.target.value)}
      />
    </label>
  )
}

function Text(props: { label: string; value: string; onChange: (v: string) => void; testid?: string }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[11px] font-medium text-warmgray">{props.label}</span>
      <input
        type="text"
        className={inputCls}
        value={props.value}
        data-testid={props.testid}
        onChange={(e) => props.onChange(e.target.value)}
      />
    </label>
  )
}

/**
 * AD-32 §25/§26: post-execution outcome entry (temporary manual ingestion —
 * the future DataZap integration posts the SAME body to /feedback/actual).
 * Fields mirror Loader_Executions.xlsx: identity, status, record/error
 * counts, stage durations in MINUTES, and the execution-level seconds total.
 */
export default function ActualOutcomeForm(props: {
  initial?: ActualOutcomeFormState
  compact?: boolean
  onIngested?: (result: ActualIngestResponse) => void
}) {
  const [state, setState] = useState<ActualOutcomeFormState>(props.initial ?? emptyOutcomeState())
  const [phase, setPhase] = useState<Phase>({ kind: 'idle' })

  const set = (key: keyof ActualOutcomeFormState) => (v: string) =>
    setState((s) => ({ ...s, [key]: v }))

  // AD-38 §18: the user must never hand-calculate a derived value. When both
  // timestamps are known, the total duration can be DERIVED from them; the
  // submitted record discloses the derivation via the derived flag below.
  // 2026-10-03: Start/End use the FIXED format YYYY-MM-DD HH:MM:SS (the
  // datetime-local picker guarantees it); the derived note also shows the
  // seconds value since all predictions are communicated in seconds.
  const derivedTotalMinutes = useMemo<{ minutes: number | null; seconds: number | null; note: string }>(() => {
    const s = state.start_time.trim()
    const e = state.end_time.trim()
    if (!s || !e) return { minutes: null, seconds: null, note: '' }
    const sd = new Date(s)
    const ed = new Date(e)
    if (Number.isNaN(sd.getTime()) || Number.isNaN(ed.getTime()) || ed.getTime() <= sd.getTime()) {
      return { minutes: null, seconds: null, note: 'End must be after Start to derive the duration.' }
    }
    const seconds = (ed.getTime() - sd.getTime()) / 1000
    const minutes = seconds / 60
    return {
      minutes: Math.round(minutes * 100) / 100,
      seconds: Math.round(seconds * 100) / 100,
      note: `Duration derived from Start/End: ${minutes.toFixed(2)} min (${Math.round(seconds)} seconds).`,
    }
  }, [state.start_time, state.end_time])

  const totalMinutesExplicit =
    state.actual_total_timetaken_in_min.trim() !== '' &&
    !Number.isNaN(Number(state.actual_total_timetaken_in_min))

  const canSubmit = useMemo(
    () =>
      state.loader_name.trim() !== '' &&
      state.dataset.trim() !== '' &&
      state.sprint.trim() !== '' &&
      state.connection.trim() !== '' &&
      state.start_time.trim() !== '' &&
      (totalMinutesExplicit || derivedTotalMinutes.minutes != null),
    [state, totalMinutesExplicit, derivedTotalMinutes],
  )

  const submit = async () => {
    if (!canSubmit || phase.kind === 'busy') return
    setPhase({ kind: 'busy' })
    const numOrNull = (v: string): number | null => {
      const t = v.trim()
      if (t === '') return null
      const n = Number(t)
      return Number.isNaN(n) ? null : n
    }
    // 2026-10-03: Start/End are submitted in the FIXED "YYYY-MM-DD HH:MM:SS"
    // form — the picker's value is normalized (T -> space, seconds forced to
    // two digits, stray milliseconds dropped) so the record can never carry
    // another format. Every timing the user typed is in SECONDS and is
    // converted to the record's native minutes (÷60) at the boundary.
    const toRecordTimestamp = (v: string): string => {
      const m = v.trim().match(/^(\d{4}-\d{2}-\d{2})[T ](\d{2}:\d{2})(?::(\d{2}))?(?:\.\d+)?$/)
      if (!m) return v.trim().replace('T', ' ')
      return `${m[1]} ${m[2]}:${m[3] ?? '00'}`
    }
    const body = {
      loader_name: state.loader_name.trim(),
      dataset: state.dataset.trim(),
      sprint: state.sprint.trim(),
      connection: state.connection.trim(),
      workflow_execution_id: state.workflow_execution_id.trim() || null,
      ldr_execution_id: state.ldr_execution_id.trim() || null,
      start_time: toRecordTimestamp(state.start_time),
      end_time: state.end_time.trim() ? toRecordTimestamp(state.end_time) : null,
      // AD-38 §18: explicit seconds win; otherwise the duration is derived
      // from Start/End (disclosed in the API body as derived_from_timestamps).
      actual_total_timetaken_in_min: totalMinutesExplicit
        ? Number(state.actual_total_timetaken_in_min) / 60
        : (derivedTotalMinutes.minutes as number),
      actual_total_derived_from_timestamps: !totalMinutesExplicit && derivedTotalMinutes.minutes != null,
      status: state.status || null,
      ...Object.fromEntries(
        [...RECORD_COUNT_FIELDS, ...ERROR_COUNT_FIELDS].map(({ key }) => [
          key,
          numOrNull(state[key]),
        ]),
      ),
      ...Object.fromEntries(
        STAGE_MINUTE_FIELDS.map(({ key }) => {
          const secs = numOrNull(state[key])
          return [key, secs === null ? null : secs / 60]
        }),
      ),
    }
    try {
      const result = await postActualOutcome(body)
      setPhase({ kind: 'done', result })
      props.onIngested?.(result)
    } catch (err) {
      setPhase({
        kind: 'error',
        message: err instanceof Error ? err.message : 'Could not submit the actual outcome.',
      })
    }
  }

  if (phase.kind === 'done') {
    const r = phase.result
    return (
      <div data-testid="outcome-done" className="rounded-md border border-linen bg-[#fbf8f1] p-4">
        <p className="text-sm font-semibold text-ink">
          Outcome recorded{r.duplicate ? ' (duplicate — already ingested)' : ''}.
        </p>
        <p className="mt-1 text-xs text-warmgray">
          Linkage: <span className="font-semibold text-ink">{r.linkage_status}</span>
          {r.linkage_method ? ` (${r.linkage_method})` : ''} · Training-eligible:{' '}
          <span className="font-semibold text-ink">{r.training_eligible ? 'yes' : 'no'}</span>
          {r.training_eligibility_reason ? ` — ${r.training_eligibility_reason}` : ''}
        </p>
        {/* 2026-10-05 (user request): an UNMATCHED entry must say WHY, in
            plain language, so the operator knows exactly what to correct. */}
        {r.linkage_status !== 'MATCHED' && r.linkage_reason && (
          <p className="mt-1 text-xs text-warmgray" data-testid="linkage-reason">
            Why: {r.linkage_reason}
          </p>
        )}
        {r.total_error && (
          <div className="mt-2 rounded border border-linen bg-white p-2 text-xs text-warmgray" data-testid="outcome-errors">
            <p className="font-semibold text-ink">Prediction vs actual (computed automatically):</p>
            <p>
              Signed error: {r.total_error.signed_error_seconds.toFixed(2)} s · Absolute:{' '}
              {r.total_error.absolute_error_seconds.toFixed(2)} s
              {r.total_error.absolute_percentage_error != null &&
                ` · ${r.total_error.absolute_percentage_error.toFixed(1)}%`}
            </p>
            {r.stage_errors && Object.keys(r.stage_errors).length > 0 && (
              <p className="mt-1">
                Stage errors computed for: {Object.keys(r.stage_errors).join(', ')}
              </p>
            )}
          </div>
        )}
        {r.warnings.length > 0 && (
          <p className="mt-2 text-xs text-bronze" data-testid="outcome-warnings">
            Data-quality warnings: {r.warnings.join('; ')}
          </p>
        )}
        <button
          type="button"
          className="mt-3 rounded border border-linen bg-white px-3 py-1.5 text-xs font-medium text-ink hover:bg-parchment"
          onClick={() => setPhase({ kind: 'idle' })}
        >
          Record another outcome
        </button>
      </div>
    )
  }

  return (
    <div className="rounded-md border border-linen bg-parchment p-4" data-testid="actual-outcome-form">
      <p className="text-xs font-semibold uppercase tracking-[0.12em] text-warmgray">
        Post-execution outcome (actual results)
      </p>

      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-3">
        <Text label="Loader name" value={state.loader_name} onChange={set('loader_name')} testid="out-loader" />
        <Text label="Dataset name" value={state.dataset} onChange={set('dataset')} testid="out-dataset" />
        <Text label="Sprint" value={state.sprint} onChange={set('sprint')} testid="out-sprint" />
        <Text label="Connection" value={state.connection} onChange={set('connection')} testid="out-connection" />
        <Text
          label="Workflow execution ID (optional linkage)"
          value={state.workflow_execution_id}
          onChange={set('workflow_execution_id')}
          testid="out-workflow"
        />
        <Text
          label="LDR execution ID (optional)"
          value={state.ldr_execution_id}
          onChange={set('ldr_execution_id')}
        />
        {/* 2026-10-03: FIXED datetime format via the native picker with a
            seconds step — the value is always YYYY-MM-DD HH:MM:SS. */}
        <label className="block">
          <span className="mb-1 block text-[11px] font-medium text-warmgray">Start time (YYYY-MM-DD HH:MM:SS)</span>
          <input
            type="datetime-local"
            step="1"
            className={inputCls}
            value={state.start_time}
            data-testid="out-start"
            onChange={(e) => set('start_time')(e.target.value)}
          />
        </label>
        <label className="block">
          <span className="mb-1 block text-[11px] font-medium text-warmgray">End time (YYYY-MM-DD HH:MM:SS)</span>
          <input
            type="datetime-local"
            step="1"
            className={inputCls}
            value={state.end_time}
            data-testid="out-end"
            onChange={(e) => set('end_time')(e.target.value)}
          />
        </label>
        <Text label="Status (e.g. SUCCEED)" value={state.status} onChange={set('status')} />
      </div>
      {derivedTotalMinutes.note && (
        <p data-testid="derived-duration-note" className="mt-1 text-[11px] text-bronze">
          {derivedTotalMinutes.note} You never need to calculate it yourself — enter it only if you
          know it independently (the system validates the two against each other).
        </p>
      )}

      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
        <Num
          label="Total time taken (sec)"
          value={state.actual_total_timetaken_in_min}
          onChange={set('actual_total_timetaken_in_min')}
          testid="out-total"
        />
        {RECORD_COUNT_FIELDS.map((f) => (
          <Num key={f.key} label={f.label} value={state[f.key]} onChange={set(f.key)} />
        ))}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-4">
        {STAGE_MINUTE_FIELDS.map((f) => (
          <Num key={f.key} label={f.label} value={state[f.key]} onChange={set(f.key)} />
        ))}
      </div>

      <div className="mt-3 grid grid-cols-2 gap-3 sm:grid-cols-5">
        {ERROR_COUNT_FIELDS.map((f) => (
          <Num key={f.key} label={f.label} value={state[f.key]} onChange={set(f.key)} />
        ))}
      </div>

      {phase.kind === 'error' && (
        <p className="mt-2 text-xs text-[#8c2f1d]" data-testid="outcome-error">
          {phase.message}
        </p>
      )}

      <button
        type="button"
        data-testid="outcome-submit"
        disabled={!canSubmit || phase.kind === 'busy'}
        onClick={submit}
        className="mt-4 rounded bg-charcoal px-4 py-2 text-xs font-semibold text-[#f7f4ee] hover:opacity-90 disabled:opacity-40"
      >
        {phase.kind === 'busy' ? 'Saving…' : 'Save Actual Outcome'}
      </button>
    </div>
  )
}
