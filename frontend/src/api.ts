/**
 * API client for the dltime backend.
 *
 * The Task 04 POST /predict contract is authoritative: this module mirrors it
 * exactly and invents no second API. Requests carry ONLY pre-execution
 * identity fields; historical/configuration features are backend-derived.
 */

export const PredictionStates = [
  'PREDICTED',
  'INSUFFICIENT_HISTORY',
  'UNRESOLVED_IDENTITY',
  'AMBIGUOUS_IDENTITY',
  'INVALID_INPUT',
] as const

export type PredictionState = (typeof PredictionStates)[number]

export interface PredictionRequestBody {
  loader_name: string
  sprint: string
  dataset: string
  /** Task 10.1: required. Validation-only — must agree with the backend's
   * authoritative definition resolution (never overrides it). */
  connection: string
  /** AD-31 legitimate pre-execution configuration (optional). The user's
   * real values for THIS execution; they override definition-derived
   * aggregates for the context row only. Never engineered features.
   * EXP-032 Part F: the source detail table is per-component (calling_type
   * separates PREVALIDATION from TRANSFORMATION), so display names are
   * collected as two independent fields. */
  prevalidation_display_name?: string | null
  transformation_display_name?: string | null
  /** Legacy combined field — audit echo only, superseded by the two
   * independent component display names above. */
  preval_trans_display_name?: string | null
  /** 2026-10-02: per-component rows matching the real detail source — one
   * row per PREVALIDATION/TRANSFORMATION component of this execution.
   * 2026-10-03: `type` (crossreference | dataobject | dataextract | dataset |
   * query | table) rides along with every row — a full pre-execution field. */
  preval_trans_components?: { display_name: string; calling_type: string; type?: string | null; execution_order: number | null }[] | null
  loader_type?: string | null
  calling_type?: string | null
  execution_order?: number | null
  prevalidation_count?: number | null
  transformation_count?: number | null
}

/**
 * Supplementary stage intelligence (Task 06; Prompt 2 availability policy).
 * Backend-derived, PREDICTED-only. Stage estimates are NEVER summed into the
 * total ETA (locked hard rule).
 */
export interface StageIntelligenceEntry {
  stage: string
  prior_positive_count: number
  evidence_count: number
  estimate_seconds: number | null
  estimator: string | null
  available: boolean
  warnings: string[]
  /** AD-31 availability taxonomy for the seven-stage view. */
  availability?: 'ML_PREDICTED' | 'CONTEXTUAL_STATISTICAL' | 'ML_UNAVAILABLE' | 'UNAVAILABLE' | 'DIRECT' | 'LEARNED' | null
  availability_reason?: string | null
  /** AD-31: how the estimate was produced (ml_stage_model vs
   * contextual_hierarchical_statistical). */
  method?: string | null
  /** Approval stages: contextual evidence tier (e.g. LOADER_DATASET, GLOBAL). */
  evidence_tier?: string | null
  evidence_tier_label?: string | null
  n_evidence?: number | null
  /** 2026-10-02: true ONLY when the estimate comes from this loader's OWN
   * approval history (LOADER tier) — broader-context estimates are disclosed
   * but excluded from the total. */
  included_in_total?: boolean | null
  total_exclusion_reason?: string | null
  /** ML stages: serving model facts. */
  model_kind?: string | null
  model_status?: string | null
  n_training_observations?: number | null
  duplicate_of?: string | null
  duplicate_note?: string | null
  note?: string | null
  policy_version?: string | null
  prior_loader_positive_count?: number | null
  dataset_modelable?: boolean | null
}

export interface StageIntelligenceSection {
  available: boolean
  reason?: string
  note?: string
  ordering?: string
  prior_filter?: string
  policy_version?: string | null
  availability_summary?: { direct: number; learned: number; unavailable: number; policy_version: string } | null
  stages: StageIntelligenceEntry[]
}

/**
 * Calibrated uncertainty interval around the point ETA (Task 07). Backend-derived,
 * PREDICTED-only; never present for non-prediction states. 2026-10-03: the
 * reliability label was removed from the system — the interval is the band.
 */
export interface UncertaintySection {
  available: boolean
  method: string
  calibration_version: string
  interval_level: number
  interval_level_display: string
  lower_seconds: number
  upper_seconds: number
  interval_width_seconds: number
  calibration_rows: number
  calibration_window: { first_start?: string; last_start?: string; scope?: string }
  interpretation: string
  /** User-facing confidence for the displayed interval. Backend-derived from
   * the calibrated conformal target level (NOT a frontend invention): the
   * percentage equals the share of calibration residuals the interval is
   * designed to contain. */
  confidence_display?: { percentage: number; meaning: string }
}

export interface PredictionResponse {
  state: PredictionState
  reason: string | null
  predicted_seconds: number | null
  predicted_seconds_human: string | null
  predicted_log1p: number | null
  model_version: string | null
  model_status: string | null
  /** Serving role of the total policy: AUTHORITATIVE_TOTAL = the deterministic
   * sum of the five operational stage predictions. */
  serving_role?: 'AUTHORITATIVE_TOTAL' | 'REFERENCE_ROLLBACK' | null
  /** The retired direct-total reference ETA for the same context
   * (comparison only — never the shown ETA). */
  direct_reference_seconds?: number | null
  /** The retired stage-aware total model's ETA for the same context
   * (traceability/rollback evidence only — never the shown ETA). */
  stage_aware_reference_seconds?: number | null
  inputs: Record<string, unknown>
  identity: {
    dataset?: string
    sprint?: string
    loader_name?: string
    resolution_status?: string
    resolved_connection?: string | null
    supplied_configuration?: Record<string, unknown>
    configuration_provenance?: Record<string, 'user_supplied' | 'definition_derived'>
  }
  history: {
    prior_loader_executions?: number
    prior_loader_conn_executions?: number
    time_since_prev_loader_sec?: number | null
    prior_loader_mean_sec?: number | null
    prior_loader_median_sec?: number | null
    prior_loader_std_sec?: number | null
    previous_loader_duration_sec?: number | null
    global_prior_count?: number
    planned_start_time?: string
    eligibility_policy?: {
      min_prior_loader_executions?: number
      source?: string
    }
  }
  features: Record<string, unknown> | null
  stage_intelligence: StageIntelligenceSection | null
  uncertainty: UncertaintySection | null
  /** 2026-10-02: the THREE loader-history cases (gate). UNKNOWN_LOADER = new
   * loader with 0 priors; BELOW_THRESHOLD = some priors but under the trust
   * threshold; KNOWN_LOADER_TRUSTED = predictions and the total are served. */
  loader_case?: {
    case: 'UNKNOWN_LOADER' | 'BELOW_THRESHOLD' | 'KNOWN_LOADER_TRUSTED' | string
    case_number?: number
    prior_count?: number
    threshold?: number
    total_available?: boolean
    message?: string
  } | null
  /**
   * SHAP model-contribution explanation (Task 08). Backend-derived,
   * PREDICTED-only; explanatory enrichment — never alters the ETA and never
   * present for non-prediction states. Contributions are in the model's
   * log-scale output space (they are NOT seconds).
   */
  explanation: (ExplanationSection & { shap_explanation?: ShapExplanationSection }) | null
  /**
   * Evidence-sentence explainability (evidence-v1). Deterministic template
   * sentences over real, verifiable numbers (no AI/LLM): Layer 1 per-stage
   * sentences vs the loader's own history + Layer 2 what-if counterfactuals
   * for unentered optional stages. PREDICTED-only; never alters the ETA.
   */
  evidence_explanation?: {
    available: boolean
    evidence_version: string
    summary: string
    sentences: string[]
    what_ifs: string[]
    ml_sentence_count: number
    approval_sentence_count: number
  } | null
  /**
   * plain-explain-v1 (2026-10-05): ONE short plain-language reason (max 2
   * sentences, non-technical) for the prediction. PREDICTED-only; never
   * alters the ETA.
   */
  plain_explanation?: string | null
  /**
   * Deterministic operational risk classification (Task 09). Interpretation
   * only — never alters the ETA and never a probability of failure.
   * PREDICTED path: NORMAL/WARNING/HIGH_RISK/LOW_CONFIDENCE/OUT_OF_DISTRIBUTION;
   * INSUFFICIENT_HISTORY state carries only the opaque INSUFFICIENT_HISTORY label.
   */
  risk: RiskSection | null
  warnings: string[]
}

export interface RiskReason {
  code: string
  message: string
  evidence: string
}

export interface RiskSection {
  label: 'NORMAL' | 'WARNING' | 'HIGH_RISK' | 'LOW_CONFIDENCE' | 'OUT_OF_DISTRIBUTION' | 'INSUFFICIENT_HISTORY'
  reasons: RiskReason[]
  evidence: Record<string, number>
  policy_version: string
  /** Prompt 3 §12: operational attention level, not failure probability. */
  interpretation?: string
}

/** One ranked local contributor (Task 08). For the retired learned-total
 * models this was a SHAP contribution in log1p space; under the authoritative
 * sum-of-five-stages policy the contributors ARE the five stage predictions
 * (additive components in exact seconds) plus explicit non-components. */
export interface ExplanationFeature {
  feature: string
  display_name: string
  value: number | string | null
  /** SHAP-space contribution (legacy learned-total payloads). */
  shap_value_log1p?: number
  /** Exact additive component in seconds (sum-of-five-stages payloads). */
  contribution_seconds?: number
  /** False for approval estimates: contextual, never part of the total. */
  included_in_total?: boolean
  direction: 'higher' | 'lower' | 'neutral' | string
  contribution_rank: number
}

/** Real per-stage SHAP/linear contribution (backend stage_shap_explanation).
 * Each entry attributes one registered pre-execution feature's effect on one
 * stage model's own output — never a restatement of the stage predictions. */
export interface ShapContributor {
  contribution_rank: number
  stage: string
  stage_display: string
  feature: string
  display_name: string
  value: unknown
  contribution_log1p: number
  stage_seconds_effect: number
  direction: 'higher' | 'lower'
  human_readable: string
}

export interface ShapExplanationSection {
  available: boolean
  policy_version?: string
  method?: string
  shap_version?: string | null
  stages_explained?: string[]
  stages_skipped?: Record<string, string>
  top_contributors?: ShapContributor[]
  human_readable_summary?: string
}

export interface ExplanationSection {
  method: string
  /** Present for legacy SHAP payloads (retired learned-total models). */
  shap_version?: string
  model_version: string
  output_space: string
  /** Sum-policy payload (authoritative): exact linear decomposition. */
  base_value_seconds?: number
  sum_contributions_seconds?: number
  model_output_seconds?: number
  /** Legacy SHAP payload (retired learned-total models). */
  base_value_log1p?: number
  sum_contributions_log1p?: number
  model_output_log1p?: number
  additivity_abs_error: number
  interpretation: string
  /** Task 10.1: deterministic plain-language summary generated by the backend
   * from the actual ranked signals (no new computation, no causal claims). */
  human_readable_summary?: string
  top_features: ExplanationFeature[]
  features: ExplanationFeature[]
}

// ------------------------------------------------------- Task 11: monitoring

export interface PredictionEventRecord {
  event_id: string
  schema_version: string
  created_at: string
  event_state: PredictionState
  request: {
    loader_name?: string | null
    dataset?: string | null
    sprint?: string | null
    connection?: string | null
    workflow_execution_id?: string | null
    planned_start_time?: string | null
  }
  identity: Record<string, unknown>
  prediction: {
    predicted_total_seconds: number
    model_version: string
    model_status: string
    prior_count: number
    interval_lower_seconds: number
    interval_upper_seconds: number
    interval_level: number
    risk_label: string
    risk_policy_version: string
  } | null
  explanation_digest: Record<string, unknown> | null
  stage_intelligence: {
    note?: string
    stages?: {
      stage: string
      available: boolean
      estimate_seconds: number | null
      evidence_count: number | null
      method?: string | null
      evidence_tier?: string | null
      model_status?: string | null
    }[]
  } | null
  features: Record<string, number> | null
  warnings: string[]
}

export interface TotalErrorRecord {
  signed_error_seconds: number
  absolute_error_seconds: number
  absolute_percentage_error: number | null
  log_absolute_error: number
  within_10_percent: boolean | null
  within_20_percent: boolean | null
  within_30_percent: boolean | null
  within_2x: boolean | null
}

export interface FeedbackRecord {
  feedback_id: string
  schema_version: string
  prediction_event_id: string | null
  outcome_id: string
  linkage: { status: 'MATCHED' | 'UNMATCHED' | 'AMBIGUOUS'; method: string | null; evidence: Record<string, unknown>; linked_at: string }
  identity: { loader_name?: string; dataset?: string; sprint?: string; connection?: string }
  timing: { start_time?: string; end_time?: string | null }
  total_error: TotalErrorRecord | null
  stage_errors: Record<string, { predicted_seconds: number; actual_seconds: number; signed_error_seconds: number; absolute_error_seconds: number; absolute_percentage_error: number | null }> | null
  training_eligible: boolean
  training_eligibility_reason: string
  created_at: string
}

export interface MonitoringSummary {
  schema_version: string
  generated_at: string
  loader_filter: string | null
  label: string
  prediction_attempts: number
  state_counts: Record<string, number>
  actual_outcomes_received: number
  linkage: { matched: number; unmatched: number; ambiguous: number; matched_without_prediction_value: number }
  metrics: {
    sample_count: number
    mae_seconds: number
    median_absolute_error_seconds: number
    rmse_seconds?: number
    mean_signed_error_seconds?: number
    mape_percent: number | null
    within_10_percent: number
    within_20_percent: number
    within_30_percent: number
    within_2x: number
    /** Prompt 3 §18: explicit denominators — never silently drop records. */
    denominators?: { matched_pairs: number; pairs_with_percentage_error: number }
    /** Prompt 3 §17: error-by-slice diagnostics (empty slices omitted). */
    slices?: {
      /** Prompt 3 §17: error-by-slice diagnostics (empty slices omitted).
       * 2026-10-03: the by_reliability slice was removed with the label. */
      by_history_bucket?: Record<string, { sample_count: number; mae_seconds: number; median_absolute_error_seconds: number }>
      by_loader?: Record<string, { sample_count: number; mae_seconds: number; median_absolute_error_seconds: number }>
      stage_metrics?: Record<
        string,
        {
          sample_count: number
          mae_seconds: number
          median_absolute_error_seconds: number
          available_on_matched: number
          scored_on_matched: number
        }
      >
    }
  } | null
  training_feedback: { eligible_actual_count: number; not_eligible_count: number; reason_counts: Record<string, number> }
  /** AD-33 §34: served stage-estimate distributions (summary; no verdict). */
  stage_prediction_distributions?: {
    event_count: number
    stages: Record<string, { n: number; mean_seconds: number; median_seconds: number; min_seconds: number; max_seconds: number; method?: string | null }>
    note: string
  }
  store_warnings?: string[]
}

export async function fetchMonitoringSummary(loader?: string): Promise<MonitoringSummary> {
  const qs = loader ? `?loader=${encodeURIComponent(loader)}` : ''
  const response = await fetch(`${BASE}/monitoring/summary${qs}`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as MonitoringSummary
}

export async function fetchMonitoringEvents(limit = 100): Promise<{ count: number; store_warnings: string[]; events: PredictionEventRecord[] }> {
  const response = await fetch(`${BASE}/monitoring/events?limit=${limit}`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as { count: number; store_warnings: string[]; events: PredictionEventRecord[] }
}

export async function fetchMonitoringFeedback(limit = 100): Promise<{ count: number; store_warnings: string[]; feedback: FeedbackRecord[] }> {
  const response = await fetch(`${BASE}/monitoring/feedback?limit=${limit}`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as { count: number; store_warnings: string[]; feedback: FeedbackRecord[] }
}

const BASE = '/api'

export class ApiError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ApiError'
  }
}

async function parseErrorResponse(response: Response): Promise<string> {
  try {
    const body = await response.json()
    if (typeof body?.detail === 'string') return body.detail
    // FastAPI validation errors: detail is an array of {loc, msg, type}
    if (Array.isArray(body?.detail)) {
      return body.detail
        .map((d: { msg?: string }) => d?.msg ?? JSON.stringify(d))
        .join('; ')
    }
    return JSON.stringify(body)
  } catch {
    return `${response.status} ${response.statusText}`
  }
}

export async function predict(request: PredictionRequestBody): Promise<PredictionResponse> {
  let response: Response
  try {
    response = await fetch(`${BASE}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
  } catch (err) {
    throw new ApiError('Cannot reach the prediction backend. Is it running?')
  }
  if (!response.ok) {
    throw new ApiError(await parseErrorResponse(response))
  }
  return (await response.json()) as PredictionResponse
}

/* ---------- technical SHAP drill-down (2026-10-06 user spec) ---------- */

/** Response of POST /explain/prediction: the standard per-prediction SHAP
 * section, plus state/reason when the prediction itself is withheld. */
export type ShapExplanation = ShapExplanationSection & {
  state?: string
  reason?: string
}

/** SHAP feature importance for ONE prediction — the technical drill-down
 * behind "Why this ETA?". Same body as /predict; READ-ONLY on the backend
 * (no monitoring event persisted, serving untouched). */
export async function fetchPredictionShap(request: PredictionRequestBody): Promise<ShapExplanation> {
  let response: Response
  try {
    response = await fetch(`${BASE}/explain/prediction`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(request),
    })
  } catch (err) {
    throw new ApiError('Cannot reach the prediction backend. Is it running?')
  }
  if (!response.ok) {
    throw new ApiError(await parseErrorResponse(response))
  }
  return (await response.json()) as ShapExplanation
}

export async function health(): Promise<{ status: string; version: string }> {
  const response = await fetch(`${BASE}/health`)
  if (!response.ok) {
    throw new ApiError(await parseErrorResponse(response))
  }
  return (await response.json()) as { status: string; version: string }
}

/** Dataset-wide stage analytics (Task 06; supplementary only). */
export interface LoaderIntelResponse {
  loader_name: string
  loader_intel_version: string
  known: boolean
  known_in_definitions: boolean
  evidence_level: 'NO_HISTORY' | 'EMERGING' | 'SUFFICIENT' | 'STRONG'
  eta_eligible: boolean
  eta_gate: {
    min_prior_executions: number
    source: string
    satisfied: boolean
    explanation: string
  }
  history: {
    eligible_execution_count: number
    first_execution: string | null
    last_execution: string | null
    show_typical: boolean
    median_sec?: number
    mean_sec?: number
    cv?: number
    p90_sec?: number
    max_sec?: number
    trend?: { basis: string; ratio: number; label: string }
  }
  context: {
    datasets: string[]
    connections: string[]
    sprints: string[]
    connection_resolution: Record<string, number>
  }
  definition_context: {
    matched: boolean
    definition_rows?: number
    datasets?: string[]
    sprints?: string[]
    connections?: string[]
  }
  cold_start: {
    applicable: boolean
    reason?: string
    eta_produced?: boolean
    statement?: string
    contextual_intelligence: {
      kind: string
      basis: string
      observation_count: number
      distinct_loaders: number
      median_sec: number
      p90_sec: number
      disclaimer: string
    } | null
    /** Prompt 2: same-connection context when a dataset context also exists. */
    contextual_intelligence_connection?: {
      kind: string
      basis: string
      observation_count: number
      distinct_loaders: number
      median_sec: number
      p90_sec: number
      disclaimer: string
    } | null
    context_note?: string
  }
  behavior?: LoaderBehavior
  stages?: { rows: LoaderStageRow[] }
  configuration?: LoaderConfiguration
  limitations: string[]
}

export async function fetchLoaderIntelligence(loaderName: string): Promise<LoaderIntelResponse> {
  const res = await fetch(`${BASE}/loaders/${encodeURIComponent(loaderName)}`, {
    method: 'GET',
    headers: { 'Content-Type': 'application/json' },
  })
  if (!res.ok) {
    let detail = `HTTP ${res.status}`
    try {
      const body = await res.json()
      if (typeof body?.detail === 'string') detail = body.detail
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail)
  }
  return (await res.json()) as LoaderIntelResponse
}

export interface StageAnalyticsEntry {
  stage: string
  source: string
  description: string
  observation_count: number
  non_positive_count: number
  coverage: number
  eligible_positive_count: number
  median_sec: number | null
  p90_sec: number | null
  max_sec: number | null
  cv: number | null
  modelability: 'MODELABLE' | 'ANALYTICS_ONLY' | 'TOO_SPARSE' | 'NOT_OBSERVED'
  duplicate_of?: string
}

export interface StageAnalyticsResponse {
  stages: StageAnalyticsEntry[]
  config: { modelable_min_observations?: number; analytics_min_observations?: number }
  note: string
}

export async function fetchStageAnalytics(): Promise<StageAnalyticsResponse> {
  let response: Response
  try {
    response = await fetch(`${BASE}/stages/analytics`)
  } catch {
    throw new ApiError('Cannot reach the prediction backend. Is it running?')
  }
  if (!response.ok) {
    throw new ApiError(await parseErrorResponse(response))
  }
  return (await response.json()) as StageAnalyticsResponse
}

// ---------------------------------------------------------------- Model Laboratory (Module 7)

export interface ExperimentRecord {
  experiment_id: string
  kind?: string
  model_family: string
  target_transformation: string
  hyperparameters: Record<string, unknown>
  feature_set_version: string
  feature_names: string[]
  training_population: Record<string, unknown>
  validation_method: Record<string, unknown>
  metrics: {
    pooled_oof?: { mae?: number; median_ae?: number; rmse?: number; within_20pct?: number; within_2x?: number }
    holdout?: { mae?: number; median_ae?: number }
    served_population?: { n?: number; mae?: number; median_ae?: number; within_2x?: number }
    /** Stage-aware-total records additionally nest per-archetype blocks. */
    direct_total?: Record<string, { mae?: number; median_ae?: number; rmse?: number; within_20pct?: number; within_2x?: number; n?: number }>
    stage_aware_total?: Record<string, { mae?: number; median_ae?: number; rmse?: number; within_20pct?: number; within_2x?: number; n?: number }>
  }
  slices?: Record<string, unknown>
  artifact_reference?: string | null
  status: string
  decision: string | null
  rationale?: string | null
  created_at: string
}

export interface ModelsStatus {
  serving_model_version: string
  serving_is_promoted: boolean
  serving_role?: string
  promoted_model: Record<string, unknown> | null
  note: string
}

export interface DriftFeatures {
  overall_status: string
  reference_rows: number
  current_rows: number
  method_numeric: string
  method_categorical: string
  thresholds: Record<string, unknown>
  features: Record<string, { status?: string; statistic?: number | null; method?: string }>
  top_shifted_features?: string[]
  note?: string
}

export interface DriftPopulation {
  overall_status: string
  dimensions: Record<string, { status: string; statistic?: number | null; max_proportion_change?: number }>
}

export interface DriftPerformance {
  status: string
  sample_count: number
  min_required: number
  has_actual_outcomes: boolean
  mae_seconds?: number
  median_absolute_error_seconds?: number
  rmse_seconds?: number
  mean_signed_error_seconds?: number
  note?: string
}

export interface RetrainingReadiness {
  readiness_state: string
  human_gate_required: boolean
  automatic_retraining: string
  signals: {
    eligible_new_actuals: { count: number; threshold: number; met: boolean }
    performance_degradation: { assessed: boolean; met?: boolean; window_medae?: number; baseline_medae?: number }
    feature_drift: { status: string }
    population_drift: { status: string }
  }
  note: string
}

export async function fetchLaboratoryExperiments(): Promise<{ experiments: ExperimentRecord[] }> {
  const response = await fetch(`${BASE}/laboratory/experiments`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as { experiments: ExperimentRecord[] }
}

export async function fetchModelsStatus(): Promise<ModelsStatus> {
  const response = await fetch(`${BASE}/models/status`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as ModelsStatus
}

export async function fetchDriftFeatures(): Promise<DriftFeatures> {
  const response = await fetch(`${BASE}/drift/features`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as DriftFeatures
}

export async function fetchDriftPopulation(): Promise<DriftPopulation> {
  const response = await fetch(`${BASE}/drift/population`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as DriftPopulation
}

export async function fetchDriftPerformance(): Promise<DriftPerformance> {
  const response = await fetch(`${BASE}/drift/performance`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as DriftPerformance
}

export interface ModelRecord {
  model_id?: string
  model_version?: string
  lifecycle_state?: string
  created_at?: string | null
  promoted_at?: string | null
  /** Stage key for single-stage candidate records. */
  stage?: string
  [k: string]: unknown
}

export async function fetchModels(): Promise<{ models: ModelRecord[]; promoted_model_id: string | null; lifecycle_states: string[] }> {
  const response = await fetch(`${BASE}/models`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as { models: ModelRecord[]; promoted_model_id: string | null; lifecycle_states: string[] }
}

export async function postModelPromote(body: { model_id: string; approved_by?: string; reason: string; stages?: string[] }): Promise<unknown> {
  const response = await fetch(`${BASE}/models/promote?model_id=${encodeURIComponent(body.model_id)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved_by: body.approved_by ?? 'local-operator', reason: body.reason, stages: body.stages }),
  })
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return response.json()
}

export interface LoaderDirectoryRow {
  loader_name: string
  executions: number
  succeed: number
  completed_with_warning: number
  failed: number
  median_duration_sec: number | null
  last_seen: string | null
  sprints?: string[]
  datasets?: string[]
  connections?: string[]
  prediction_count: number
  matched_feedback: { absolute_error_seconds: number; absolute_percentage_error: number } | null
}

export interface LoaderBehaviorPoint {
  start_time: string
  seconds: number
  status: string | null
}

export interface LoaderBehavior {
  n: number
  series?: LoaderBehaviorPoint[]
  distribution?: {
    bins: { label: string; count: number }[]
    min_sec?: number
    max_sec?: number
    median_sec?: number
    p25_sec?: number
    p75_sec?: number
    variability?: string
    cv?: number
  }
  recent_median_sec?: number
  median_sec?: number
}

export interface LoaderStageRow {
  stage: string
  display: string
  available?: boolean
  reason?: string
  observations?: number
  coverage_pct?: number
  median_sec?: number
  typical_sec?: number
  p75_sec?: number
  cv?: number
  variability?: string
}

export interface LoaderConfiguration {
  available?: boolean
  reason?: string
  rows?: { label: string; value: string }[]
  basis?: string
}

export async function fetchLoaderDirectory(): Promise<{ loaders: LoaderDirectoryRow[] }> {
  const response = await fetch(`${BASE}/loaders`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as { loaders: LoaderDirectoryRow[] }
}

export async function fetchRetrainingReadiness(): Promise<RetrainingReadiness> {
  const response = await fetch(`${BASE}/retraining/readiness`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as RetrainingReadiness
}

/* ------------------------------------------------------------------ *
 * Drift & Retraining overview (composed, evidence-backed payload).
 * The backend reuses the existing drift/performance/readiness signals —
 * no new statistics. Every status carries its evidence; nothing retrains
 * or promotes automatically.
 * ------------------------------------------------------------------ */

export interface DriftOverviewHeadline {
  data_health: string
  model_health: string
  new_eligible_outcomes: number
  action: string
}

export interface DriftOverviewEvidence {
  id: string
  label: string
  met: boolean
  detail: string
}

export interface DriftOverviewAssessment {
  verdict: string
  tone: string
  narrative: string
  evidence: DriftOverviewEvidence[]
}

export interface DriftFeatureRow {
  feature: string
  status: string
  statistic?: number | null
  method?: string
  monitored_inputs: number
}

/** 2026-10-05: live error of the now-serving model, computed ONLY
 * on matched outcomes recorded after the serving epoch (the moment
 * the current model was activated). Shown from the first outcome;
 * the Health badge stays policy-gated. */
export interface PostPromotionError {
  model_id?: string | null
  promoted_at: string
  outcomes: number
  median_absolute_error_seconds: number
  mean_absolute_error_seconds: number
  source?: string | null
}

export interface ServingEpoch {
  model_id: string
  promoted_at: string
  since: string
  source: string
}

/** Fallback for the "After retraining" column after a page reload:
 * the newest EXP-RETRAIN-* experiment record (per-stage MedAEs). */
export interface LatestCandidate {
  experiment_id: string
  created_at?: string | null
  per_stage: Record<string, { median_ae?: number | null; n?: number | null }>
}

export interface DriftOverview {
  overview_version: string
  scope?: { kind: string; value: string | null }
  headline: DriftOverviewHeadline
  assessment: DriftOverviewAssessment
  data_drift: {
    overall_status: string
    feature_rows: DriftFeatureRow[]
    population_dimensions: { dimension: string; status: string; statistic?: string | null }[]
    reference_window: Record<string, unknown>
  }
  prediction_drift: {
    reference_predictions: { count: number; min_seconds?: number; max_seconds?: number }
    recent_predictions: { count: number; min_seconds?: number; max_seconds?: number }
    statistic?: number | null
    method: string
    status: string
    note: string
  }
  model_performance: {
    overall: DriftPerformance
    stage_health: {
      stages: Record<
        string,
        {
          stage: string
          status: string
          recent_outcomes: number
          min_required: number
          baseline_medae_seconds?: number
          recent_medae_seconds?: number
          post_promotion?: PostPromotionError | null
          note: string
        }
      >
      overall_status: string
      outcomes_with_stage_errors: number
      total_post_promotion?: PostPromotionError | null
      note: string
    }
    serving_epoch?: Record<string, ServingEpoch>
  }
  retraining: {
    readiness_state: string
    human_gate_required: boolean
    automatic_retraining: string
    signals: RetrainingReadiness['signals']
    latest_candidate?: LatestCandidate | null
  }
  notes: string[]
}

/* ------------------------------------------------------------------ *
 * Stage performance lineage (append-only; the source for the
 * Baseline / Previous / Current columns of the comparison table).
 * ------------------------------------------------------------------ */

export interface LineageSnapshot {
  schema_version: string
  stage: string
  role: 'baseline' | 'previous' | 'current'
  model_version: string
  holdout_medae?: number | null
  holdout_n?: number | null
  recent_medae?: number | null
  at?: string | null
}

export interface StageLineageResponse {
  schema_version: string
  stages: Record<string, { baseline?: LineageSnapshot | null; previous?: LineageSnapshot | null; current?: LineageSnapshot | null }>
  active_candidate: {
    kind: 'stage-candidate' | 'bundle-candidate'
    stages?: string[]
    candidate_versions?: Record<string, string | null>
    candidate_version?: string
    activated_at?: string | null
  } | null
}

export async function fetchStageLineage(): Promise<StageLineageResponse> {
  const response = await fetch(`${BASE}/monitoring/stage-lineage`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as StageLineageResponse
}

export async function fetchDriftOverview(loader?: string): Promise<DriftOverview> {
  const qs = loader ? `?loader=${encodeURIComponent(loader)}` : ''
  const response = await fetch(`${BASE}/drift/overview${qs}`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as DriftOverview
}

export interface LifecycleEvent {
  at: string
  model_id: string
  event_type: string
  kind: string
  label: string
  reason?: string
}

export async function fetchModelsLifecycle(): Promise<{ events: LifecycleEvent[] }> {
  const response = await fetch(`${BASE}/models/lifecycle`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as { events: LifecycleEvent[] }
}

export async function postModelRollback(body: { approved_by?: string; reason: string; stages?: string[] }): Promise<unknown> {
  const response = await fetch(`${BASE}/models/rollback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved_by: body.approved_by ?? 'local-operator', reason: body.reason, stages: body.stages }),
  })
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return response.json()
}

// --------------------------------------------------- Retraining workflow (locked §9-16)

export interface RetrainingDecision {
  decision_id: string
  decision_type: 'CONTINUE_MONITORING' | 'START_RETRAINING'
  decided_by: string
  reason: string
  decided_at: string
  model_version_at_decision: string
  assessment_snapshot: { verdict?: string | null; data_health?: string | null; model_health?: string | null; new_eligible_outcomes?: number | null }
}

export async function fetchRetrainingDecisions(): Promise<RetrainingDecision[]> {
  const response = await fetch(`${BASE}/retraining/decisions`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  const body = (await response.json()) as { decisions: RetrainingDecision[] }
  return body.decisions ?? []
}

export async function postRetrainingDecision(body: {
  decision_type: string
  reason: string
  decided_by?: string
}): Promise<unknown> {
  const response = await fetch(`${BASE}/retraining/decide`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...body, decided_by: body.decided_by ?? 'local-operator' }),
  })
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return response.json()
}

export interface CandidateEvaluation {
  candidate_version: string
  verdict: 'ACCEPTABLE_FOR_REVIEW' | 'NOT_SUFFICIENTLY_BETTER' | 'INSUFFICIENT_EVIDENCE'
  rationale: string
  per_stage: Array<{ stage: string; candidate_median_ae: number | null; baseline_median_ae: number | null; ratio: number | null; candidate_n: number | null }>
  total: { candidate_median_ae: number | null; baseline_median_ae: number | null; ratio: number | null; candidate_n: number | null; baseline_n?: number | null }
  human_gate_required: boolean
  automatic_promotion: string
}

export interface TrainCandidateResponse {
  /** Single-stage training returns ONE record; multi-stage returns a list. */
  registered: ModelRecord | ModelRecord[]
  evaluation: CandidateEvaluation | CandidateEvaluation[]
  /** Stage keys trained (present on multi-stage responses). */
  stages?: string[]
  serving_untouched: boolean
  serving_version_before: string
}

export async function postTrainCandidate(body: {
  approved_by?: string
  reason: string
  /** 2026-10-05: per-stage retraining — retrain ONE stage with the
   * identical protocol. Omitted = the whole five-stage bundle (the
   * original path, unchanged). */
  stages?: string[]
}): Promise<TrainCandidateResponse> {
  const response = await fetch(`${BASE}/retraining/train-candidate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ approved_by: body.approved_by ?? 'local-operator', reason: body.reason, stages: body.stages }),
  })
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return response.json()
}

export interface ServingStatus {
  serving_version: string
  candidate_override_active: boolean
  candidate_version: string | null
}

export async function fetchServingStatus(): Promise<ServingStatus> {
  const response = await fetch(`${BASE}/retraining/serving-status`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return response.json()
}

// ---------------------------------------------------------------- Overview (final UI integration)

export interface OverviewRecentEvent {
  created_at: string | null
  event_state: string | null
  loader_name: string | null
  dataset: string | null
  sprint: string | null
  predicted_seconds: number | null
  risk_label: string | null
  workflow_execution_id: string | null
}

export interface OverviewReport {
  generated_at: string
  data: {
    total_loaders: number
    total_executions: number
    training_eligible: number
    median_duration_sec: number
    p95_duration_sec: number
    duration_stat_definition?: string
    duration_target_source_counts?: Record<string, number>
    duration_span_consistency?: { checked: number; within_5pct: number } | null
  }
  quality: {
    resolution: Record<string, number>
    eligibility_counts: Record<string, number>
  }
  sources: { name: string; role: string; rows: number; columns: number }[]
  model: {
    model_version: string
    model_status: string | null
    feature_version: string | null
    feature_count: number
    calibration: { version: string; coverage_90: number | null; rows: number | null }
  }
  monitoring: {
    prediction_attempts: number
    state_counts: Record<string, number>
    actual_outcomes: number
    matched_outcomes: number
  }
  retraining: { readiness_state: string; automatic_retraining: string }
  drift: {
    feature_status: string
    population_status: string
    performance_status: string
  }
  recent_events: OverviewRecentEvent[]
}

export async function fetchOverview(): Promise<OverviewReport> {
  const response = await fetch(`${BASE}/overview`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as OverviewReport
}

// ------------------------------------------- AD-32/33: feedback + notifications

/** XLSX-faithful post-execution outcome body (AD-32 §26 + AD-34). Stage
 * durations use the source's MINUTES semantics; the authoritative total is
 * `total_timetaken_in_min` (EXP-032: matches the physical Start/End span;
 * legacy seconds field is corrupted). */
export interface ActualOutcomeBody {
  loader_name: string
  dataset: string
  sprint: string
  connection: string
  workflow_execution_id?: string | null
  ldr_execution_id?: string | null
  start_time: string
  end_time?: string | null
  actual_total_timetaken_in_min: number
  actual_total_seconds?: number | null
  status?: string | null
  total_records?: number | null
  success_records?: number | null
  error_records?: number | null
  prevalidation_errors?: number | null
  format_errors?: number | null
  constraint_errors?: number | null
  application_errors?: number | null
  cascade_errors?: number | null
  staging_timetaken_in_min?: number | null
  prevalidation_timetaken_in_min?: number | null
  transformation_timetaken_in_min?: number | null
  pre_load_timetaken_in_min?: number | null
  data_loading_timetaken_in_min?: number | null
  datamart_approval_timetaken_in_min?: number | null
  dataloading_approval_timetaken_in_min?: number | null
  /** Legacy free-form seconds map; explicit XLSX-faithful fields win. */
  stage_actuals?: Record<string, number>
}

export interface ActualIngestResponse {
  outcome_id: string
  linkage_status: 'MATCHED' | 'UNMATCHED' | 'AMBIGUOUS'
  linkage_method: string | null
  /** 2026-10-05: plain reason why linkage succeeded/failed. */
  linkage_reason?: string | null
  prediction_event_id: string | null
  feedback_id: string | null
  total_error: {
    signed_error_seconds: number
    absolute_error_seconds: number
    absolute_percentage_error: number | null
    within_10_percent: boolean | null
    within_20_percent: boolean | null
    within_2x: boolean | null
  } | null
  stage_errors: Record<string, unknown> | null
  training_eligible: boolean
  training_eligibility_reason: string | null
  duplicate: boolean
  warnings: string[]
}

export async function postActualOutcome(body: ActualOutcomeBody): Promise<ActualIngestResponse> {
  const response = await fetch(`${BASE}/feedback/actual`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as ActualIngestResponse
}

export interface NotificationRecord {
  notification_id: string
  notification_key: string
  alert_type: string
  severity: 'INFO' | 'WARNING' | 'HIGH' | 'CRITICAL'
  title: string
  message: string
  context: Record<string, unknown>
  created_at: string
  read: boolean
  read_at: string | null
}

export interface NotificationsFeed {
  notifications: NotificationRecord[]
  unread_count: number
  read_warnings?: string[]
  evaluated_alerts?: number
  new_notifications?: number
}

export async function fetchNotifications(): Promise<NotificationsFeed> {
  const response = await fetch(`${BASE}/notifications`)
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as NotificationsFeed
}

export async function evaluateNotifications(): Promise<NotificationsFeed> {
  const response = await fetch(`${BASE}/notifications/evaluate`, { method: 'POST' })
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as NotificationsFeed
}

export async function markNotificationRead(notificationId: string): Promise<NotificationsFeed> {
  const response = await fetch(`${BASE}/notifications/${encodeURIComponent(notificationId)}/read`, {
    method: 'POST',
  })
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as NotificationsFeed
}

export async function markAllNotificationsRead(): Promise<NotificationsFeed> {
  const response = await fetch(`${BASE}/notifications/read-all`, { method: 'POST' })
  if (!response.ok) throw new ApiError(await parseErrorResponse(response))
  return (await response.json()) as NotificationsFeed
}
