/**
 * Prediction Studio tests (Task 05).
 *
 * Contract assertions only — no brittle hardcoded prediction numbers.
 * fetch is mocked per test to return the actual backend response shapes.
 */
import { describe, expect, it, vi, beforeEach } from 'vitest'
import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import PredictionStudio from '../pages/PredictionStudio'

async function fillRequiredFields(user: ReturnType<typeof userEvent.setup>) {
  // user-event calls must be awaited sequentially (never Promise.all)
  await user.type(screen.getByLabelText(/Loader Name/i), 'LoaderA')
  await user.type(screen.getByLabelText(/Sprint/i), 'S1')
  await user.type(screen.getByLabelText(/Dataset/i), 'DS1')
  await user.type(screen.getByLabelText(/^Connection/i), 'test1')
}

const STAGE_INTEL = {
  available: true,
  policy_version: 'stage-intel-v3-ad31',
  availability_summary: { ml_predicted: 5, ml_unavailable: 0, contextual_statistical: 1, unavailable: 0, policy_version: 'stage-intel-v3-ad31' },
  note: 'Applicable-Stage Execution View (5 stages for this execution): ML stage predictions + contextual statistical approval estimates. The total is the EXACT sum of the applicable stage predictions plus the approval estimates when available. ML stage outputs are model predictions, never actuals.',
  ordering: 'Start_Time, tie-break LDR_Execution_Id (deterministic)',
  prior_filter: 'training_eligible',
  stages: [
    // Canonical operational order: staging, prevalidation, transformation,
    // pre_load, data_loading (matches the ML_STAGES sum order); the five
    // values sum to EXACTLY 18.16 = the mock's predicted_seconds.
    {
      stage: 'staging',
      prior_positive_count: 9,
      evidence_count: 9,
      estimate_seconds: 1.4,
      estimator: 'stage extra_trees (AD-31)',
      available: true,
      warnings: [],
      method: 'ml_stage_model',
      availability: 'ML_PREDICTED',
      availability_reason: 'Stage ML prediction (extra_trees, trained on 326 eligible observations; experimental candidate, not promoted).',
      model_kind: 'extra_trees',
      model_status: 'EXPERIMENTAL',
      n_training_observations: 326,
      prior_loader_positive_count: 3,
      duplicate_of: null,
    },
    {
      stage: 'prevalidation',
      prior_positive_count: 4,
      evidence_count: 4,
      estimate_seconds: 0.85,
      estimator: 'stage extra_trees (AD-31)',
      available: true,
      warnings: [],
      method: 'ml_stage_model',
      availability: 'ML_PREDICTED',
      availability_reason: 'Stage ML prediction (extra_trees, trained on 65 eligible observations; experimental candidate, not promoted).',
      model_kind: 'extra_trees',
      model_status: 'EXPERIMENTAL',
      n_training_observations: 65,
      prior_loader_positive_count: 0,
      duplicate_of: null,
    },
    {
      stage: 'transformation',
      prior_positive_count: 5,
      evidence_count: 5,
      estimate_seconds: 0.76,
      estimator: 'stage random_forest (AD-31)',
      available: true,
      warnings: [],
      method: 'ml_stage_model',
      availability: 'ML_PREDICTED',
      availability_reason: 'Stage ML prediction (random_forest, trained on 60 eligible observations; experimental candidate, not promoted).',
      model_kind: 'random_forest',
      model_status: 'EXPERIMENTAL',
      n_training_observations: 60,
      prior_loader_positive_count: 2,
      duplicate_of: null,
    },
    {
      stage: 'pre_load',
      prior_positive_count: 2,
      evidence_count: 2,
      estimate_seconds: 13.2,
      estimator: null,
      available: true,
      warnings: [],
      method: 'ml_stage_model',
      availability: 'ML_PREDICTED',
      availability_reason:
        'Stage ML prediction (extra_trees, trained on 19 eligible observations; SMALL-SAMPLE stage — honest high-variance limitation disclosed, model remains an experimental candidate; experimental candidate, not promoted).',
      model_kind: 'extra_trees',
      model_status: 'EXPERIMENTAL_SMALL_SAMPLE',
      n_training_observations: 19,
      prior_loader_positive_count: 1,
      duplicate_of: null,
    },
    {
      stage: 'data_loading',
      prior_positive_count: 6,
      evidence_count: 6,
      estimate_seconds: 1.95,
      estimator: 'stage ridge (AD-31)',
      available: true,
      warnings: [],
      method: 'ml_stage_model',
      availability: 'ML_PREDICTED',
      availability_reason: 'Stage ML prediction (ridge, trained on 313 eligible observations; experimental candidate, not promoted).',
      model_kind: 'ridge',
      model_status: 'EXPERIMENTAL',
      n_training_observations: 313,
      prior_loader_positive_count: 1,
      duplicate_of: null,
    },
    {
      stage: 'dataloading_approval',
      prior_positive_count: 0,
      evidence_count: 6,
      estimate_seconds: 2.7,
      estimator: 'contextual hierarchical median (approval-hier-v1)',
      available: true,
      warnings: [],
      method: 'contextual_hierarchical_statistical',
      availability: 'CONTEXTUAL_STATISTICAL',
      availability_reason: 'Contextual statistical estimate from strictly-prior executions.',
      evidence_tier: 'GLOBAL',
      evidence_tier_label: 'Global historical median',
      n_evidence: 6,
      prior_loader_positive_count: 0,
      duplicate_of: null,
    },
  ],
}

const UNCERTAINTY = {
  available: true,
  method: 'log1p_space_absolute_residual_conformal',
  calibration_version: 'unc-log1p-conformal-v1',
  interval_level: 0.9,
  interval_level_display: '90%',
  lower_seconds: 1.061,
  upper_seconds: 176.99,
  interval_width_seconds: 175.929,
  calibration_rows: 88,
  calibration_window: {
    first_start: '2026-07-15 17:48:04.915000',
    last_start: '2026-08-13 14:41:43.809000',
    scope: 'final chronological holdout',
  },
  interpretation:
    'Based on the model historical errors on earlier executions, the actual duration is estimated to fall within this range. This is an estimated range, not a guarantee.',
  // 2026-10-03: the reliability label was removed from the system — the
  // calibrated interval itself is the honesty mechanism; no label is mocked.
  confidence_display: {
    percentage: 90,
    meaning:
      'Designed to contain the actual duration 90% of the time, based on the errors of the current stage models on recent executions.',
  },
}

const EXPLANATION = {
  method: 'deterministic_stage_sum',
  model_version: 'sum-of-five-stages-v1',
  output_space:
    'seconds: the total is exactly the sum of the applicable operational stage predictions plus the available approval estimates (linear additivity of the displayed values; no learned total model and no log-space attribution is involved)',
  base_value_seconds: 0,
  sum_contributions_seconds: 18.16,
  model_output_seconds: 18.16,
  model_output_log1p: 3.217,
  additivity_abs_error: 0,
  interpretation:
    'Each entry below is one of the applicable stage predictions or approval estimates that sum exactly to the total — actual model / statistical outputs, not causal effects and not guarantees.',
  human_readable_summary:
    'The predicted total is the exact sum of the applicable stage predictions and approval estimates: Pre-load prediction 13.2s + Data Loading prediction 1.95s + Staging prediction 1.4s + Prevalidation prediction 0.85s + Transformation prediction 0.76s + Datamart Approval estimate 98.4s + Dataloading Approval estimate 2.7s = 18.16s. Operational stages are ML model outputs; approvals are historical-median statistics.',
  top_features: [
    {
      feature: 'pred_stage_pre_load_sec',
      display_name: 'Pre-load prediction',
      value: 13.2,
      contribution_seconds: 13.2,
      included_in_total: true,
      direction: 'component of the total',
      contribution_rank: 1,
    },
    {
      feature: 'pred_stage_data_loading_sec',
      display_name: 'Data Loading prediction',
      value: 1.95,
      contribution_seconds: 1.95,
      included_in_total: true,
      direction: 'component of the total',
      contribution_rank: 2,
    },
  ],
  features: [
    {
      feature: 'pred_stage_datamart_approval_sec',
      display_name: 'Datamart Approval estimate',
      value: 98.4,
      contribution_seconds: 98.4,
      included_in_total: true,
      direction: 'approval component (historical median statistic)',
      contribution_rank: 3,
    },
  ],
}

const PREDICTED_BODY = {
  state: 'PREDICTED',
  reason: null,
  // The five operational stage values in STAGE_INTEL sum to EXACTLY 18.16
  // (1.40 + 0.85 + 0.76 + 13.20 + 1.95) — the mock is arithmetically faithful
  // to the authoritative sum-of-applicable-stages serving policy.
  predicted_seconds: 18.16,
  predicted_seconds_human: '18s',
  predicted_log1p: 3.217,
  model_version: 'sum-of-five-stages-v1',
  model_status: 'DETERMINISTIC_POLICY',
  inputs: {},
  identity: {
    loader_name: 'LoaderA',
    sprint: 'S1',
    dataset: 'DS1',
    resolution_status: 'RESOLVED',
    resolved_connection: 'test1',
  },
  history: {
    prior_loader_executions: 11,
    eligibility_policy: { min_prior_loader_executions: 3 },
  },
  features: {},
  stage_intelligence: STAGE_INTEL,
  uncertainty: UNCERTAINTY,
  explanation: EXPLANATION,
  evidence_explanation: {
    available: true,
    evidence_version: 'evidence-v1',
    summary: 'Total 18s = the ML stage predictions + the statistical approval estimates above.',
    sentences: [
      'Staging: your loader ran this stage 10 time(s) with a typical time of 1.2s — this run is predicted at 1.4s, right in line with your usual.',
      'Datamart approval: estimated at 5s as the historical median of 7 matching past approval(s) (global historical median).',
    ],
    what_ifs: [
      'If this execution also runs prevalidation, add roughly 13s: enter its display name or count in the form and the total will include it.',
    ],
    ml_sentence_count: 1,
    approval_sentence_count: 1,
  },
  plain_explanation:
    // plain-explain-v2 (2026-10-06, what build_plain_explanation now returns):
    // the biggest driver vs the loader's own usual + the calibrated range.
    "Most of the expected time comes from the pre-load step, which this run's inputs make about 34% longer " +
      "than your loader's usual 9.8s. On similar past runs this model's " +
      "misses stayed within about 16s–21s, so the actual time should land close to the prediction.",
  risk: {
    label: 'WARNING',
    reasons: [
      {
        code: 'ETA_ABOVE_LOADER_HISTORY',
        message: 'The predicted duration is well above this loader\'s own historical median.',
        evidence: 'ETA is 4.1x the prior loader median (threshold 3x)',
      },
    ],
    evidence: {
      predicted_seconds: 18.155,
      prior_loader_executions: 11,
      config_range_excess: 0,
      eta_vs_prior_loader_median: 4.078,
    },
    policy_version: 'risk-hierarchy-v1',
  },
  warnings: [],
}

const INSUFFICIENT_BODY = {
  ...PREDICTED_BODY,
  state: 'INSUFFICIENT_HISTORY',
  reason: 'NO_DIRECT_HISTORY',
  predicted_seconds: null,
  predicted_seconds_human: null,
  predicted_log1p: null,
  model_version: null,
  model_status: null,
  stage_intelligence: null,
  uncertainty: null,
  explanation: null,
  loader_case: {
    case: 'BELOW_THRESHOLD',
    case_number: 3,
    prior_count: 2,
    threshold: 3,
    total_available: false,
  },
  identity: { loader_name: 'Loader_X', resolution_status: 'RESOLVED', resolved_connection: 'test1' },
  risk: {
    label: 'INSUFFICIENT_HISTORY',
    reasons: [
      {
        code: 'INSUFFICIENT_HISTORY',
        message:
          'No operational risk assessment is produced: the loader does not meet the minimum prior-history policy, so no ETA exists to assess.',
        evidence: 'AD-16 minimum prior-history not met',
      },
    ],
    evidence: {},
    policy_version: 'risk-hierarchy-v1',
  },
  history: {
    prior_loader_executions: 0,
    eligibility_policy: { min_prior_loader_executions: 3 },
  },
}

const UNRESOLVED_BODY = {
  ...PREDICTED_BODY,
  state: 'UNRESOLVED_IDENTITY',
  reason: 'no definition match for Dataset+Sprint+Loader',
  predicted_seconds: null,
  predicted_seconds_human: null,
  predicted_log1p: null,
  model_version: null,
  model_status: null,
  identity: { resolution_status: 'UNRESOLVED', resolved_connection: null },
  history: {},
}

const AMBIGUOUS_BODY = {
  ...PREDICTED_BODY,
  state: 'AMBIGUOUS_IDENTITY',
  reason: 'identity matches multiple definition connections; never silently chosen',
  predicted_seconds: null,
  predicted_seconds_human: null,
  predicted_log1p: null,
  model_version: null,
  model_status: null,
  identity: { resolution_status: 'AMBIGUOUS', resolved_connection: null },
  history: {},
}

const INVALID_BODY = {
  ...PREDICTED_BODY,
  state: 'INVALID_INPUT',
  reason:
    'provided_connection_disagrees_with_definition_resolution: "test2" != "test1"',
  predicted_seconds: null,
  predicted_seconds_human: null,
  predicted_log1p: null,
  model_version: null,
  model_status: null,
  history: {},
}

function mockFetchOnce(body: unknown, ok = true, status = 200) {
  return vi.fn().mockResolvedValueOnce({
    ok,
    status,
    statusText: status === 200 ? 'OK' : 'Error',
    json: async () => body,
  })
}

beforeEach(() => {
  vi.stubGlobal('fetch', vi.fn())
})

describe('Prediction Studio', () => {
  it('shows required-field validation without calling the API', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    render(<PredictionStudio />)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    expect(await screen.findByText('Loader Name is required.')).toBeInTheDocument()
    expect(screen.getByText('Sprint is required.')).toBeInTheDocument()
    expect(screen.getByText('Dataset is required.')).toBeInTheDocument()
    expect(screen.getByText('Connection is required.')).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
  })

  it('posts the exact pre-execution contract and renders PREDICTED', async () => {
    const user = userEvent.setup()
    const fetchMock = mockFetchOnce(PREDICTED_BODY)
    vi.stubGlobal('fetch', fetchMock)
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))

    await waitFor(() =>
      expect(screen.getByTestId('predicted-eta')).toHaveTextContent('18.16 seconds'),
    )
    // 2026-10-02: total in seconds (primary) + minutes equivalent below
    expect(screen.getByTestId('predicted-human')).toHaveTextContent('≈ 0.3 min')
    await user.click(screen.getByTestId('prediction-insights-toggle'))
    // ⓘ insights show ONLY the explanation (final spec §5, updated):
    // confidence + interval removed; no model identity, no provenance,
    // no technical metadata.
    expect(screen.queryByText(/sum-of-five-stages-v1/)).not.toBeInTheDocument()
    expect(screen.queryByText(/11 prior execution/i)).not.toBeInTheDocument()
    // identity echo: the entered connection stays in the (left-side) form input
    expect(screen.getByDisplayValue('test1')).toBeInTheDocument()

    // contract: only pre-execution fields posted — connection is now REQUIRED (Task 10.1)
    const call = fetchMock.mock.calls[0] as unknown as [string, RequestInit]
    expect(call[0]).toBe('/api/predict')
    const sent = JSON.parse(String(call[1].body))
    expect(sent.connection).toBe('test1')
    expect(Object.keys(sent).sort()).toEqual([
      'connection',
      'dataset',
      'loader_name',
      'sprint',
    ])
  })

  it('renders INSUFFICIENT_HISTORY without any ETA', async () => {
    vi.stubGlobal('fetch', mockFetchOnce(INSUFFICIENT_BODY))
    render(<PredictionStudio />)
    const user2 = userEvent.setup()
    await fillRequiredFields(user2)
    await user2.click(screen.getByRole('button', { name: /Predict/i }))

    await waitFor(() =>
      expect(screen.getByTestId('insufficient-history-note')).toBeInTheDocument(),
    )
    // 2026-10-02: short message only — no long explanation, no Reason block
    expect(screen.getByTestId('insufficient-history-note')).toHaveTextContent(
      /Insufficient history to predict: loader 'Loader_X' has only 2 prior execution\(s\); the trust threshold is 3/,
    )
    expect(screen.queryByText(/prediction is intentionally withheld/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Reason:/)).not.toBeInTheDocument()
    expect(screen.queryByTestId('predicted-eta')).not.toBeInTheDocument()
    // 2026-10-02: post-execution entry IS available for refused-history states
    expect(screen.getByTestId('actual-outcome-toggle')).toBeInTheDocument()
  })

  it('shows the short new-loader message and the post-execution button for UNKNOWN_LOADER', async () => {
    vi.stubGlobal('fetch', mockFetchOnce({
      ...INSUFFICIENT_BODY,
      loader_case: {
        case: 'UNKNOWN_LOADER',
        case_number: 2,
        prior_count: 0,
        threshold: 3,
        total_available: false,
      },
      history: { prior_loader_executions: 0, eligibility_policy: { min_prior_loader_executions: 3 } },
    }))
    render(<PredictionStudio />)
    const user2 = userEvent.setup()
    await fillRequiredFields(user2)
    await user2.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() =>
      expect(screen.getByTestId('insufficient-history-note')).toBeInTheDocument(),
    )
    // 2026-10-02: short new-loader message only
    expect(screen.getByTestId('insufficient-history-note')).toHaveTextContent(
      /New loader 'Loader_X' — it has no prior executions yet/,
    )
    expect(screen.queryByText(/Reason:/)).not.toBeInTheDocument()
    expect(screen.queryByText(/LOW-CONFIDENCE warm-up estimate/i)).not.toBeInTheDocument()
    // post-execution entry exists so history accumulates toward the threshold
    expect(screen.getByTestId('actual-outcome-toggle')).toBeInTheDocument()
  })

  it('posts per-component rows as entered (prevalidation + transformation orders)', async () => {
    const user = userEvent.setup()
    const fetchMock = mockFetchOnce(PREDICTED_BODY)
    vi.stubGlobal('fetch', fetchMock)
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    // row 1: a prevalidation component
    fireEvent.click(screen.getByTestId('component-add'))
    await user.type(screen.getByTestId('component-display-name-0'), 'Preval_1_employeeName')
    // row 2: a transformation component
    fireEvent.click(screen.getByTestId('component-add'))
    await user.type(screen.getByTestId('component-display-name-1'), 'tf')
    await user.selectOptions(screen.getByTestId('component-calling-type-1'), 'TRANSFORMATION')
    await user.type(screen.getByTestId('component-execution-order-1'), '1')
    // row 3: left empty -> must NOT be sent
    fireEvent.click(screen.getByTestId('component-add'))
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    const sent = JSON.parse(String((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body))
    // 2026-10-03: every row carries its component TYPE (default crossreference)
    // alongside display name, calling type and order.
    expect(sent.preval_trans_components).toEqual([
      { display_name: 'Preval_1_employeeName', calling_type: 'PREVALIDATION', type: 'crossreference', execution_order: null },
      { display_name: 'tf', calling_type: 'TRANSFORMATION', type: 'crossreference', execution_order: 1 },
    ])
  })

  it('shows live prevalidation/transformation counts BEFORE Predict and sends the chosen type', async () => {
    const user = userEvent.setup()
    const fetchMock = mockFetchOnce(PREDICTED_BODY)
    vi.stubGlobal('fetch', fetchMock)
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    // counts are absent while there are no rows
    expect(screen.queryByTestId('component-counts')).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId('component-add'))
    await user.type(screen.getByTestId('component-display-name-0'), 'Preval_1_employeeName')
    fireEvent.click(screen.getByTestId('component-add'))
    await user.type(screen.getByTestId('component-display-name-1'), 'tf')
    await user.selectOptions(screen.getByTestId('component-calling-type-1'), 'TRANSFORMATION')
    await user.selectOptions(screen.getByTestId('component-type-1'), 'query')
    // live counts appear BEFORE Predict — the user sees exactly what will be sent
    expect(screen.getByTestId('component-counts')).toHaveTextContent('1 prevalidation')
    expect(screen.getByTestId('component-counts')).toHaveTextContent('1 transformation')
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    const sent = JSON.parse(String((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body))
    expect(sent.preval_trans_components[1].type).toBe('query')
  })

  it('renders UNRESOLVED_IDENTITY without guessing', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(UNRESOLVED_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() =>
      expect(
        screen.getByRole('heading', { name: /could not be resolved/i }),
      ).toBeInTheDocument(),
    )
    expect(screen.queryByTestId('predicted-eta')).not.toBeInTheDocument()
    expect(screen.getByText(/could not be resolved against the registered loader definitions/i)).toBeInTheDocument()
  })

  it('renders AMBIGUOUS_IDENTITY without guessing', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(AMBIGUOUS_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() =>
      expect(screen.getByText(/refuses to silently choose/i)).toBeInTheDocument(),
    )
    expect(screen.queryByTestId('predicted-eta')).not.toBeInTheDocument()
    expect(screen.getByText(/matches more than one distinct definition connection/i)).toBeInTheDocument()
  })

  it('renders INVALID_INPUT from the backend (connection disagreement)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(INVALID_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() =>
      expect(screen.getByText(/disagrees_with_definition_resolution/i)).toBeInTheDocument(),
    )
    expect(screen.queryByTestId('predicted-eta')).not.toBeInTheDocument()
  })

  it('renders a network failure with retry, no fabricated result', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      vi.fn().mockRejectedValueOnce(new TypeError('Failed to fetch')),
    )
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(screen.getByText(/Cannot reach the prediction backend/i)).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /Retry prediction/i })).toBeInTheDocument()
    expect(screen.queryByTestId('predicted-eta')).not.toBeInTheDocument()
  })

  it('disables the button and shows loading during an active request', async () => {
    const user = userEvent.setup()
    let resolveFetch: (v: unknown) => void = () => {}
    vi.stubGlobal(
      'fetch',
      vi.fn().mockReturnValueOnce(
        new Promise((res) => {
          resolveFetch = res
        }),
      ),
    )
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    const button = screen.getByRole('button', { name: /Predict/i })
    await user.click(button)
    expect(screen.getByRole('button', { name: /Predicting…/i })).toBeDisabled()
    expect(screen.getByRole('status')).toBeInTheDocument()
    resolveFetch({
      ok: true,
      status: 200,
      json: async () => PREDICTED_BODY,
    })
    await waitFor(() =>
      expect(screen.getByRole('button', { name: /Predict Execution Time/i })).toBeEnabled(),
    )
  })

  it('exposes no historical or post-execution manual inputs', () => {
    render(<PredictionStudio />)
    // AD-31 contract: the four identity fields + the per-component
    // pre-execution rows. NO planned-start and NO workflow-ID input exist —
    // linkage happens at outcome recording, not at prediction time.
    expect(screen.getByLabelText(/Loader Name/)).toBeInTheDocument()
    expect(screen.getByLabelText(/^Sprint/)).toBeInTheDocument()
    expect(screen.getByLabelText(/^Dataset/)).toBeInTheDocument()
    expect(screen.getByLabelText(/^Connection/)).toBeInTheDocument()
    // 2026-10-02: per-component rows appear only after "Add Component"; there
    // is no scalar Type/Calling Type/Order input outside a component row
    expect(screen.queryByLabelText(/^Type$/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/^Calling Type$/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/^Display Name$/)).not.toBeInTheDocument()
    fireEvent.click(screen.getByTestId('component-add'))
    expect(screen.getByLabelText(/^Display Name$/)).toBeInTheDocument()
    expect(screen.getByLabelText(/^Calling Type$/)).toBeInTheDocument()
    expect(screen.getByLabelText(/^Order$/)).toBeInTheDocument()
    // no scalar legacy config inputs remain (superseded by component rows)
    expect(screen.queryByLabelText(/^Execution Order$/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/^Prevalidation Count$/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/^Transformation Count$/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/^Prevalidation Display Name$/)).not.toBeInTheDocument()
    expect(screen.queryByLabelText(/^Transformation Display Name$/)).not.toBeInTheDocument()
    // The boundary: no manual INPUT exists for any historical/derived/post-execution value.
    const forbiddenInputs = [
      /historical count/i,
      /prior mean/i,
      /prior median/i,
      /standard deviation/i,
      /previous duration/i,
      /time since previous/i,
      /similarity/i,
      /feature vector/i,
      /shap/i,
      /residual/i,
      /actual duration/i,
      /actual execution/i,
      /error count/i,
      /planned start/i,
      /workflow execution/i,
    ]
    for (const pattern of forbiddenInputs) {
      expect(screen.queryByLabelText(pattern)).not.toBeInTheDocument()
    }
  })

  // ------------------------------------------------ Task 06 + 10.1: stage intelligence

  it('renders the five-stage result with compact stage rows and Approval Timings (AD-31)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(PREDICTED_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))

    // 2026-10-04: the total-definition sentence was removed from the UI per
    // user spec — wait on the stage rows themselves (the exact arithmetic is
    // pinned by the backend controlled-arithmetic tests and the cases below).
    await waitFor(() =>
      expect(screen.getByTestId('stage-rows')).toBeInTheDocument(),
    )
    // stage rows render with the human-readable name (raw id on the testid)
    expect(screen.getByTestId('stage-name-staging')).toHaveTextContent('Staging')
    expect(screen.getByTestId('stage-name-data_loading')).toHaveTextContent('Data Loading')
    expect(screen.getByText('1.95 seconds')).toBeInTheDocument()
    // all five operational stages predict — including small-sample Pre-load
    expect(screen.getByTestId('stage-name-pre_load')).toHaveTextContent('Pre-load')
    expect(screen.getByText('13.20 seconds')).toBeInTheDocument()
    expect(screen.queryByTestId('stage-omitted-note')).not.toBeInTheDocument()
    // §4: no method/evidence badges in the normal result view
    expect(screen.queryByTestId('stage-tier-staging')).not.toBeInTheDocument()
    expect(screen.queryByTestId('stage-tier-pre_load')).not.toBeInTheDocument()
    // approvals render separately under "Approval Timings" as historical
    // median statistics — included in the total when available (mock carries
    // only the dataloading approval)
    expect(screen.queryByText('Datamart Approval')).not.toBeInTheDocument()
    expect(screen.getByText('Dataloading Approval')).toBeInTheDocument()
    // 2026-10-04: the 'Added to the total only…' badge sentence was removed
    // from the UI per user spec.
  })

  it('shows no Stage Intelligence section outside the PREDICTED state', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(INSUFFICIENT_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() =>
      expect(screen.getByTestId('insufficient-history-note')).toBeInTheDocument(),
    )
    expect(screen.queryByText(/Seven-Stage Execution View/)).not.toBeInTheDocument()
  })

  // --------------------------------------------------- Task 07: uncertainty

  it('renders the uncertainty range and reliability for PREDICTED (inside Prediction Insights)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(PREDICTED_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))

    await waitFor(() =>
      expect(screen.getByTestId('predicted-eta')).toHaveTextContent('18.16 seconds'),
    )
    // user clarification 2026-10-01: interval RESTORED as a visible block;
    // only confidence was removed. No range outside the panel.
    expect(screen.queryByTestId('uncertainty-range')).not.toBeInTheDocument()
    await user.click(screen.getByTestId('prediction-insights-toggle'))
    // the calibrated prediction interval renders as a dedicated block
    expect(screen.getByTestId('prediction-range')).toBeInTheDocument()
    expect(screen.getByText('Prediction range')).toBeInTheDocument()
    // 2026-10-05: confidence level removed from the label and the
    // calibration sentence removed from the block
    expect(screen.queryByText('Prediction range (90%)')).not.toBeInTheDocument()
    expect(screen.queryByText(/Calibrated band around/)).not.toBeInTheDocument()
    expect(screen.getByTestId('uncertainty-range')).toHaveTextContent(
      '1.06 seconds — 2m 57s',
    )
    // §5: no reliability label inside the insights panel
    expect(screen.queryByTestId('reliability-badge')).not.toBeInTheDocument()
    expect(screen.getAllByText(/Why this ETA\?/i).length).toBeGreaterThan(0)
    // confidence block is gone entirely
    expect(screen.queryByText(/^Confidence$/)).not.toBeInTheDocument()
    // §5: reliability evidence reasons are NOT rendered in the insights panel
    expect(
      screen.queryByText(/Interval is wide relative to the point estimate/i),
    ).not.toBeInTheDocument()
    // the point ETA still renders unchanged
    expect(screen.getByTestId('predicted-eta')).toHaveTextContent('18.16 seconds')
  })

  it('renders no fake range when uncertainty is null (non-PREDICTED)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(INSUFFICIENT_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() =>
      expect(screen.getByTestId('insufficient-history-note')).toBeInTheDocument(),
    )
    expect(screen.queryByTestId('reliability-badge')).not.toBeInTheDocument()
    expect(screen.queryByTestId('predicted-eta')).not.toBeInTheDocument()
    // no insights panel at all outside the PREDICTED state (no fake range)
    expect(screen.queryByTestId('prediction-insights-toggle')).not.toBeInTheDocument()
  })

  it('renders PREDICTED correctly when uncertainty is absent/null (graceful)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      mockFetchOnce({ ...PREDICTED_BODY, uncertainty: null }),
    )
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    // the calibrated-range caption degrades to its fallback text (panel opened)
    await user.click(screen.getByTestId('prediction-insights-toggle'))
    expect(screen.getByTestId('uncertainty-range')).toHaveTextContent(
      'Range not available for this prediction.',
    )
    // stage view still renders
    expect(screen.getByTestId('stage-rows')).toBeInTheDocument()
  })

  // -------------------------------------------------- Task 08: explainability
  it('keeps Prediction Insights minimal: no evidence/technical blocks (2026-10-05 spec)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(PREDICTED_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() =>
      expect(screen.getByTestId('predicted-eta')).toHaveTextContent('18.16 seconds'),
    )
    await user.click(screen.getByTestId('prediction-insights-toggle'))
    expect(screen.queryByTestId('evidence-explanation')).not.toBeInTheDocument()
    expect(screen.queryByTestId('evidence-sentences')).not.toBeInTheDocument()
    expect(screen.queryByTestId('evidence-what-ifs')).not.toBeInTheDocument()
    expect(screen.queryByText(/Show technical model contributions/i)).not.toBeInTheDocument()
  })

  it('renders the Why-this-ETA explanation for PREDICTED (inside Prediction Insights)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(PREDICTED_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    await user.click(screen.getByTestId('prediction-insights-toggle'))
    expect(screen.getByTestId('plain-explanation')).toBeInTheDocument()
    // 2026-10-05 minimal view: plain lines only, no per-feature bars
    expect(screen.queryByTestId('explanation-feature-1')).not.toBeInTheDocument()
  })

  it('renders the minimal Why-this-ETA lines for the sum policy (2026-10-05 spec)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(PREDICTED_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    await user.click(screen.getByTestId('prediction-insights-toggle'))
    expect(screen.getByTestId('plain-explanation')).toBeInTheDocument()
    // 2026-10-05: no per-feature bars, no technical collapsible
    expect(screen.queryByTestId('explanation-feature-1')).not.toBeInTheDocument()
    expect(screen.queryByText(/Show technical model contributions/i)).not.toBeInTheDocument()
  })

  it('controlled arithmetic: 10+20+30+40+50 renders total 150 (no hidden adjustment)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch',
      mockFetchOnce(bodyWithStageSum(
        { staging: 10, prevalidation: 20, transformation: 30, pre_load: 40, data_loading: 50 },
        { datamart_approval: 999, dataloading_approval: 777 },
      )),
    )
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    // the displayed total is EXACTLY the sum of the five displayed stages
    // (2026-10-02: seconds are primary — 150s -> '150.00 seconds' + '≈ 2.5 min')
    expect(screen.getByTestId('predicted-eta')).toHaveTextContent('150.00 seconds')
    expect(screen.getByTestId('predicted-human')).toHaveTextContent('≈ 2.5 min')
    // the huge approval estimates appear ONLY as contextual analytics
    // (formatSeconds renders >=60s values in clock units); the on-page
    // definition sentence was removed per user spec — the arithmetic itself
    // is pinned by the backend controlled-arithmetic tests.
    expect(screen.getByText('16m 39s')).toBeInTheDocument()
    expect(screen.getByText('12m 57s')).toBeInTheDocument()
  })

  it('controlled arithmetic delta: changing one stage changes the total by exactly that delta', async () => {
    // §35: Staging=10, Prevalidation=20, Transformation=30, Pre-load=40,
    // Data Loading=60 -> total MUST be 160 (not 150, not anything else);
    // only data_loading changed (+10 vs the 150 case) and ONLY the total
    // moved by exactly that amount.
    const user = userEvent.setup()
    vi.stubGlobal('fetch',
      mockFetchOnce(bodyWithStageSum(
        { staging: 10, prevalidation: 20, transformation: 30, pre_load: 40, data_loading: 60 },
        { datamart_approval: 999 },
      )),
    )
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    expect(screen.getByTestId('predicted-eta')).toHaveTextContent('160.00 seconds')
  })

  it('shows no explanation section for non-PREDICTED states', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(INSUFFICIENT_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() =>
      expect(screen.getByTestId('insufficient-history-note')).toBeInTheDocument(),
    )
    expect(screen.queryByTestId('explanation-section')).not.toBeInTheDocument()
  })

  it('renders PREDICTED gracefully when explanation is null (failure path)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      mockFetchOnce({ ...PREDICTED_BODY, explanation: null, warnings: ['Explanation unavailable: synthetic'] }),
    )
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    // stage-intelligence layers still generate the why-ETA summary
    // when the backend explanation payload is null
    expect(screen.getByTestId('predicted-eta')).toHaveTextContent('18.16')
    await user.click(screen.getByTestId('prediction-insights-toggle'))
    expect(screen.getByTestId('uncertainty-range')).toBeInTheDocument()
    // 2026-10-06: backend plain-explain-v2 (the essence) is preferred over
    // the sum-equation fallback when the payload carries it
    expect(screen.getByTestId('plain-explanation')).toHaveTextContent(
      'Most of the expected time comes from the pre-load step',
    )
  })

  // ------------------------------------------------------ Task 09: risk
  it('shows NO risk pop-up/toast for PREDICTED (2026-10-05: pop-ups fully removed)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(PREDICTED_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    // risk never pops up on the result: no toast, no permanent card, no label
    expect(screen.queryByTestId('transient-risk-toast')).not.toBeInTheDocument()
    expect(screen.queryByTestId('risk-section')).not.toBeInTheDocument()
    expect(screen.queryByTestId('risk-label')).not.toBeInTheDocument()
    // the short plain-language explanation shows in the insights panel instead
    await user.click(screen.getByTestId('prediction-insights-toggle'))
    expect(screen.getByTestId('plain-explanation')).toHaveTextContent(
      'Most of the expected time comes from the pre-load step',
    )
    // ETA stays intact
    expect(screen.getByTestId('predicted-eta')).toHaveTextContent('18.16')
  })

  // --------------------------------------------- Task 10.1: conditional risk

  it('does NOT render a prominent risk card for NORMAL risk (alert-only)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch',
      mockFetchOnce({
        ...PREDICTED_BODY,
        risk: { label: 'NORMAL', reasons: [], evidence: {}, policy_version: 'risk-hierarchy-v1' },
      }),
    )
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    // no risk section at all for NORMAL — risk behaves like an alert
    expect(screen.queryByTestId('risk-section')).not.toBeInTheDocument()
    expect(screen.queryByTestId('risk-label')).not.toBeInTheDocument()
    // everything else intact
    expect(screen.getByTestId('predicted-eta')).toHaveTextContent('18.16')
  })

  it('renders the backend plain-explain-v2 essence when the payload carries it (2026-10-06 spec)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(PREDICTED_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    await user.click(screen.getByTestId('prediction-insights-toggle'))
    await waitFor(() => expect(screen.getByTestId('plain-explanation')).toBeInTheDocument())
    const summary = screen.getByTestId('plain-explanation')
    // the essence: biggest driver vs the loader's own usual time (plain-explain-v2)
    expect(summary).toHaveTextContent('Most of the expected time comes from the pre-load step')
    expect(summary).toHaveTextContent('longer than your loader\'s usual 9.8s')
    // the calibrated range sentence comes from the uncertainty layer
    expect(summary).toHaveTextContent('On similar past runs')
    // 2026-10-05: technical collapsible fully removed
    expect(screen.queryByText(/Show technical model contributions/i)).not.toBeInTheDocument()
  })

  it('falls back to the sum-equation summary when the backend explanation is null', async () => {
    const body = { ...PREDICTED_BODY, plain_explanation: null }
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(body))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    await user.click(screen.getByTestId('prediction-insights-toggle'))
    await waitFor(() => expect(screen.getByTestId('plain-explanation')).toBeInTheDocument())
    const summary = screen.getByTestId('plain-explanation')
    // generated from the ACTUAL stage-intelligence components
    expect(summary).toHaveTextContent('Pre-load 13.20 seconds')
    expect(summary).toHaveTextContent('Data Loading 1.95 seconds')
    expect(summary).toHaveTextContent('Transformation 0.76 seconds')
    // the approval timing is disclosed as a historical-median exclusion
    expect(summary).toHaveTextContent(/Approval timings/i)
  })

  it('blocks submission when Connection is missing (no network request)', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.fn()
    vi.stubGlobal('fetch', fetchMock)
    render(<PredictionStudio />)
    // fill everything EXCEPT connection
    await user.type(screen.getByLabelText(/Loader Name/i), 'LoaderA')
    await user.type(screen.getByLabelText(/Sprint/i), 'S1')
    await user.type(screen.getByLabelText(/Dataset/i), 'DS1')
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    expect(screen.getByText('Connection is required.')).toBeInTheDocument()
    expect(fetchMock).not.toHaveBeenCalled()
    expect(screen.queryByTestId('predicted-eta')).not.toBeInTheDocument()
  })

  it('renders NORMAL risk without fabricated reasons (legacy body still safe)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch',
      mockFetchOnce({
        ...PREDICTED_BODY,
        risk: { label: 'NORMAL', reasons: [], evidence: {}, policy_version: 'risk-hierarchy-v1' },
      }),
    )
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    expect(screen.queryByTestId('risk-reasons')).not.toBeInTheDocument()
  })

  it('renders no risk pop-up for LOW_CONFIDENCE risk (2026-10-05 policy)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch',
      mockFetchOnce({
        ...PREDICTED_BODY,
        risk: {
          label: 'LOW_CONFIDENCE',
          reasons: [
            {
              // 2026-10-03 wording: the interval-width criterion replaced the
              // removed label — same signal, no reliability wording.
              code: 'WIDE_INTERVAL',
              message:
                'The calibrated prediction interval is very wide relative to the point estimate: the evidence behind the ETA range is weak.',
              evidence: 'interval width exceeds 0.472x the point estimate (p90-anchored criterion)',
            },
          ],
          evidence: { predicted_seconds: 18.155, prior_loader_executions: 4 },
          policy_version: 'risk-hierarchy-v1',
        },
      }),
    )
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    expect(screen.queryByTestId('transient-risk-toast')).not.toBeInTheDocument()
    expect(screen.queryByTestId('risk-label')).not.toBeInTheDocument()
  })

  it('shows only the opaque INSUFFICIENT_HISTORY risk note — no fake ETA/risk evidence', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce(INSUFFICIENT_BODY))
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() =>
      expect(screen.getByTestId('insufficient-history-note')).toBeInTheDocument(),
    )
    expect(screen.queryByTestId('risk-label')).not.toBeInTheDocument()
    expect(screen.queryByTestId('transient-risk-toast')).not.toBeInTheDocument()
    expect(screen.queryByTestId('predicted-eta')).not.toBeInTheDocument()
    expect(screen.queryByTestId('uncertainty-range')).not.toBeInTheDocument()
    expect(screen.queryByTestId('explanation-section')).not.toBeInTheDocument()
  })

  it('shows no fabricated risk for other non-prediction states', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch',
      mockFetchOnce({
        ...PREDICTED_BODY,
        state: 'UNRESOLVED_IDENTITY',
        risk: null,
        uncertainty: null,
        explanation: null,
        stage_intelligence: null,
      }),
    )
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() =>
      expect(screen.getByText('Identity could not be resolved')).toBeInTheDocument(),
    )
    expect(screen.queryByTestId('risk-section')).not.toBeInTheDocument()
  })

  it('renders PREDICTED gracefully when risk is null (failure path)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch',
      mockFetchOnce({ ...PREDICTED_BODY, risk: null, warnings: ['Risk assessment unavailable: synthetic'] }),
    )
    render(<PredictionStudio />)
    await fillRequiredFields(user)
    await user.click(screen.getByRole('button', { name: /Predict/i }))
    await waitFor(() => expect(screen.getByTestId('predicted-eta')).toBeInTheDocument())
    expect(screen.queryByTestId('risk-section')).not.toBeInTheDocument()
    // ETA + insights unaffected by risk absence
    expect(screen.getByTestId('predicted-eta')).toHaveTextContent('18.16')
    await user.click(screen.getByTestId('prediction-insights-toggle'))
    expect(screen.getByTestId('uncertainty-range')).toBeInTheDocument()
    expect(screen.getByTestId('plain-explanation')).toBeInTheDocument()
    expect(screen.getByText(/Risk assessment unavailable/)).toBeInTheDocument()
  })

  // ------------------------------------ §35 controlled total arithmetic

  function bodyWithStageSum(stages: Record<string, number>, approvals: Record<string, number> = {}) {
    const stageEntries = Object.entries(stages).map(([stage, estimate_seconds]) => ({
      stage,
      prior_positive_count: 3,
      evidence_count: 3,
      estimate_seconds,
      estimator: null,
      available: true,
      warnings: [],
      method: 'ml_stage_model',
      availability: 'ML_PREDICTED',
      availability_reason: 'Stage ML prediction (controlled arithmetic probe).',
      model_kind: 'probe',
      model_status: 'EXPERIMENTAL',
      n_training_observations: 100,
      prior_loader_positive_count: 1,
      duplicate_of: null,
    }))
    const approvalEntries = Object.entries(approvals).map(([stage, estimate_seconds]) => ({
      stage,
      prior_positive_count: 0,
      evidence_count: 5,
      estimate_seconds,
      estimator: 'contextual hierarchical median (approval-hier-v1)',
      available: true,
      warnings: [],
      method: 'contextual_hierarchical_statistical',
      availability: 'CONTEXTUAL_STATISTICAL',
      availability_reason: 'Contextual statistical estimate from strictly-prior executions.',
      evidence_tier: 'GLOBAL',
      evidence_tier_label: 'Global historical median',
      n_evidence: 5,
      prior_loader_positive_count: 0,
      duplicate_of: null,
    }))
    const total = Object.values(stages).reduce((a, b) => a + b, 0)
    // human-readable format faithful to the backend human_readable_duration
    const secs = Math.round(total)
    const m = Math.floor(secs / 60)
    const rs = secs % 60
    const human = m > 0 ? `${m}m ${rs}s` : `${rs}s`
    return {
      ...PREDICTED_BODY,
      stage_intelligence: {
        ...STAGE_INTEL,
        stages: [...stageEntries, ...approvalEntries],
      },
      predicted_seconds: total,
      predicted_seconds_human: human,
    }
  }

})
