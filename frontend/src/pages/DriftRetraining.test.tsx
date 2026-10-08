/**
 * Drift & Retraining tests — 2026-10-06 decision-oriented left side.
 *
 * Layout: LEFT = System Recommendation (Continue Monitoring / Retrain Now,
 * naming which stage models need retraining and why) + the stage-model error
 * comparison table (Baseline | Current serving (live, post-promotion) |
 * Previous). RIGHT unchanged: retraining decision + model versions + combined
 * history timeline. No approver-name input — the API layer records
 * 'local-operator'; the human supplies only the reason.
 *
 * All fetch calls are mocked with the actual backend response shapes.
 * No network access, no real backend.
 */
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import DriftRetraining from '../pages/DriftRetraining'

function overviewFixture(overrides: Record<string, unknown> = {}) {
  return {
    overview_version: 'drift-overview-v1',
    scope: { kind: 'system', value: null },
    headline: {
      data_health: 'STABLE',
      model_health: 'INSUFFICIENT_ACTUAL_OUTCOMES',
      new_eligible_outcomes: 4,
      action: 'NO_RETRAINING_NEEDED',
    },
    assessment: {
      verdict: 'NO_RETRAINING_NEEDED',
      tone: 'healthy',
      narrative: 'The system is currently healthy. Inputs are stable and the model is not yet assessable.',
      evidence: [
        { id: 'new_eligible_outcomes', label: '4 new eligible execution outcomes since the current model', met: true, detail: 'Threshold for review-readiness: 30.' },
        { id: 'data_drift', label: 'Input distributions broadly stable', met: false, detail: 'Compared over 300 recent vs 256 reference rows.' },
        { id: 'prediction_drift', label: 'Predicted time ranges consistent with the reference period', met: false, detail: 'One signal among several.' },
        { id: 'model_performance', label: 'Not enough matched actual outcomes to assess error', met: false, detail: 'Error compared against the chronological baseline.' },
        { id: 'stage_degradation', label: 'Stage models need more matched outcomes before assessment', met: false, detail: 'Stage health needs linked outcomes.' },
      ],
    },
    data_drift: {
      overall_status: 'NO_MEANINGFUL_DRIFT',
      feature_rows: [
        { feature: 'Execution Order', status: 'NO_MEANINGFUL_DRIFT', statistic: 0.08, method: 'KS', monitored_inputs: 1 },
        { feature: 'Historical duration context', status: 'WARNING', statistic: 0.17, method: 'KS', monitored_inputs: 4 },
      ],
      population_dimensions: [{ dimension: 'Loader', status: 'NO_MEANINGFUL_DRIFT', statistic: 0.05 }],
      reference_window: { first_created: '2025-10-10', last_created: '2026-07-15' },
    },
    prediction_drift: {
      reference_predictions: { count: 3, min_seconds: 8, max_seconds: 25 },
      recent_predictions: { count: 2, min_seconds: 15, max_seconds: 40 },
      statistic: 0.2,
      method: 'KS',
      status: 'WARNING',
      note: 'A change can simply reflect different workloads.',
    },
    model_performance: {
      overall: {
        status: 'INSUFFICIENT_OUTCOMES',
        sample_count: 4,
        min_required: 20,
        has_actual_outcomes: true,
        median_absolute_error_seconds: 8.45,
        note: 'Operational error summary.',
      },
      stage_health: {
        stages: {
          staging: { stage: 'Staging', status: 'INSUFFICIENT_EVIDENCE', recent_outcomes: 1, min_required: 20, note: 'Not enough linked actual outcomes for this stage yet.' },
          prevalidation: { stage: 'Prevalidation', status: 'INSUFFICIENT_EVIDENCE', recent_outcomes: 0, min_required: 20, note: 'Not enough linked actual outcomes.' },
          transformation: { stage: 'Transformation', status: 'INSUFFICIENT_EVIDENCE', recent_outcomes: 0, min_required: 20, note: 'Not enough linked actual outcomes.' },
          pre_load: { stage: 'Pre-load', status: 'DEGRADATION_DETECTED', recent_outcomes: 20, min_required: 20, baseline_medae_seconds: 12.155, recent_medae_seconds: 25.3, note: 'Recent median absolute stage error compared against the baseline.' },
          data_loading: { stage: 'Data Loading', status: 'INSUFFICIENT_EVIDENCE', recent_outcomes: 1, min_required: 20, note: 'Not enough linked actual outcomes.' },
        },
        overall_status: 'DEGRADATION_DETECTED',
        outcomes_with_stage_errors: 1,
        note: 'Stage health compares recent linked-actual stage errors with each stage baseline.',
      },
    },
    retraining: {
      readiness_state: 'NOT_READY',
      human_gate_required: true,
      automatic_retraining: 'OFF',
      signals: {
        eligible_new_actuals: { count: 4, threshold: 30, met: false },
        performance_degradation: { assessed: false, met: false },
        feature_drift: { status: 'NO_MEANINGFUL_DRIFT' },
        population_drift: { status: 'NO_MEANINGFUL_DRIFT' },
      },
    },
    notes: [
      'Drift means a material, persistent distribution change — new executions alone never trigger it.',
      'No training is launched automatically; a human approves any retraining or model promotion.',
    ],
    ...overrides,
  }
}

const MODELS = {
  models: [
    { model_id: 'stage-aware-total-ad34', lifecycle_state: 'PROMOTED', created_at: '2026-09-23T19:02:37Z' },
    { model_id: 'total-et-v3-candidate', lifecycle_state: 'RETIRED', created_at: '2026-09-23T09:21:23Z' },
  ],
  promoted_model_id: 'stage-aware-total-ad34',
  lifecycle_states: [],
}

const STATUS = {
  serving_model_version: 'sum-of-five-stages-v1',
  serving_is_promoted: true,
  serving_role: 'AUTHORITATIVE_TOTAL',
  promoted_model: null,
  note: 'ok',
}

const LIFECYCLE = {
  events: [
    { at: '2026-09-23T19:02:37Z', model_id: 'stage-aware-total-ad34', event_type: 'PROMOTION_APPROVED', kind: 'promoted', label: "Model 'stage-aware-total-ad34' approved and promoted" },
    { at: '2026-09-23T09:21:23Z', model_id: 'total-et-v3-candidate', event_type: 'REGISTERED', kind: 'registered', label: "Model 'total-et-v3-candidate' registered" },
  ],
}

function mockFetch(overview: Record<string, unknown> = overviewFixture()) {
  return vi.spyOn(globalThis, 'fetch').mockImplementation(async (input: RequestInfo | URL) => {
    const url = String(input)
    const json = (body: unknown) =>
      new Response(JSON.stringify(body), { status: 200, headers: { 'Content-Type': 'application/json' } })
    if (url.includes('/drift/overview')) return json(overview)
    if (url.includes('/models/lifecycle')) return json(LIFECYCLE)
    if (url.includes('/models/status')) return json(STATUS)
    if (url.includes('/models')) return json(MODELS)
    if (url.includes('/retraining/decisions')) return json({ decisions: DECISIONS })
    if (url.includes('/retraining/serving-status'))
      return json({ serving_version: 'sum-of-five-stages-v1 (stage-serving-v2)', candidate_override_active: false, candidate_version: null })
    if (url.includes('/retraining/train-candidate')) return json(TRAIN_CANDIDATE_RESPONSE)
    if (url.includes('/retraining/decide')) return json({ decision: DECISIONS[0] })
    return new Response('not found', { status: 404 })
  })
}

const DECISIONS = [
  {
    decision_id: 'dec-1',
    decision_type: 'CONTINUE_MONITORING',
    decided_by: 'local-operator',
    reason: 'Drift without degradation yet',
    decided_at: '2026-09-29T10:00:00Z',
    model_version_at_decision: 'sum-of-five-stages-v1',
    assessment_snapshot: { verdict: 'CONTINUE_MONITORING' },
  },
]

const TRAIN_CANDIDATE_RESPONSE = {
  registered: { model_id: 'stage-serving-candidate-20260929', status: 'CANDIDATE_EVALUATED' },
  evaluation: {
    candidate_version: 'stage-serving-candidate-20260929',
    verdict: 'ACCEPTABLE_FOR_REVIEW',
    rationale: 'Candidate holdout MedAE is 0.9x the served baseline. A human decides promotion.',
    per_stage: [
      { stage: 'staging', candidate_median_ae: 0.4, baseline_median_ae: 0.5, ratio: 0.8, candidate_n: 20 },
      { stage: 'data_loading', candidate_median_ae: 0.7, baseline_median_ae: 0.889, ratio: 0.787, candidate_n: 20 },
    ],
    total: { candidate_median_ae: 1.6, baseline_median_ae: 1.823, ratio: 0.878, candidate_n: 85 },
    human_gate_required: true,
    automatic_promotion: 'OFF',
  },
  serving_untouched: true,
  serving_version_before: 'sum-of-five-stages-v1 (stage-serving-v2)',
}

describe('Drift & Retraining (2026-10-04 compact layout)', () => {
  let fetchMock: ReturnType<typeof vi.spyOn>

  beforeEach(() => {
    vi.restoreAllMocks()
    fetchMock = mockFetch()
  })

  it('posts a continue-monitoring decision with reason only (no name input) and shows it in the timeline', async () => {
    const user = userEvent.setup()
    render(<DriftRetraining />)
    await waitFor(() => expect(screen.getByTestId('retraining-actions')).toBeInTheDocument())
    expect(screen.queryByLabelText('Decider')).not.toBeInTheDocument()
    await user.type(screen.getByLabelText('Decision reason'), 'no drift observed')
    await user.click(screen.getByTestId('continue-monitoring-button'))
    await waitFor(() => expect(screen.getByTestId('retraining-notice')).toHaveTextContent('continue monitoring'))
    const posted = JSON.parse((fetchMock.mock.calls.find((c: unknown[]) => String(c[0]).includes('/retraining/decide'))?.[1] as RequestInit).body as string)
    expect(posted.decision_type).toBe('CONTINUE_MONITORING')
    expect(posted.decided_by).toBe('local-operator')
  })

  it('retrain-now trains a candidate, shows evaluation table, production unchanged', async () => {
    const user = userEvent.setup()
    render(<DriftRetraining />)
    await waitFor(() => expect(screen.getByTestId('retraining-actions')).toBeInTheDocument())
    await user.click(screen.getByTestId('retrain-now-button'))
    await waitFor(() => expect(screen.getByTestId('candidate-evaluation')).toBeInTheDocument())
    expect(screen.getByTestId('candidate-verdict')).toHaveTextContent('Candidate ready for human review')
    expect(screen.getByTestId('candidate-stage-table')).toHaveTextContent('Total (sum of stages)')
    expect(screen.getByTestId('candidate-evaluation')).toHaveTextContent('Production model unchanged')
    // 2026-10-06 rework: the LEFT table no longer carries a candidate column —
    // candidates are evaluated on the right; production is unchanged.
    expect(screen.queryByTestId('stage-row-total')).not.toBeInTheDocument()
    const posted = JSON.parse((fetchMock.mock.calls.find((c: unknown[]) => String(c[0]).includes('/retraining/train-candidate'))?.[1] as RequestInit).body as string)
    expect(posted.approved_by).toBe('local-operator')
  })

  it('top recommendation: Continue Monitoring with the no-retraining reason when evidence is clean', async () => {
    const clean = overviewFixture()
    const stages = (clean.model_performance as Record<string, unknown>).stage_health as { stages: Record<string, { status: string }> }
    stages.stages.pre_load.status = 'INSUFFICIENT_EVIDENCE'
    vi.restoreAllMocks()
    fetchMock = mockFetch(clean)
    render(<DriftRetraining />)
    await waitFor(() => expect(screen.getByTestId('recommendation-badge')).toHaveTextContent('Continue Monitoring'))
    expect(screen.getByTestId('recommendation-headline')).toHaveTextContent(
      'Continue Monitoring — no stage model currently requires retraining.',
    )
    expect(screen.getByTestId('recommendation-reasons')).toHaveTextContent(
      'Neither full retraining nor individual stage retraining is currently required',
    )
  })

  it('top recommendation: names the exact stage models that require retraining', async () => {
    const user = userEvent.setup()
    render(<DriftRetraining />)
    // The fixture carries pre_load = DEGRADATION_DETECTED while the aggregate
    // verdict stays NO_RETRAINING_NEEDED (per-stage evidence wins) — the
    // recommendation must name that stage and offer per-stage retraining.
    await waitFor(() => expect(screen.getByTestId('recommendation-badge')).toHaveTextContent('Retrain Now'))
    const headline = screen.getByTestId('recommendation-headline')
    expect(headline).toHaveTextContent('Retraining Recommended')
    expect(headline).toHaveTextContent('Pre-load')
    expect(screen.getByTestId('recommendation-reasons')).toHaveTextContent(
      'Individual stage retraining is currently required for: Pre-load.',
    )
    // per-stage action becomes available for that stage
    await user.click(screen.getByTestId('retrain-stage-pre_load'))
    await waitFor(() => expect(screen.getByTestId('stage-notice')).toHaveTextContent(/pre.?load/i))
    const posted = JSON.parse(
      (fetchMock.mock.calls.find((c: unknown[]) => String(c[0]).includes('/retraining/train-candidate'))?.[1] as RequestInit).body as string,
    )
    expect(posted.stages).toEqual(['pre_load'])
  })

  it('stage model error comparison table: 5 stages x (baseline | current serving | previous)', async () => {
    render(<DriftRetraining />)
    await waitFor(() => expect(screen.getByTestId('stage-comparison')).toBeInTheDocument())
    for (const key of ['staging', 'prevalidation', 'transformation', 'pre_load', 'data_loading']) {
      expect(screen.getByTestId(`stage-row-${key}`)).toBeInTheDocument()
    }
    // baseline + live current-serving cell for the degraded stage
    expect(screen.getByTestId('stage-row-pre_load')).toHaveTextContent('12.15 seconds')
    expect(screen.getByTestId('current-serving-pre_load')).toHaveTextContent('—')
    // previous column exists and stays '—' with no lineage history
    expect(screen.getByTestId('previous-pre_load')).toHaveTextContent('—')
    // no after-retraining / linked-outcomes / health columns anymore
    expect(screen.queryByText(/After retraining/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Linked outcomes/i)).not.toBeInTheDocument()
    // drill-down detail still available per stage
    await userEvent.setup().click(screen.getByRole('button', { name: 'Pre-load' }))
    expect(screen.getByTestId('stage-detail')).toHaveTextContent(/Since promotion|No matched outcomes recorded since/)
  })

  it('removed sections are gone from the left side (2026-10-06 rework)', async () => {
    render(<DriftRetraining />)
    await waitFor(() => expect(screen.getByTestId('system-recommendation')).toBeInTheDocument())
    // data-drift / prediction-drift / statistics cards no longer on the page
    expect(screen.queryByTestId('drift-feature-table')).not.toBeInTheDocument()
    expect(screen.queryByTestId('pred-drift-status')).not.toBeInTheDocument()
    expect(screen.queryByTestId('median-ae')).not.toBeInTheDocument()
    expect(screen.queryByTestId('matched-outcomes')).not.toBeInTheDocument()
    expect(screen.queryByTestId('statistical-detail')).not.toBeInTheDocument()
    // right side stays intact
    expect(screen.getByTestId('retraining-actions')).toBeInTheDocument()
    expect(screen.getByTestId('model-versions')).toBeInTheDocument()
    expect(screen.getByTestId('lifecycle-timeline')).toBeInTheDocument()
  })

  it('retraining signals keep the human gate explicit', async () => {
    render(<DriftRetraining />)
    await waitFor(() => expect(screen.getByTestId('retraining-signals')).toBeInTheDocument())
    expect(screen.getByTestId('retraining-signals')).toHaveTextContent('4 of 30 needed')
  })

  it('model versions + combined timeline render promotions, decisions, timestamps', async () => {
    render(<DriftRetraining />)
    await waitFor(() => expect(screen.getByTestId('model-versions')).toBeInTheDocument())
    expect(screen.getByText('stage-aware-total-ad34')).toBeInTheDocument()
    expect(screen.getByTestId('serving-model-version')).toHaveTextContent('sum-of-five-stages-v1')
    expect(screen.getByTestId('lifecycle-timeline')).toHaveTextContent('approved and promoted')
    expect(screen.getByTestId('lifecycle-timeline')).toHaveTextContent('Continue monitoring')
    expect(screen.queryByText(/\.pkl/)).not.toBeInTheDocument()
  })

  it('governance actions post promote/rollback without an approver-name input', async () => {
    const user = userEvent.setup()
    render(<DriftRetraining />)
    await waitFor(() => expect(screen.getByTestId('governance-actions')).toBeInTheDocument())
    expect(screen.queryByLabelText('Approver')).not.toBeInTheDocument()
    await user.type(screen.getByLabelText('Model to promote'), 'stage-aware-total-ad35')
    await user.type(screen.getByLabelText('Reason'), 'better holdout error')
    await user.click(screen.getByTestId('promote-button'))
    await waitFor(() => expect(screen.getByTestId('governance-notice')).toHaveTextContent('Promoted'))
    const promoteCall = fetchMock.mock.calls.find(
      (c: unknown[]) => String(c[0]).includes('/models/promote'),
    )
    expect(promoteCall).toBeTruthy()
    const [, init] = promoteCall as [RequestInfo | URL, RequestInit]
    const body = JSON.parse(String(init.body))
    expect(body.approved_by).toBe('local-operator')
    expect(body.reason).toBe('better holdout error')
  })

  it('system scope only: overview queried without a loader param, no scope selector', async () => {
    render(<DriftRetraining />)
    await waitFor(() => expect(screen.getByTestId('system-recommendation')).toBeInTheDocument())
    const overviewUrls = fetchMock.mock.calls
      .map((c: unknown[]) => String(c[0]))
      .filter((u: string) => u.includes('/drift/overview'))
    expect(overviewUrls.length).toBeGreaterThan(0)
    expect(overviewUrls[0]).not.toContain('loader=')
    expect(screen.queryByTestId('drift-scope')).not.toBeInTheDocument()
  })
})
