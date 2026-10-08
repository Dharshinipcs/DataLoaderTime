import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, afterEach } from 'vitest'
import PredictionHistory from './PredictionHistory'
import type { PredictionEventRecord } from '../api'

function mockFetchOnce(body: unknown, ok = true, status = 200) {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    json: () => Promise.resolve(body),
  })
}

const EVENT: PredictionEventRecord = {
  event_id: 'evt-1',
  schema_version: 't11-v1',
  created_at: '2026-09-21T12:00:00Z',
  event_state: 'PREDICTED',
  request: { loader_name: 'LoaderA', dataset: 'DS', sprint: 'S1', connection: 'C1' },
  identity: {
    loader_name: 'LoaderA',
    dataset: 'DS',
    sprint: 'S1',
    ldr_workflow_execution_id: 'WF-1',
  },    prediction: {
    predicted_total_seconds: 18.155,
    model_version: 'sum-of-available-stages-v2',
    model_status: 'CANDIDATE_EVALUATED',
    prior_count: 11,
    interval_lower_seconds: 1.061,
    interval_upper_seconds: 176.99,
    interval_level: 0.9,
    risk_label: 'WARNING',
    risk_policy_version: 'risk-hierarchy-v1',
  },
  explanation_digest: null,
  stage_intelligence: null,
  features: null,
  warnings: [],
}

const WITHHELD: PredictionEventRecord = {
  ...EVENT,
  event_id: 'evt-2',
  event_state: 'INSUFFICIENT_HISTORY',
  prediction: null,
}

afterEach(() => vi.unstubAllGlobals())

describe('Prediction History', () => {
  it('renders real prediction events with lifecycle states and intervals', async () => {
    vi.stubGlobal('fetch', mockFetchOnce({ count: 2, store_warnings: [], events: [EVENT, WITHHELD] }))
    render(<PredictionHistory />)
    await waitFor(() => expect(screen.getByTestId('history-table')).toBeInTheDocument())
    expect(screen.getByTestId('history-table')).toHaveTextContent('Predictions')
    // 2026-10-03: Status and Risk are NOT table columns (they live in the
    // filters); the withheld event simply shows no fabricated prediction.
    expect(screen.queryByRole('columnheader', { name: 'Status' })).not.toBeInTheDocument()
    expect(screen.queryByRole('columnheader', { name: 'Risk' })).not.toBeInTheDocument()
    expect(screen.getAllByText('—').length).toBeGreaterThan(0)
  })

  it('does not expose raw event/workflow ids in the table', async () => {
    vi.stubGlobal('fetch', mockFetchOnce({ count: 1, store_warnings: [], events: [EVENT] }))
    render(<PredictionHistory />)
    await waitFor(() => expect(screen.getByTestId('history-table')).toBeInTheDocument())
    expect(screen.queryByText('WF-1')).not.toBeInTheDocument()
    expect(screen.queryByText('evt-1')).not.toBeInTheDocument()
  })

  it('filters by status via Apply', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce({ count: 2, store_warnings: [], events: [EVENT, WITHHELD] }))
    render(<PredictionHistory />)
    await waitFor(() => expect(screen.getByTestId('history-table')).toBeInTheDocument())
    await user.click(screen.getByTestId('more-filters-toggle'))
    await user.selectOptions(screen.getByLabelText(/Prediction status filter/i), 'INSUFFICIENT_HISTORY')
    await user.click(screen.getByRole('button', { name: /Apply filters/i }))
    expect(screen.getByTestId('history-table')).toHaveTextContent('Predictions')
    expect(screen.getByTestId('history-table')).not.toHaveTextContent('18.16')
  })

  it('shows the live accuracy counts, filters chips and the event detail drawer', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', mockFetchOnce({ count: 2, store_warnings: [], events: [EVENT, WITHHELD] }))
    render(<PredictionHistory />)
    await waitFor(() => expect(screen.getByTestId('history-table')).toBeInTheDocument())
    // live accuracy counts over the filtered population (2026-10-03 layout:
    // LEFT column = graph + accuracy cards, RIGHT column = the list)
    expect(screen.getByTestId('history-summary')).toHaveTextContent('Attempts')
    expect(screen.getByTestId('history-summary')).toHaveTextContent('2')
    expect(screen.getByTestId('history-chart')).toBeInTheDocument()
    expect(screen.getByTestId('history-accuracy')).toBeInTheDocument()
    // row click opens the detail drawer — interval + confidence removed per
    // final spec; the served total and model version show instead
    await user.click(screen.getAllByTestId('history-row')[0])
    expect(screen.getByTestId('event-detail')).toBeInTheDocument()
    expect(screen.queryByTestId('detail-interval')).not.toBeInTheDocument()
    expect(screen.getByText(/Model: sum-of-available-stages-v2/i)).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: /close details/i }))
    expect(screen.queryByTestId('event-detail')).not.toBeInTheDocument()
    // the unavailable event's drawer shows the user-friendly reason
    await user.click(screen.getAllByTestId('history-row')[1])
    expect(screen.getByTestId('detail-unavailable-reason')).toHaveTextContent(/Insufficient historical evidence/)
  })

  it('renders the empty state when no events exist', async () => {
    vi.stubGlobal('fetch', mockFetchOnce({ count: 0, store_warnings: [], events: [] }))
    render(<PredictionHistory />)
    await waitFor(() =>
      expect(screen.getByText(/No predictions match these filters/i)).toBeInTheDocument(),
    )
  })
})
