import { render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import LoaderIntelligence from './LoaderIntelligence'
import type { LoaderDirectoryRow, LoaderIntelResponse } from '../api'

const STRONG: LoaderIntelResponse = {
  loader_name: 'MigrationSummarySubTableLoaderDataset',
  loader_intel_version: 'loader-intel-v1',
  known: true,
  known_in_definitions: true,
  evidence_level: 'STRONG',
  eta_eligible: true,
  eta_gate: {
    min_prior_executions: 3,
    source: 'AD-16',
    satisfied: true,
    explanation: '11 eligible executions satisfy the gate of 3.',
  },
  history: {
    eligible_execution_count: 11,
    first_execution: '2026-07-15 17:48:04.915',
    last_execution: '2026-08-13 14:41:43.809',
    show_typical: true,
    median_sec: 4.452,
    mean_sec: 8.048,
    cv: 0.883,
    p90_sec: 21.5,
    max_sec: 186240,
  },
  context: {
    datasets: ['Migration_Summary_Sub_Fact_Table_LDR_Datase_new'],
    connections: ['MDM_DM_TEST'],
    sprints: ['TEST'],
    connection_resolution: { RESOLVED: 11 },
  },
  definition_context: {
    matched: true,
    definition_rows: 3,
    datasets: ['Migration_Summary_Sub_Fact_Table_LDR_Datase_new'],
    sprints: ['TEST'],
    connections: ['MDM_DM_TEST'],
  },
  cold_start: {
    applicable: false,
    reason: 'loader already satisfies the ETA history gate',
    contextual_intelligence: null,
  },
  behavior: {
    n: 4,
    median_sec: 4.452,
    recent_median_sec: 3.9,
    series: [
      { start_time: '2026-08-01 10:00:00', seconds: 3.1, status: 'SUCCEED' },
      { start_time: '2026-08-02 10:00:00', seconds: 8.4, status: 'SUCCEED_WITH_WARNING' },
      { start_time: '2026-08-03 10:00:00', seconds: 2.2, status: 'SUCCEED' },
      { start_time: '2026-08-04 10:00:00', seconds: 12.0, status: 'KILLED' },
    ],
    distribution: {
      bins: [
        { label: '0-5s', count: 3 },
        { label: '5-15s', count: 1 },
      ],
      min_sec: 2.2,
      max_sec: 12.0,
    },
  },
  limitations: [],
}

const UNKNOWN: LoaderIntelResponse = {
  ...STRONG,
  loader_name: 'definitely_not_a_real_loader_xyz',
  known: false,
  known_in_definitions: false,
  evidence_level: 'NO_HISTORY',
  eta_eligible: false,
  behavior: undefined,
  history: { eligible_execution_count: 0, first_execution: null, last_execution: null, show_typical: false },
  limitations: ['Loader is unknown to the registered definitions and has no execution history.'],
}

const DIRECTORY: { loaders: LoaderDirectoryRow[] } = {
  loaders: [
    {
      loader_name: 'MigrationSummarySubTableLoaderDataset',
      executions: 11,
      succeed: 9,
      completed_with_warning: 1,
      failed: 1,
      median_duration_sec: 4.452,
      last_seen: '2026-08-13 14:41:43.809',
      sprints: ['TEST'],
      datasets: ['Migration_Summary_Sub_Fact_Table_LDR_Datase_new'],
      connections: ['MDM_DM_TEST'],
      prediction_count: 3,
      matched_feedback: { absolute_error_seconds: 1.2, absolute_percentage_error: 6.6 },
    },
  ],
}

/** Dispatch by URL: the directory listing is GET /loaders; the per-loader
 * intelligence is GET /loaders/<encoded name>. */
function stubBackend(intel: LoaderIntelResponse) {
  return vi.fn().mockImplementation((input: RequestInfo | URL) => {
    const url = String(input)
    if (url.endsWith('/loaders')) {
      return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(DIRECTORY) })
    }
    return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve(intel) })
  })
}

describe('Loader Intelligence (simple panel, 2026-10-03 spec)', () => {
  beforeEach(() => vi.restoreAllMocks())
  afterEach(() => vi.unstubAllGlobals())

  it('opens the SIMPLE panel after the filters when a loader is clicked', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', stubBackend(STRONG))
    render(<LoaderIntelligence />)
    // directory renders, then the user clicks a loader
    await waitFor(() => expect(screen.getByTestId('loader-directory')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: 'MigrationSummarySubTableLoaderDataset' }))
    await waitFor(() => expect(screen.getByTestId('loader-intel-card')).toBeInTheDocument())
    expect(screen.getByTestId('intel-loader-name')).toHaveTextContent('MigrationSummarySubTableLoaderDataset')

    // simple intelligence ONLY: counts, typical/recent, range, trend
    expect(screen.getByTestId('intel-exec-count')).toHaveTextContent('11')
    expect(screen.getByTestId('intel-success-count')).toHaveTextContent('2')
    expect(screen.getByTestId('intel-warn-count')).toHaveTextContent('1')
    expect(screen.getByTestId('intel-killed-count')).toHaveTextContent('1')
    expect(screen.getByTestId('intel-typical')).toHaveTextContent('4.45 seconds')
    expect(screen.getByTestId('intel-recent')).toHaveTextContent('3.90 seconds')
    expect(screen.getByTestId('intel-range')).toHaveTextContent('2.20 seconds')
    expect(screen.getByText(/Execution time trend/i)).toBeInTheDocument()
  })

  it('shows NO tabs, stage intelligence, configuration or cold-start sections', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', stubBackend(STRONG))
    render(<LoaderIntelligence />)
    await waitFor(() => expect(screen.getByTestId('loader-directory')).toBeInTheDocument())
    await user.click(screen.getByRole('button', { name: 'MigrationSummarySubTableLoaderDataset' }))
    await waitFor(() => expect(screen.getByTestId('loader-intel-card')).toBeInTheDocument())
    // the old tabbed workspace is gone
    expect(screen.queryByRole('tab')).not.toBeInTheDocument()
    expect(screen.queryByRole('tablist')).not.toBeInTheDocument()
    expect(screen.queryByTestId('stage-intel-rows')).not.toBeInTheDocument()
    expect(screen.queryByText(/Cold-start context/i)).not.toBeInTheDocument()
    expect(screen.queryByText(/Loader configuration/i)).not.toBeInTheDocument()
  })

  it('renders the directory WITHOUT the Intelligence column but keeps the filter', async () => {
    vi.stubGlobal('fetch', stubBackend(STRONG))
    render(<LoaderIntelligence />)
    await waitFor(() => expect(screen.getByTestId('loader-directory')).toBeInTheDocument())
    const header = screen.getByTestId('loader-directory')
    expect(header).toHaveTextContent('Typical duration')
    expect(header).not.toHaveTextContent('Intelligence')
    // the intelligence FILTER stays available under More Filters
    await userEvent.setup().click(screen.getByTestId('more-filters-toggle'))
    expect(screen.getByLabelText(/Intelligence filter/i)).toBeInTheDocument()
  })

  it('handles an unknown loader honestly (no fabricated history)', async () => {
    const user = userEvent.setup()
    vi.stubGlobal('fetch', stubBackend(UNKNOWN))
    render(<LoaderIntelligence />)
    await user.type(screen.getByLabelText('Loader name'), 'definitely_not_a_real_loader_xyz')
    await user.click(screen.getByRole('button', { name: /Look up loader/i }))
    await waitFor(() => expect(screen.getByTestId('loader-intel-card')).toBeInTheDocument())
    expect(screen.getByTestId('intel-exec-count')).toHaveTextContent('0')
    expect(screen.getByTestId('intel-eta-eligible')).toHaveTextContent(/Insufficient historical evidence/i)
    expect(screen.getByText(/No recorded execution durations for this loader yet\./i)).toBeInTheDocument()
  })

  it('shows a backend error without crashing', async () => {
    const user = userEvent.setup()
    vi.stubGlobal(
      'fetch',
      vi.fn().mockImplementation((input: RequestInfo | URL) => {
        const url = String(input)
        if (url.endsWith('/loaders')) {
          return Promise.resolve({ ok: true, status: 200, json: () => Promise.resolve({ loaders: [] }) })
        }
        return Promise.resolve({ ok: false, status: 500, json: () => Promise.resolve({ detail: 'boom' }) })
      }),
    )
    render(<LoaderIntelligence />)
    await user.type(screen.getByLabelText('Loader name'), 'AnyLoader')
    await user.click(screen.getByRole('button', { name: /Look up loader/i }))
    await waitFor(() => expect(screen.getByRole('alert')).toBeInTheDocument())
    expect(screen.getByRole('alert')).toHaveTextContent('boom')
  })
})
