import { fireEvent, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi, afterEach } from 'vitest'
import ActualOutcomeForm from './ActualOutcomeForm'

function mockFetchOnce(body: unknown, ok = true, status = 200) {
  return vi.fn().mockResolvedValue({
    ok,
    status,
    json: () => Promise.resolve(body),
  })
}

const OK_RESPONSE = {
  duplicate: false,
  linkage_status: 'MATCHED',
  linkage_method: 'EXACT_WORKFLOW_EXECUTION_ID',
  training_eligible: true,
  training_eligibility_reason: null,
  total_error: {
    signed_error_seconds: 0.5,
    absolute_error_seconds: 0.5,
    absolute_percentage_error: 2.7,
  },
  stage_errors: {},
  warnings: [],
}

afterEach(() => vi.unstubAllGlobals())

describe('ActualOutcomeForm (2026-10-03 contract: fixed datetime format, seconds-only input)', () => {
  it('submits Start/End in the fixed YYYY-MM-DD HH:MM:SS form and converts seconds to the record minutes', { timeout: 20000 }, async () => {
    const user = userEvent.setup()
    const fetchMock = mockFetchOnce(OK_RESPONSE)
    vi.stubGlobal('fetch', fetchMock)
    render(<ActualOutcomeForm />)

    await user.type(screen.getByTestId('out-loader'), 'LoaderA')
    await user.type(screen.getByTestId('out-dataset'), 'DS')
    await user.type(screen.getByTestId('out-sprint'), 'S1')
    await user.type(screen.getByTestId('out-connection'), 'C1')

    // datetime-local inputs with a seconds step — the value is the picker's
    // canonical T-form; the SUBMITTED body must carry the fixed space form.
    fireEvent.change(screen.getByTestId('out-start'), { target: { value: '2026-07-31T12:59:00' } })
    fireEvent.change(screen.getByTestId('out-end'), { target: { value: '2026-07-31T12:59:18' } })

    // the derived-duration helper the user asked to keep (now with seconds)
    expect(screen.getByTestId('derived-duration-note')).toHaveTextContent(
      'Duration derived from Start/End: 0.30 min (18 seconds).',
    )

    // ALL timings entered in SECONDS: total 18 s and one stage 120 s
    await user.type(screen.getByTestId('out-total'), '18')
    await user.type(screen.getByLabelText('Staging (sec)'), '120')

    await user.click(screen.getByTestId('outcome-submit'))

    await waitFor(() => expect(screen.getByTestId('outcome-done')).toBeInTheDocument())
    const body = JSON.parse(String((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body))
    // fixed format, T exchanged for a space
    expect(body.start_time).toBe('2026-07-31 12:59:00')
    expect(body.end_time).toBe('2026-07-31 12:59:18')
    // seconds -> the record's native minutes (÷60)
    expect(body.actual_total_timetaken_in_min).toBeCloseTo(0.3, 6)
    expect(body.staging_timetaken_in_min).toBeCloseTo(2.0, 6)
    // explicit entry wins over the derived duration
    expect(body.actual_total_derived_from_timestamps).toBe(false)
  })

  it('derives the duration from Start/End when no explicit total is entered', { timeout: 20000 }, async () => {
    const user = userEvent.setup()
    const fetchMock = mockFetchOnce(OK_RESPONSE)
    vi.stubGlobal('fetch', fetchMock)
    render(<ActualOutcomeForm />)

    await user.type(screen.getByTestId('out-loader'), 'LoaderA')
    await user.type(screen.getByTestId('out-dataset'), 'DS')
    await user.type(screen.getByTestId('out-sprint'), 'S1')
    await user.type(screen.getByTestId('out-connection'), 'C1')
    fireEvent.change(screen.getByTestId('out-start'), { target: { value: '2026-07-31T12:59:23' } })
    fireEvent.change(screen.getByTestId('out-end'), { target: { value: '2026-07-31T12:59:53' } })
    await user.click(screen.getByTestId('outcome-submit'))

    await waitFor(() => expect(screen.getByTestId('outcome-done')).toBeInTheDocument())
    const body = JSON.parse(String((fetchMock.mock.calls[0] as unknown as [string, RequestInit])[1].body))
    expect(body.start_time).toBe('2026-07-31 12:59:23')
    expect(body.actual_total_timetaken_in_min).toBeCloseTo(0.5, 6)
    // the derivation is disclosed in the API body
    expect(body.actual_total_derived_from_timestamps).toBe(true)
  })
})
