import { useState } from 'react'
import NotificationBell from './components/NotificationBell'
import PredictionStudio from './pages/PredictionStudio'
import LoaderIntelligence from './pages/LoaderIntelligence'
import PredictionHistory from './pages/PredictionHistory'
import DriftRetraining from './pages/DriftRetraining'

type Page = 'prediction' | 'loaders' | 'history' | 'drift'

const NAV: { id: Page; label: string; glyph: string }[] = [
  { id: 'prediction', label: 'Pre-Execution Features', glyph: '◷' },
  { id: 'loaders', label: 'Loader Intelligence', glyph: '☰' },
  { id: 'history', label: 'Prediction History', glyph: '≡' },
  { id: 'drift', label: 'Drift & Retraining', glyph: '◔' },
]

export default function App() {
  const [page, setPage] = useState<Page>('prediction')

  return (
    <div className="min-h-screen bg-ivory">
      {/* Charcoal sidebar — the primary navigation surface. */}
      <div className="mx-auto flex max-w-[1400px]">
        <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col bg-charcoal text-[#e8e4da] md:flex print:hidden">
          <div className="border-b border-white/10 px-5 py-5">
            <div className="flex items-start justify-between gap-2">
              <div>
                <p className="font-serif text-lg font-bold tracking-tight text-[#f7f4ee]">
                  Data Loader Time
                </p>
              </div>
              {/* AD-33 §21: system-level notification center (top-right of the
                  shell, persistent across pages). */}
              <NotificationBell />
            </div>
          </div>
          <nav aria-label="Main" className="flex-1 space-y-0.5 px-3 py-4">
            {NAV.map((item) => {
              const active = page === item.id
              return (
                <button
                  key={item.id}
                  type="button"
                  onClick={() => setPage(item.id)}
                  aria-current={active ? 'page' : undefined}
                  className={
                    'flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-left text-sm transition ' +
                    (active
                      ? 'bg-white/10 font-semibold text-[#f7f4ee] shadow-[inset_2px_0_0_0_#c6a15b]'
                      : 'text-[#c9c4b6] hover:bg-white/5 hover:text-[#f7f4ee]')
                  }
                >
                  <span aria-hidden="true" className={active ? 'text-gold' : 'text-[#8a9186]'}>
                    {item.glyph}
                  </span>
                  {item.label}
                </button>
              )
            })}
          </nav>
          <div className="border-t border-white/10 px-5 py-3" />
        </aside>

        <div className="min-w-0 flex-1">
          {/* Compact top bar (mobile nav + brand on small screens). */}
          <header className="border-b border-linen bg-charcoal md:hidden">
            <div className="px-4 py-3">
              <p className="font-serif text-base font-bold text-[#f7f4ee]">Data Loader Time</p>
              <nav aria-label="Main" className="mt-2 flex flex-wrap gap-1">
                {NAV.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => setPage(item.id)}
                    aria-current={page === item.id ? 'page' : undefined}
                    className={
                      'rounded px-2 py-1 text-xs ' +
                      (page === item.id
                        ? 'bg-white/10 font-semibold text-gold'
                        : 'text-[#c9c4b6] hover:bg-white/5')
                    }
                  >
                    {item.label}
                  </button>
                ))}
              </nav>
            </div>
          </header>

          {/* Compact vertical rhythm on Prediction Studio so the enter-fields →
              predict → read-result workflow fits one desktop viewport. */}
          <main
            className={
              page === 'prediction'
                ? 'px-4 pt-5 pb-3 sm:px-6 lg:px-8'
                : 'px-4 py-8 sm:px-6 lg:px-8'
            }
          >
            <div className="mx-auto max-w-6xl">
              {page === 'prediction' ? (
                <PredictionStudio />
              ) : page === 'loaders' ? (
                <LoaderIntelligence />
              ) : page === 'history' ? (
                <PredictionHistory />
              ) : (
                <DriftRetraining />
              )}
            </div>
          </main>


        </div>
      </div>
    </div>
  )
}
