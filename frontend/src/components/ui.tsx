import type { ReactNode } from 'react'

/**
 * Shared vintage-theme UI primitives.
 *
 * One coherent visual system for every page: warm ivory workspace, soft-white
 * cards, charcoal typography, bronze/sage accents, semantic badges that always
 * carry a text label (never color alone).
 */

export function PageHeader({
  eyebrow,
  title,
  description,
  children,
}: {
  eyebrow: string
  title: string
  description?: string
  children?: ReactNode
}) {
  return (
    <header className="mb-6">
      {eyebrow ? <p className="text-xs font-semibold uppercase tracking-[0.14em] text-bronze">{eyebrow}</p> : null}
      <h1 className="mt-1 font-serif text-2xl font-bold tracking-tight text-ink">{title}</h1>
      {description && <p className="mt-2 max-w-3xl text-sm leading-relaxed text-warmgray">{description}</p>}
      {children}
    </header>
  )
}

export function Card({
  title,
  hint,
  children,
  className = '',
  testId,
}: {
  title?: string
  hint?: ReactNode
  children: ReactNode
  className?: string
  testId?: string
}) {
  return (
    <section className={`card p-5 ${className}`} data-testid={testId}>
      {title && (
        <h2 className="text-xs font-semibold uppercase tracking-[0.12em] text-warmgray">{title}</h2>
      )}
      {hint && <p className="mt-1 text-xs leading-relaxed text-warmgray">{hint}</p>}
      <div className={title || hint ? 'mt-3' : ''}>{children}</div>
    </section>
  )
}

export function Metric({
  label,
  value,
  children,
  sub,
  accent = false,
  testId,
}: {
  label: string
  value?: ReactNode
  /** Children form is accepted for call-site brevity (<Metric>x</Metric>). */
  children?: ReactNode
  sub?: string
  /** Restrained antique-gold left rule for the few hero metrics. */
  accent?: boolean
  testId?: string
}) {
  return (
    <div
      className={`rounded-md border border-linen bg-[#fbf8f1] p-3 ${accent ? 'border-l-2 border-l-gold' : ''}`}
      data-testid={testId}
    >
      <dt className="text-[11px] font-medium uppercase tracking-wide text-warmgray">{label}</dt>
      <dd className="mt-0.5 text-lg font-bold text-ink">{value ?? children}</dd>
      {sub && <p className="mt-0.5 text-[11px] leading-snug text-warmgray">{sub}</p>}
    </div>
  )
}

/** Semantic badge palette. Every badge renders its label — color is never the
 * sole carrier of meaning (accessibility rule). */
export type BadgeTone = 'neutral' | 'success' | 'warning' | 'risk' | 'error' | 'bronze' | 'sage' | 'gold'

const BADGE_TONES: Record<BadgeTone, string> = {
  neutral: 'bg-[#efece4] text-warmgray',
  success: 'bg-[#e6ece7] text-forest',
  warning: 'bg-[#f3ead9] text-ochre',
  risk: 'bg-[#f2e3df] text-terra',
  error: 'bg-[#f1e0de] text-ember',
  bronze: 'bg-[#ece3d8] text-bronze',
  sage: 'bg-[#e7ebe6] text-sage',
  gold: 'bg-[#f4ead6] text-[#8a6a33]',
}

export function Badge({
  tone = 'neutral',
  children,
  testId,
  title,
}: {
  tone?: BadgeTone
  children: ReactNode
  testId?: string
  title?: string
}) {
  return (
    <span className={`badge ${BADGE_TONES[tone]}`} data-testid={testId} title={title}>
      {children}
    </span>
  )
}

export function EmptyState({
  title,
  children,
  action,
}: {
  title: string
  children?: ReactNode
  action?: ReactNode
}) {
  return (
    <div className="rounded-md border border-dashed border-linen bg-[#fbf8f1] p-6 text-center">
      <p className="text-sm font-semibold text-ink">{title}</p>
      {children && <p className="mx-auto mt-1 max-w-lg text-sm leading-relaxed text-warmgray">{children}</p>}
      {action && <div className="mt-3">{action}</div>}
    </div>
  )
}

export function LoadingState({ message }: { message: string }) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex min-h-40 items-center justify-center rounded-md border border-linen bg-parchment p-8"
    >
      <div className="text-center">
        <div className="mx-auto h-7 w-7 animate-spin rounded-full border-2 border-bronze border-t-transparent" />
        <p className="mt-3 text-sm text-warmgray">{message}</p>
      </div>
    </div>
  )
}

export function ErrorState({ title, message }: { title: string; message: string }) {
  return (
    <div role="alert" className="rounded-md border border-[#e3c6c1] bg-[#f6e7e4] p-4">
      <h3 className="text-sm font-semibold text-ember">{title}</h3>
      <p className="mt-1 text-sm text-ember">{message}</p>
    </div>
  )
}

/** Small vintage divider with a gold tick, used between major sections. */
export function SectionRule() {
  return (
    <div className="my-5 flex items-center gap-2" aria-hidden="true">
      <div className="h-px flex-1 bg-linen" />
      <div className="h-1 w-1 rounded-full bg-gold" />
      <div className="h-px flex-1 bg-linen" />
    </div>
  )
}

export function DataTable({ children, testId }: { children: ReactNode; testId?: string }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm" data-testid={testId}>
        {children}
      </table>
    </div>
  )
}

export function Th({ children, className = '' }: { children: ReactNode; className?: string }) {
  return (
    <th
      className={`border-b border-linen py-2 pr-3 text-[11px] font-semibold uppercase tracking-wide text-warmgray ${className}`}
    >
      {children}
    </th>
  )
}

export function Td({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <td className={`border-b border-[#efece4] py-2 pr-3 ${className}`}>{children}</td>
}
