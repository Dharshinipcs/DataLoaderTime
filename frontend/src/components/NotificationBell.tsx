import { useCallback, useEffect, useRef, useState } from 'react'
import {
  evaluateNotifications,
  fetchNotifications,
  markAllNotificationsRead,
  markNotificationRead,
  type NotificationRecord,
} from '../api'

/** Severity → label/icon/color triple. Text + icon always carry the signal;
 * color is a supporting cue only (accessibility rule). */
const SEVERITY_VIEW: Record<
  NotificationRecord['severity'],
  { glyph: string; className: string }
> = {
  INFO: { glyph: 'ℹ', className: 'text-sage' },
  WARNING: { glyph: '▲', className: 'text-bronze' },
  HIGH: { glyph: '◆', className: 'text-[#a4552f]' },
  CRITICAL: { glyph: '✖', className: 'text-[#8c2f1d]' },
}

function timeAgo(iso: string): string {
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return iso
  const mins = Math.max(0, Math.round((Date.now() - then) / 60000))
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins} min ago`
  const hrs = Math.round(mins / 60)
  if (hrs < 48) return `${hrs} h ago`
  return `${Math.round(hrs / 24)} d ago`
}

function contextLine(n: NotificationRecord): string | null {
  const parts: string[] = []
  for (const key of ['loader_name', 'dataset', 'stage', 'model_version', 'model_id']) {
    const v = n.context?.[key]
    if (typeof v === 'string' && v) parts.push(v)
  }
  return parts.length ? parts.join(' · ') : null
}

/**
 * AD-33 §21: system-level notification bell in the application shell.
 * Persisted unread/read state; notifications stay until explicitly marked
 * read. Alerts come only from real subsystem signals (risk, drift,
 * readiness, registry, feedback quality, stage coverage).
 */
export default function NotificationBell() {
  const [open, setOpen] = useState(false)
  const [notifications, setNotifications] = useState<NotificationRecord[]>([])
  const [unread, setUnread] = useState(0)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const popRef = useRef<HTMLDivElement | null>(null)

  const refresh = useCallback(() => {
    fetchNotifications()
      .then((feed) => {
        setNotifications(feed.notifications)
        setUnread(feed.unread_count)
      })
      .catch(() => setError('Notification feed unavailable.'))
  }, [])

  useEffect(() => {
    refresh()
    const timer = window.setInterval(refresh, 60_000)
    return () => window.clearInterval(timer)
  }, [refresh])

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (popRef.current && !popRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const onEvaluate = async () => {
    setBusy(true)
    setError(null)
    try {
      const feed = await evaluateNotifications()
      setNotifications(feed.notifications)
      setUnread(feed.unread_count)
    } catch {
      setError('Could not evaluate system signals.')
    } finally {
      setBusy(false)
    }
  }

  const onMarkRead = async (id: string) => {
    setBusy(true)
    try {
      const feed = await markNotificationRead(id)
      setNotifications(feed.notifications)
      setUnread(feed.unread_count)
    } catch {
      setError('Could not mark as read.')
    } finally {
      setBusy(false)
    }
  }

  const onMarkAll = async () => {
    setBusy(true)
    try {
      const feed = await markAllNotificationsRead()
      setNotifications(feed.notifications)
      setUnread(feed.unread_count)
    } catch {
      setError('Could not mark all as read.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="relative" ref={popRef}>
      <button
        type="button"
        aria-label={`Notifications (${unread} unread)`}
        data-testid="notification-bell"
        onClick={() => {
          setOpen((o) => !o)
          if (!open) refresh()
        }}
        className={
          'relative rounded-md px-2.5 py-1.5 text-sm transition ' +
          (open
            ? 'bg-white/10 text-[#f7f4ee]'
            : 'text-[#c9c4b6] hover:bg-white/5 hover:text-[#f7f4ee]')
        }
      >
        <span aria-hidden="true" className="text-base leading-none">
          🔔
        </span>
        {unread > 0 && (
          <span
            data-testid="notification-unread-badge"
            className="absolute -right-1 -top-1 flex h-4 min-w-4 items-center justify-center rounded-full bg-[#a4552f] px-1 text-[10px] font-bold leading-none text-[#f7f4ee]"
          >
            {unread > 99 ? '99+' : unread}
          </span>
        )}
      </button>

      {open && (
        <div
          data-testid="notification-panel"
          className="fixed left-16 top-16 z-50 w-[26rem] max-w-[92vw] rounded-md border border-linen bg-ivory shadow-lg"
        >
          <div className="flex items-center justify-between border-b border-linen px-4 py-2.5">
            <p className="font-serif text-sm font-bold text-ink">System notifications</p>
            <div className="flex items-center gap-2">
              <button
                type="button"
                data-testid="notification-evaluate"
                disabled={busy}
                onClick={onEvaluate}
                className="rounded px-2 py-1 text-[11px] font-medium text-bronze hover:bg-parchment disabled:opacity-50"
              >
                Check signals
              </button>
              <button
                type="button"
                data-testid="notification-mark-all"
                disabled={busy || unread === 0}
                onClick={onMarkAll}
                className="rounded px-2 py-1 text-[11px] font-medium text-warmgray hover:bg-parchment disabled:opacity-40"
              >
                Mark all read
              </button>
            </div>
          </div>

          {error && <p className="px-4 py-2 text-xs text-[#8c2f1d]">{error}</p>}

          <div className="max-h-[24rem] overflow-y-auto">
            {notifications.length === 0 ? (
              <p className="px-4 py-6 text-center text-xs text-warmgray" data-testid="notification-empty">
                No notifications. Use “Check signals” to evaluate drift, retraining
                readiness, registry candidates, and stage coverage.
              </p>
            ) : (
              <ul className="divide-y divide-linen">
                {notifications.map((n) => {
                  const sev = SEVERITY_VIEW[n.severity] ?? SEVERITY_VIEW.INFO
                  const ctx = contextLine(n)
                  return (
                    <li
                      key={n.notification_id}
                      data-testid="notification-item"
                      className={'px-4 py-3 ' + (n.read ? 'opacity-60' : 'bg-parchment/60')}
                    >
                      <div className="flex items-start gap-2.5">
                        <span aria-hidden="true" className={'mt-0.5 text-sm ' + sev.className}>
                          {sev.glyph}
                        </span>
                        <div className="min-w-0 flex-1">
                          <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5">
                            <span className="text-[10px] font-bold uppercase tracking-wider text-ink">
                              [{n.severity}]
                            </span>
                            <span className="text-xs font-semibold text-ink">{n.title}</span>
                            <span className="ml-auto text-[10px] text-warmgray" title={n.created_at}>
                              {timeAgo(n.created_at)}
                            </span>
                          </div>
                          <p className="mt-1 text-xs leading-relaxed text-warmgray">{n.message}</p>
                          {ctx && <p className="mt-1 text-[10px] text-warmgray">{ctx}</p>}
                          {!n.read && (
                            <button
                              type="button"
                              disabled={busy}
                              onClick={() => onMarkRead(n.notification_id)}
                              className="mt-1.5 rounded px-1.5 py-0.5 text-[11px] font-medium text-bronze hover:bg-parchment disabled:opacity-50"
                            >
                              Mark as read
                            </button>
                          )}
                        </div>
                        {!n.read && (
                          <span
                            aria-label="unread"
                            className="mt-1.5 h-2 w-2 shrink-0 rounded-full bg-[#a4552f]"
                          />
                        )}
                      </div>
                    </li>
                  )
                })}
              </ul>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
