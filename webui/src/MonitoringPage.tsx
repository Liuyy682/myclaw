import { useCallback, useEffect, useMemo, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Bot,
  Clock3,
  Database,
  RefreshCw,
  Search,
  Wrench,
  X,
} from 'lucide-react'
import {
  getObservabilitySummary,
  getTrace,
  listObservabilityLogs,
  listTraces,
} from './api'
import type {
  ObservabilityLog,
  ObservabilitySummary,
  TraceDetail,
  TraceSpan,
  TraceStatus,
  TraceSummary,
} from './types'

const WINDOWS = ['1h', '24h', '7d']

function formatDuration(value: number | null) {
  if (value == null) return '—'
  if (value < 1000) return `${Math.round(value)} ms`
  return `${(value / 1000).toFixed(value < 10_000 ? 2 : 1)} s`
}

function formatTime(value: string) {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : new Intl.DateTimeFormat('zh-CN', {
    month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
  }).format(date)
}

function shortId(value: string) {
  return value ? value.slice(0, 8) : '—'
}

function spanDepth(span: TraceSpan, spans: TraceSpan[]) {
  const parents = new Map(spans.map((item) => [item.span_id, item.parent_span_id]))
  let parent = span.parent_span_id
  let depth = 0
  while (parent && parents.has(parent) && depth < 8) {
    depth += 1
    parent = parents.get(parent) || null
  }
  return depth
}

export default function MonitoringPage({ onBack }: { onBack: () => void }) {
  const [window, setWindow] = useState('24h')
  const [status, setStatus] = useState<TraceStatus | ''>('')
  const [level, setLevel] = useState('')
  const [query, setQuery] = useState('')
  const [tab, setTab] = useState<'traces' | 'logs'>('traces')
  const [summary, setSummary] = useState<ObservabilitySummary | null>(null)
  const [traces, setTraces] = useState<TraceSummary[]>([])
  const [logs, setLogs] = useState<ObservabilityLog[]>([])
  const [detail, setDetail] = useState<TraceDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')

  const refresh = useCallback(async (quiet = false) => {
    if (!quiet) setLoading(true)
    try {
      const [nextSummary, nextTraces, nextLogs] = await Promise.all([
        getObservabilitySummary(window),
        listTraces({ window, status }),
        listObservabilityLogs({ window, level, query }),
      ])
      setSummary(nextSummary)
      setTraces(nextTraces)
      setLogs(nextLogs)
      setError('')
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : String(requestError))
    } finally {
      setLoading(false)
    }
  }, [level, query, status, window])

  useEffect(() => {
    void refresh()
    const timer = globalThis.setInterval(() => void refresh(true), 5000)
    return () => globalThis.clearInterval(timer)
  }, [refresh])

  const openTrace = async (trace: TraceSummary) => {
    try {
      setDetail(await getTrace(trace.trace_id))
      setError('')
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : String(requestError))
    }
  }

  const openTraceId = async (traceId: string) => {
    try {
      setDetail(await getTrace(traceId))
      setError('')
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : String(requestError))
    }
  }

  const maxSeries = useMemo(
    () => Math.max(1, ...(summary?.series.map((item) => item.requests) || [1])),
    [summary],
  )

  return (
    <main className="monitor-panel">
      <header className="monitor-header">
        <button className="icon-button" onClick={onBack} aria-label="返回对话"><ArrowLeft size={19} /></button>
        <div>
          <span className="eyebrow">本地可观测性</span>
          <h1>运行监控</h1>
        </div>
        <div className="monitor-header-actions">
          <select aria-label="监控时间范围" value={window} onChange={(event) => setWindow(event.target.value)}>
            {WINDOWS.map((item) => <option key={item} value={item}>{item}</option>)}
          </select>
          <button className="icon-button" onClick={() => void refresh()} aria-label="刷新监控数据">
            <RefreshCw size={17} className={loading ? 'spin' : ''} />
          </button>
        </div>
      </header>

      <section className="monitor-content">
        {error && <div className="monitor-error"><AlertTriangle size={17} />{error}</div>}
        <div className="metric-grid">
          <Metric icon={Activity} label="请求数" value={summary?.requests ?? '—'} note={`${summary?.running ?? 0} 个运行中`} />
          <Metric icon={Bot} label="成功率" value={summary?.success_rate == null ? '—' : `${(summary.success_rate * 100).toFixed(1)}%`} note={`${summary?.errors ?? 0} 个错误`} />
          <Metric icon={Clock3} label="P95 耗时" value={formatDuration(summary?.duration_ms.p95 ?? null)} note={`P50 ${formatDuration(summary?.duration_ms.p50 ?? null)}`} />
          <Metric icon={Wrench} label="工具调用" value={summary?.tool_calls ?? '—'} note={`${summary?.tool_errors ?? 0} 个错误`} />
          <Metric icon={Database} label="Token" value={summary?.tokens.available ? summary.tokens.total.toLocaleString() : '未知'} note={`${summary?.llm_calls ?? 0} 次 LLM 调用`} />
        </div>

        <section className="monitor-card trend-card" aria-label="请求趋势">
          <div className="monitor-card-title"><div><span className="eyebrow">趋势</span><h2>请求与错误</h2></div></div>
          <div className="trend-bars">
            {(summary?.series || []).slice(-24).map((item) => (
              <div className="trend-column" key={item.bucket} title={`${item.bucket}: ${item.requests} 请求 / ${item.errors} 错误`}>
                <div className="trend-total" style={{ height: `${Math.max(6, item.requests / maxSeries * 100)}%` }}>
                  {item.errors > 0 && <div className="trend-error" style={{ height: `${item.errors / item.requests * 100}%` }} />}
                </div>
              </div>
            ))}
            {summary?.series.length === 0 && <div className="monitor-empty">当前时间范围内还没有 Trace</div>}
          </div>
        </section>

        <div className="monitor-tabs" role="tablist">
          <button role="tab" className={tab === 'traces' ? 'active' : ''} onClick={() => setTab('traces')}>Trace</button>
          <button role="tab" className={tab === 'logs' ? 'active' : ''} onClick={() => setTab('logs')}>日志</button>
        </div>

        {tab === 'traces' ? (
          <section className="monitor-card">
            <div className="monitor-toolbar">
              <select aria-label="Trace 状态" value={status} onChange={(event) => setStatus(event.target.value as TraceStatus | '')}>
                <option value="">全部状态</option>
                <option value="running">运行中</option><option value="ok">成功</option>
                <option value="error">错误</option><option value="cancelled">已取消</option>
                <option value="abandoned">已中断</option>
              </select>
              <span>{traces.length} 条</span>
            </div>
            <div className="trace-list">
              {traces.map((trace) => (
                <button key={trace.trace_id} className="trace-row" onClick={() => void openTrace(trace)}>
                  <span className={`status-badge ${trace.status}`}>{trace.status}</span>
                  <span><strong>{trace.name}</strong><small>{trace.kind} · {shortId(trace.trace_id)}</small></span>
                  <span><strong>{trace.session_key || '后台任务'}</strong><small>{trace.model || trace.channel || '—'}</small></span>
                  <span><strong>{formatDuration(trace.duration_ms)}</strong><small>{formatTime(trace.started_at)}</small></span>
                </button>
              ))}
              {!loading && traces.length === 0 && <div className="monitor-empty">没有符合条件的 Trace</div>}
            </div>
          </section>
        ) : (
          <section className="monitor-card">
            <div className="monitor-toolbar log-toolbar">
              <select aria-label="日志级别" value={level} onChange={(event) => setLevel(event.target.value)}>
                <option value="">全部级别</option><option value="INFO">INFO</option>
                <option value="WARNING">WARNING</option><option value="ERROR">ERROR</option>
                <option value="CRITICAL">CRITICAL</option>
              </select>
              <label className="monitor-search"><Search size={15} /><input aria-label="搜索日志" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="搜索日志" /></label>
            </div>
            <div className="log-list">
              {logs.map((log) => (
                <div className="log-row" key={log.id}>
                  <span className={`log-level ${log.level.toLowerCase()}`}>{log.level}</span>
                  <time>{formatTime(log.timestamp)}</time>
                  <strong>{log.component}</strong>
                  <span>{log.message}</span>
                  {log.trace_id && <button onClick={() => void openTraceId(log.trace_id)}>{shortId(log.trace_id)}</button>}
                </div>
              ))}
              {!loading && logs.length === 0 && <div className="monitor-empty">没有符合条件的日志</div>}
            </div>
          </section>
        )}
      </section>

      {detail && <TraceDrawer detail={detail} onClose={() => setDetail(null)} />}
    </main>
  )
}

function Metric({ icon: Icon, label, value, note }: { icon: typeof Activity; label: string; value: string | number; note: string }) {
  return <div className="metric-card"><Icon size={18} /><span>{label}</span><strong>{value}</strong><small>{note}</small></div>
}

function TraceDrawer({ detail, onClose }: { detail: TraceDetail; onClose: () => void }) {
  const start = new Date(detail.trace.started_at).getTime()
  const duration = Math.max(detail.trace.duration_ms || 1, 1)
  return <>
    <div className="drawer-scrim" onClick={onClose} />
    <aside className="trace-drawer" aria-label="Trace 详情" aria-modal="true">
      <div className="trace-drawer-header">
        <div><span className="eyebrow">Trace {shortId(detail.trace.trace_id)}</span><h2>{detail.trace.name}</h2></div>
        <button className="icon-button" onClick={onClose} aria-label="关闭 Trace 详情"><X size={19} /></button>
      </div>
      <div className="trace-meta">
        <span className={`status-badge ${detail.trace.status}`}>{detail.trace.status}</span>
        <span>{detail.trace.session_key || '后台任务'}</span><span>{formatDuration(detail.trace.duration_ms)}</span>
      </div>
      <div className="waterfall">
        {detail.spans.map((span) => {
          const offset = Math.max(0, (new Date(span.started_at).getTime() - start) / duration * 100)
          const width = Math.max(1.5, span.duration_ms / duration * 100)
          return <div className="waterfall-row" key={span.span_id}>
            <div className="waterfall-label" style={{ paddingLeft: `${spanDepth(span, detail.spans) * 14}px` }}>
              <strong>{span.name}</strong><small>{span.kind} · {formatDuration(span.duration_ms)}</small>
            </div>
            <div className="waterfall-track"><span className={span.status} style={{ left: `${Math.min(98.5, offset)}%`, width: `${Math.min(100 - offset, width)}%` }} /></div>
            {(span.error_message || Object.keys(span.attributes).length > 0) && <details><summary>属性</summary><pre>{JSON.stringify({ ...span.attributes, error: span.error_message }, null, 2)}</pre></details>}
          </div>
        })}
        {detail.spans.length === 0 && <div className="monitor-empty">该 Trace 暂无已完成 Span</div>}
      </div>
    </aside>
  </>
}
