import type {
  GatewayEvent,
  MemoryPayload,
  ObservabilityLog,
  ObservabilitySummary,
  SessionDetail,
  SessionSummary,
  TraceDetail,
  TraceStatus,
  TraceSummary,
} from '../types/gateway'

async function readJson<T>(response: Response): Promise<T> {
  const payload = (await response.json()) as T & { error?: string }
  if (!response.ok) {
    throw new Error(payload.error || `请求失败（${response.status}）`)
  }
  return payload
}

export async function listSessions(): Promise<SessionSummary[]> {
  const response = await fetch('/api/sessions')
  const payload = await readJson<{ sessions: SessionSummary[] }>(response)
  return payload.sessions
}

export async function getSession(key: string): Promise<SessionDetail> {
  const response = await fetch(`/api/sessions?key=${encodeURIComponent(key)}`)
  return readJson<SessionDetail>(response)
}

export async function getMemory(): Promise<MemoryPayload> {
  const response = await fetch('/api/memory')
  return readJson<MemoryPayload>(response)
}

export async function sendMessage(input: {
  chatId: string
  content: string
  sessionKey: string | null
}): Promise<{ id: string; trace_id: string; chat_id: string; accepted: boolean }> {
  const body: { chat_id: string; content: string; session_key?: string } = {
    chat_id: input.chatId,
    content: input.content,
  }
  if (input.sessionKey) body.session_key = input.sessionKey

  const response = await fetch('/api/messages', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  return readJson(response)
}

export async function getObservabilitySummary(window = '24h'): Promise<ObservabilitySummary> {
  const response = await fetch(`/api/observability/summary?window=${encodeURIComponent(window)}`)
  return readJson<ObservabilitySummary>(response)
}

export async function listTraces(input: {
  window?: string
  status?: TraceStatus | ''
  sessionKey?: string
  limit?: number
} = {}): Promise<TraceSummary[]> {
  const query = new URLSearchParams({
    window: input.window || '24h',
    limit: String(input.limit || 50),
  })
  if (input.status) query.set('status', input.status)
  if (input.sessionKey) query.set('session_key', input.sessionKey)
  const response = await fetch(`/api/observability/traces?${query}`)
  const payload = await readJson<{ traces: TraceSummary[] }>(response)
  return payload.traces
}

export async function getTrace(traceId: string): Promise<TraceDetail> {
  const response = await fetch(`/api/observability/traces/${encodeURIComponent(traceId)}`)
  return readJson<TraceDetail>(response)
}

export async function listObservabilityLogs(input: {
  window?: string
  level?: string
  traceId?: string
  query?: string
  limit?: number
} = {}): Promise<ObservabilityLog[]> {
  const query = new URLSearchParams({
    window: input.window || '24h',
    limit: String(input.limit || 200),
  })
  if (input.level) query.set('level', input.level)
  if (input.traceId) query.set('trace_id', input.traceId)
  if (input.query) query.set('query', input.query)
  const response = await fetch(`/api/observability/logs?${query}`)
  const payload = await readJson<{ logs: ObservabilityLog[] }>(response)
  return payload.logs
}

export function openEventStream(
  chatId: string,
  handlers: {
    onEvent: (event: GatewayEvent) => void
    onOpen: () => void
    onError: () => void
  },
): () => void {
  const source = new EventSource(`/api/events?chat_id=${encodeURIComponent(chatId)}`)
  source.onopen = handlers.onOpen
  source.onerror = handlers.onError
  source.onmessage = (message) => {
    try {
      handlers.onEvent(JSON.parse(message.data) as GatewayEvent)
    } catch {
      handlers.onEvent({
        type: 'error',
        chat_id: chatId,
        content: '收到无法解析的服务器事件',
        terminal: false,
      })
    }
  }
  return () => source.close()
}
