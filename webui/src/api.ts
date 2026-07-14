import type { GatewayEvent, MemoryPayload, SessionDetail, SessionSummary } from './types'

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
}): Promise<{ id: string; chat_id: string; accepted: boolean }> {
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
