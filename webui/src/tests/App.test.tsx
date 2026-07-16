import { act, cleanup, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, beforeEach, describe, expect, test, vi } from 'vitest'
import App from '../App'
import type { GatewayEvent } from '../types'

class MockEventSource {
  static instances: MockEventSource[] = []
  readonly url: string
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  onmessage: ((event: MessageEvent<string>) => void) | null = null
  closed = false

  constructor(url: string | URL) {
    this.url = String(url)
    MockEventSource.instances.push(this)
  }

  close() {
    this.closed = true
  }

  emit(event: GatewayEvent) {
    this.onmessage?.({ data: JSON.stringify(event) } as MessageEvent<string>)
  }
}

function jsonResponse(payload: unknown, status = 200) {
  return Promise.resolve(new Response(JSON.stringify(payload), {
    status,
    headers: { 'Content-Type': 'application/json' },
  }))
}

const savedSession = {
  key: 'cli:work',
  channel: 'cli',
  title: '工作讨论',
  preview: '继续昨天的任务',
  created_at: '2026-07-14T08:00:00',
  updated_at: new Date().toISOString(),
  message_count: 2,
}

describe('MyClaw WebUI', () => {
  beforeEach(() => {
    MockEventSource.instances = []
    vi.stubGlobal('EventSource', MockEventSource)
  })

  afterEach(() => {
    cleanup()
    vi.unstubAllGlobals()
  })

  test('creates a new conversation, sends a message, and merges streamed output', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/sessions') return jsonResponse({ sessions: [] })
      if (url === '/api/messages' && init?.method === 'POST') {
        return jsonResponse({ id: 'req-1', chat_id: 'web-test', accepted: true }, 202)
      }
      throw new Error(`Unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<App />)

    const input = await screen.findByLabelText('消息内容')
    await user.type(input, '你好{enter}')

    expect(screen.getByText('你好')).toBeInTheDocument()
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/messages', expect.objectContaining({
      method: 'POST',
    })))
    const postCall = fetchMock.mock.calls.find(([url]) => url === '/api/messages')!
    const body = JSON.parse(String(postCall[1]?.body))
    expect(body.content).toBe('你好')
    expect(body.chat_id).toMatch(/^web-/)
    expect(body).not.toHaveProperty('session_key')

    const stream = MockEventSource.instances.at(-1)!
    act(() => {
      stream.emit({ type: 'message_delta', id: 'req-1', chat_id: body.chat_id, content: '你', terminal: false })
      stream.emit({ type: 'message_delta', id: 'req-1', chat_id: body.chat_id, content: '好！', terminal: false })
      stream.emit({ type: 'message', id: 'req-1', chat_id: body.chat_id, content: '你好！', terminal: true })
    })

    expect(screen.getByText('你好！')).toBeInTheDocument()
    expect(input).not.toBeDisabled()
  })

  test('loads a saved session and resumes it with its session key', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url === '/api/sessions') return jsonResponse({ sessions: [savedSession] })
      if (url.includes('/api/sessions?key=')) {
        return jsonResponse({
          key: 'cli:work',
          title: '工作讨论',
          messages: [
            { role: 'user', content: '继续昨天的任务' },
            { role: 'assistant', content: '好的，我们继续。' },
          ],
        })
      }
      if (url === '/api/messages' && init?.method === 'POST') {
        return jsonResponse({ id: 'req-2', chat_id: 'cli:work', accepted: true }, 202)
      }
      throw new Error(`Unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<App />)

    await user.click(await screen.findByRole('button', { name: /工作讨论/ }))
    expect(await screen.findByText('好的，我们继续。')).toBeInTheDocument()
    await waitFor(() => expect(MockEventSource.instances.at(-1)?.url).toContain('chat_id=cli%3Awork'))

    const input = screen.getByLabelText('消息内容')
    await user.type(input, '下一步{enter}')
    await waitFor(() => expect(fetchMock).toHaveBeenCalledWith('/api/messages', expect.anything()))
    const postCall = fetchMock.mock.calls.find(([url]) => url === '/api/messages')!
    expect(JSON.parse(String(postCall[1]?.body))).toMatchObject({
      chat_id: 'cli:work',
      session_key: 'cli:work',
      content: '下一步',
    })
  })

  test('shows progress and reconnecting state without mixing status into assistant text', async () => {
    vi.stubGlobal('fetch', vi.fn(() => jsonResponse({ sessions: [] })))
    render(<App />)
    await screen.findByText('今天想一起完成什么？')
    const stream = MockEventSource.instances.at(-1)!

    act(() => {
      stream.onopen?.()
      stream.emit({ type: 'tool_progress', id: 'req-3', chat_id: 'x', content: '正在运行工具', terminal: false })
      stream.emit({ type: 'error', id: 'req-3', chat_id: 'x', content: '工具执行失败', terminal: true })
    })

    expect(screen.getByText('正在运行工具')).toBeInTheDocument()
    expect(screen.getByText('工具执行失败')).toBeInTheDocument()
    expect(screen.getAllByText('在线').length).toBeGreaterThan(0)

    act(() => stream.onerror?.())
    expect(screen.getAllByText('正在重连').length).toBeGreaterThan(0)
  })

  test('opens the read-only memory drawer and switches between memory sections', async () => {
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url === '/api/sessions') return jsonResponse({ sessions: [] })
      if (url === '/api/memory') {
        return jsonResponse({
          memory: '# 项目记忆\n\n使用 Python。',
          user: '# 用户信息\n\n偏好简洁回答。',
          soul: '',
        })
      }
      throw new Error(`Unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: /查看记忆/ }))
    expect(await screen.findByText('使用 Python。')).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: /用户信息/ }))
    expect(screen.getByText('偏好简洁回答。')).toBeInTheDocument()
    await user.click(screen.getByRole('tab', { name: /助手设定/ }))
    expect(screen.getByText('助手设定还是空的')).toBeInTheDocument()
  })

  test('uses Shift+Enter for a newline without sending', async () => {
    const fetchMock = vi.fn(() => jsonResponse({ sessions: [] }))
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<App />)
    const input = await screen.findByLabelText('消息内容')

    await user.type(input, '第一行{shift>}{enter}{/shift}第二行')

    expect(input).toHaveValue('第一行\n第二行')
    expect(fetchMock).toHaveBeenCalledTimes(1)
  })

  test('opens monitoring, shows metrics, and displays a trace waterfall', async () => {
    const trace = {
      trace_id: 'a'.repeat(32), root_span_id: 'b'.repeat(16), request_id: 'req-1',
      name: 'agent.request', kind: 'conversation', status: 'ok',
      started_at: new Date().toISOString(), ended_at: new Date().toISOString(), duration_ms: 120,
      session_key: 'gateway:direct', channel: 'gateway', model: 'fake',
      error_type: null, error_message: null, attributes: {},
      prompt_tokens: 4, completion_tokens: 2, total_tokens: 6,
    }
    const fetchMock = vi.fn((input: RequestInfo | URL) => {
      const url = String(input)
      if (url === '/api/sessions') return jsonResponse({ sessions: [] })
      if (url.startsWith('/api/observability/summary')) return jsonResponse({
        window: '24h', since: new Date().toISOString(), generated_at: new Date().toISOString(),
        requests: 1, running: 0, errors: 0, success_rate: 1,
        duration_ms: { p50: 120, p95: 120 }, queue_wait_ms: { p95: 2 },
        llm_calls: 1, tool_calls: 0, tool_errors: 0,
        tokens: { prompt: 4, completion: 2, total: 6, available: true },
        series: [{ bucket: new Date().toISOString(), requests: 1, errors: 0 }],
      })
      if (url.startsWith('/api/observability/traces?')) return jsonResponse({ traces: [trace] })
      if (url.startsWith('/api/observability/logs')) return jsonResponse({ logs: [] })
      if (url === `/api/observability/traces/${trace.trace_id}`) return jsonResponse({
        trace,
        spans: [{
          span_id: 'c'.repeat(16), trace_id: trace.trace_id, parent_span_id: trace.root_span_id,
          name: 'llm.complete', kind: 'llm', status: 'ok',
          started_at: trace.started_at, ended_at: trace.ended_at, duration_ms: 90,
          error_type: null, error_message: null, attributes: { model: 'fake' },
          prompt_tokens: 4, completion_tokens: 2, total_tokens: 6,
        }],
        logs: [],
      })
      throw new Error(`Unexpected request: ${url}`)
    })
    vi.stubGlobal('fetch', fetchMock)
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('button', { name: '运行监控' }))
    expect(await screen.findByText('100.0%')).toBeInTheDocument()
    await user.click(await screen.findByRole('button', { name: /agent.request/ }))
    expect(await screen.findByText('llm.complete')).toBeInTheDocument()
    expect(screen.getByRole('complementary', { name: 'Trace 详情' })).toBeInTheDocument()
  })
})
