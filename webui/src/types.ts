export type MessageRole = 'user' | 'assistant'

export interface SessionSummary {
  key: string
  channel: string
  title: string
  preview: string
  created_at: string
  updated_at: string
  message_count: number
}

export interface SessionDetail {
  key: string
  title: string
  messages: Array<{ role: MessageRole; content: string }>
}

export interface MemoryPayload {
  memory: string
  user: string
  soul: string
}

export interface GatewayEvent {
  type: 'message_delta' | 'message' | 'tool_progress' | 'control' | 'error'
  id?: string
  chat_id: string
  content: string
  terminal: boolean
  metadata?: {
    request_id?: string
    trace_id?: string
    session_key?: string
    [key: string]: unknown
  }
}

export interface ViewMessage {
  id: string
  role: MessageRole | 'status' | 'error'
  content: string
}

export type ConnectionState = 'connecting' | 'connected' | 'reconnecting'

export type TraceStatus = 'running' | 'ok' | 'error' | 'cancelled' | 'abandoned'

export interface TraceSummary {
  trace_id: string
  root_span_id: string
  request_id: string
  name: string
  kind: string
  status: TraceStatus
  started_at: string
  ended_at: string | null
  duration_ms: number | null
  session_key: string
  channel: string
  model: string
  error_type: string | null
  error_message: string | null
  attributes: Record<string, unknown>
  prompt_tokens: number | null
  completion_tokens: number | null
  total_tokens: number | null
}

export interface TraceSpan {
  span_id: string
  trace_id: string
  parent_span_id: string | null
  name: string
  kind: string
  status: TraceStatus
  started_at: string
  ended_at: string
  duration_ms: number
  error_type: string | null
  error_message: string | null
  attributes: Record<string, unknown>
  prompt_tokens: number | null
  completion_tokens: number | null
  total_tokens: number | null
}

export interface ObservabilityLog {
  id: number
  timestamp: string
  level: string
  component: string
  message: string
  trace_id: string
  span_id: string
  request_id: string
  session_key: string
  error_type: string | null
  error_message: string | null
  attributes: Record<string, unknown>
}

export interface ObservabilitySummary {
  window: string
  since: string
  generated_at: string
  requests: number
  running: number
  errors: number
  success_rate: number | null
  duration_ms: { p50: number | null; p95: number | null }
  queue_wait_ms: { p95: number | null }
  llm_calls: number
  tool_calls: number
  tool_errors: number
  tokens: { prompt: number; completion: number; total: number; available: boolean }
  series: Array<{ bucket: string; requests: number; errors: number }>
}

export interface TraceDetail {
  trace: TraceSummary
  spans: TraceSpan[]
  logs: ObservabilityLog[]
}
