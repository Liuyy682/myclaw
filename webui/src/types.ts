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
