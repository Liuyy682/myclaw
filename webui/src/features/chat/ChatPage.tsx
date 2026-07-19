import { FormEvent, KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Bot,
  BrainCircuit,
  ChevronRight,
  Menu,
  Send,
} from 'lucide-react'
import { getSession, listSessions, sendMessage } from '../../shared/api/gateway'
import MemoryDrawer from '../memory/MemoryDrawer'
import SessionSidebar from '../sessions/SessionSidebar'
import { useGatewayEvents } from './useGatewayEvents'
import MessageList from './MessageList'
import type {
  SessionSummary,
} from '../../shared/types/gateway'

function createChatId() {
  const id = typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
  return `web-${id}`
}

function chatIdFromSessionKey(key: string) {
  return key.startsWith('gateway:') ? key.slice('gateway:'.length) || 'direct' : key
}


function ChatPage({ onShowMonitoring }: { onShowMonitoring: () => void }) {
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [sessionsLoading, setSessionsLoading] = useState(true)
  const [sessionsError, setSessionsError] = useState('')
  const [active, setActive] = useState(() => ({
    chatId: createChatId(),
    sessionKey: null as string | null,
    title: '新对话',
  }))
  const [draft, setDraft] = useState('')
  const [conversationLoading, setConversationLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [memoryOpen, setMemoryOpen] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const refreshSessions = useCallback(async () => {
    setSessionsError('')
    try {
      setSessions(await listSessions())
    } catch (error) {
      setSessionsError(error instanceof Error ? error.message : String(error))
    } finally {
      setSessionsLoading(false)
    }
  }, [])

  useEffect(() => {
    void refreshSessions()
  }, [refreshSessions])

  const {
    connection,
    isSending,
    markPendingAsAnswered,
    messages,
    setIsSending,
    setMessages,
  } = useGatewayEvents(active.chatId, refreshSessions)

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const startNewConversation = useCallback(() => {
    setActive({ chatId: createChatId(), sessionKey: null, title: '新对话' })
    setMessages([])
    setDraft('')
    setIsSending(false)
    setConversationLoading(false)
    setSidebarOpen(false)
  }, [])

  const selectSession = useCallback(async (session: SessionSummary) => {
    if (session.key === active.sessionKey) {
      setSidebarOpen(false)
      return
    }
    setConversationLoading(true)
    setIsSending(false)
    setMessages([])
    setSidebarOpen(false)
    try {
      const detail = await getSession(session.key)
      setActive({
        chatId: chatIdFromSessionKey(detail.key),
        sessionKey: detail.key,
        title: detail.title || session.title,
      })
      setMessages(detail.messages.map((message, index) => ({
        id: `history:${detail.key}:${index}`,
        role: message.role,
        content: message.content,
      })))
    } catch (error) {
      setMessages([{
        id: 'session-load-error',
        role: 'error',
        content: error instanceof Error ? error.message : String(error),
      }])
    } finally {
      setConversationLoading(false)
    }
  }, [active.sessionKey])

  const submitMessage = useCallback(async () => {
    const content = draft.trim()
    if (!content || isSending || conversationLoading) return

    const optimisticId = `user:${Date.now()}`
    setMessages((current) => [...current, { id: optimisticId, role: 'user', content }])
    markPendingAsAnswered()
    setDraft('')
    setIsSending(true)
    try {
      await sendMessage({
        chatId: active.chatId,
        sessionKey: active.sessionKey,
        content,
      })
      if (!active.sessionKey) {
        setActive((current) => ({ ...current, sessionKey: `gateway:${current.chatId}` }))
      }
    } catch (error) {
      setIsSending(false)
      setMessages((current) => [...current, {
        id: `send-error:${Date.now()}`,
        role: 'error',
        content: error instanceof Error ? error.message : String(error),
      }])
    }
  }, [active.chatId, active.sessionKey, conversationLoading, draft, isSending, markPendingAsAnswered, setIsSending, setMessages])

  const handleSubmit = (event: FormEvent) => {
    event.preventDefault()
    void submitMessage()
  }

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault()
      void submitMessage()
    }
  }


  const connectionLabel = useMemo(() => ({
    connecting: '正在连接',
    connected: '在线',
    reconnecting: '正在重连',
  })[connection], [connection])


  return (
    <div className="app-shell">
      <div className={`mobile-scrim ${sidebarOpen ? 'visible' : ''}`} onClick={() => setSidebarOpen(false)} />

      <SessionSidebar
        activeSessionKey={active.sessionKey}
        connection={connection}
        connectionLabel={connectionLabel}
        error={sessionsError}
        loading={sessionsLoading}
        onClose={() => setSidebarOpen(false)}
        onNewConversation={startNewConversation}
        onRefresh={refreshSessions}
        onSelect={selectSession}
        onShowMonitoring={onShowMonitoring}
        open={sidebarOpen}
        sessions={sessions}
      />

      <main className="chat-panel">
        <header className="chat-header">
          <button className="icon-button mobile-only" onClick={() => setSidebarOpen(true)} aria-label="打开会话栏">
            <Menu size={20} />
          </button>
          <div className="chat-title">
            <span className="eyebrow">当前对话</span>
            <h1>{active.title}</h1>
          </div>
          <div className="header-actions">
            <span className={`connection-pill ${connection}`}>
              <span />{connectionLabel}
            </span>
            <button className="memory-button" onClick={() => setMemoryOpen(true)}>
              <BrainCircuit size={18} />
              <span>查看记忆</span>
              <ChevronRight size={15} />
            </button>
          </div>
        </header>

        <MessageList
          endRef={messagesEndRef}
          isSending={isSending}
          loading={conversationLoading}
          messages={messages}
          onSuggestion={setDraft}
        />

        <form className="composer" onSubmit={handleSubmit}>
          <div className="composer-box">
            <textarea
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
              onKeyDown={handleComposerKeyDown}
              placeholder="给 MyClaw 发消息…"
              aria-label="消息内容"
              rows={1}
              disabled={conversationLoading}
            />
            <button
              className="send-button"
              type="submit"
              disabled={!draft.trim() || isSending || conversationLoading}
              aria-label="发送消息"
            >
              <Send size={18} />
            </button>
          </div>
          <p>Enter 发送 · Shift + Enter 换行 · 你的数据仅保存在本地</p>
        </form>
      </main>

      <MemoryDrawer open={memoryOpen} onClose={() => setMemoryOpen(false)} />
    </div>
  )
}

export default ChatPage
