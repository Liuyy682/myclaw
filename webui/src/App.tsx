import { FormEvent, KeyboardEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  Bot,
  Activity,
  BrainCircuit,
  ChevronRight,
  Menu,
  MessageSquare,
  PanelLeftClose,
  Plus,
  RefreshCw,
  Send,
  Sparkles,
  Terminal,
  UserRound,
  X,
} from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getMemory, getSession, listSessions, openEventStream, sendMessage } from './api'
import MonitoringPage from './MonitoringPage'
import type {
  ConnectionState,
  GatewayEvent,
  MemoryPayload,
  SessionSummary,
  ToolActivity,
  ViewMessage,
} from './types'

const EMPTY_MEMORY: MemoryPayload = { memory: '', user: '', soul: '' }

const MEMORY_TABS = [
  { key: 'memory' as const, label: '项目记忆', icon: BrainCircuit },
  { key: 'user' as const, label: '用户信息', icon: UserRound },
  { key: 'soul' as const, label: '助手设定', icon: Sparkles },
]

function createChatId() {
  const id = typeof crypto.randomUUID === 'function'
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(16).slice(2)}`
  return `web-${id}`
}

function chatIdFromSessionKey(key: string) {
  return key.startsWith('gateway:') ? key.slice('gateway:'.length) || 'direct' : key
}

function formatRelativeTime(value: string) {
  const timestamp = new Date(value).getTime()
  if (!Number.isFinite(timestamp)) return ''
  const minutes = Math.max(0, Math.floor((Date.now() - timestamp) / 60_000))
  if (minutes < 1) return '刚刚'
  if (minutes < 60) return `${minutes} 分钟前`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours} 小时前`
  const days = Math.floor(hours / 24)
  if (days < 7) return `${days} 天前`
  return new Intl.DateTimeFormat('zh-CN', { month: 'short', day: 'numeric' }).format(new Date(timestamp))
}

function markdown(content: string) {
  return (
    <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>
      {content}
    </ReactMarkdown>
  )
}

function toolActivity(event: GatewayEvent): ToolActivity {
  const rawProgress = event.metadata?.progress
  const progress = rawProgress && typeof rawProgress === 'object' ? rawProgress as Record<string, unknown> : {}
  const name = typeof progress.tool_name === 'string' ? progress.tool_name : 'tool'
  const rawArguments = progress.arguments
  const arguments_ = rawArguments && typeof rawArguments === 'object' && !Array.isArray(rawArguments)
    ? rawArguments as Record<string, unknown>
    : {}
  const index = typeof progress.index === 'number' ? progress.index : 0
  return {
    id: typeof progress.tool_call_id === 'string' ? progress.tool_call_id : `${name}:${index}`,
    name,
    arguments: arguments_,
    state: progress.event === 'tool_completed' ? 'completed' : 'running',
    fallback: event.content,
  }
}

function formatToolInvocation(tool: ToolActivity) {
  if (tool.name === 'exec' && typeof tool.arguments.cmd === 'string') return tool.arguments.cmd
  const argumentsText = Object.keys(tool.arguments).length > 0 ? ` ${JSON.stringify(tool.arguments, null, 2)}` : ''
  return `${tool.name}${argumentsText}` || tool.fallback
}

function App() {
  const [view, setView] = useState<'chat' | 'monitor'>('chat')
  const [sessions, setSessions] = useState<SessionSummary[]>([])
  const [sessionsLoading, setSessionsLoading] = useState(true)
  const [sessionsError, setSessionsError] = useState('')
  const [active, setActive] = useState(() => ({
    chatId: createChatId(),
    sessionKey: null as string | null,
    title: '新对话',
  }))
  const [messages, setMessages] = useState<ViewMessage[]>([])
  const [draft, setDraft] = useState('')
  const [isSending, setIsSending] = useState(false)
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const [conversationLoading, setConversationLoading] = useState(false)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [memoryOpen, setMemoryOpen] = useState(false)
  const [memory, setMemory] = useState<MemoryPayload>(EMPTY_MEMORY)
  const [memoryTab, setMemoryTab] = useState<keyof MemoryPayload>('memory')
  const [memoryLoading, setMemoryLoading] = useState(false)
  const [memoryError, setMemoryError] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const pendingAskRequestIdsRef = useRef(new Set<string>())
  const answeredAskRequestIdsRef = useRef(new Set<string>())

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

  const handleGatewayEvent = useCallback((event: GatewayEvent) => {
    const requestId = event.id || event.metadata?.request_id || active.chatId
    const assistantId = `assistant:${requestId}`
    const moveAssistantAfterAnswer = answeredAskRequestIdsRef.current.has(requestId)

    if (event.type === 'message_delta') {
      setMessages((current) => {
        const index = current.findIndex((item) => item.id === assistantId)
        if (index < 0) {
          return [...current, { id: assistantId, role: 'assistant', content: event.content }]
        }
        const assistant = { ...current[index], content: current[index].content + event.content }
        return moveAssistantAfterAnswer
          ? [...current.filter((item) => item.id !== assistantId), assistant]
          : current.map((item, itemIndex) => itemIndex === index ? assistant : item)
      })
      return
    }

    if (event.type === 'message') {
      setMessages((current) => {
        const existing = current.find((item) => item.id === assistantId)
        const assistant = existing
          ? { ...existing, content: event.content }
          : { id: assistantId, role: 'assistant' as const, content: event.content }
        return existing && moveAssistantAfterAnswer
          ? [...current.filter((item) => item.id !== assistantId), assistant]
          : existing
            ? current.map((item) => item.id === assistantId ? assistant : item)
            : [...current, assistant]
      })
    } else if (event.type === 'tool_progress') {
      const tool = toolActivity(event)
      const groupId = `tools:${requestId}`
      setMessages((current) => {
        const groupIndex = current.findIndex((item) => item.id === groupId)
        if (groupIndex < 0) return [...current, { id: groupId, role: 'tool_group', content: '', tools: [tool] }]
        return current.map((item, index) => {
          if (index !== groupIndex) return item
          const tools = item.tools || []
          const toolIndex = tools.findIndex((candidate) => candidate.id === tool.id)
          const nextTools = toolIndex < 0
            ? [...tools, tool]
            : tools.map((candidate, candidateIndex) => candidateIndex === toolIndex ? { ...candidate, ...tool } : candidate)
          return { ...item, tools: nextTools }
        })
      })
    } else {
      const role = event.type === 'error' ? 'error' : 'status'
      setMessages((current) => [
        ...current,
        {
          id: `${event.type}:${requestId}:${current.length}`,
          role,
          content: event.content,
        },
      ])
    }

    if (event.terminal) {
      setIsSending(false)
      void refreshSessions()
    }
    if (event.type === 'ask') {
      pendingAskRequestIdsRef.current.add(requestId)
      setIsSending(false)
    }
    if (event.terminal) {
      pendingAskRequestIdsRef.current.delete(requestId)
      answeredAskRequestIdsRef.current.delete(requestId)
    }
  }, [active.chatId, refreshSessions])

  useEffect(() => {
    setConnection('connecting')
    return openEventStream(active.chatId, {
      onEvent: handleGatewayEvent,
      onOpen: () => setConnection('connected'),
      onError: () => {
        setConnection('reconnecting')
        setIsSending(false)
      },
    })
  }, [active.chatId, handleGatewayEvent])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' })
  }, [messages])

  const startNewConversation = useCallback(() => {
    setView('chat')
    setActive({ chatId: createChatId(), sessionKey: null, title: '新对话' })
    setMessages([])
    setDraft('')
    setIsSending(false)
    setConversationLoading(false)
    setSidebarOpen(false)
  }, [])

  const selectSession = useCallback(async (session: SessionSummary) => {
    setView('chat')
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
    for (const requestId of pendingAskRequestIdsRef.current) {
      answeredAskRequestIdsRef.current.add(requestId)
    }
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
  }, [active.chatId, active.sessionKey, conversationLoading, draft, isSending])

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

  const loadMemory = useCallback(async () => {
    setMemoryLoading(true)
    setMemoryError('')
    try {
      setMemory(await getMemory())
    } catch (error) {
      setMemoryError(error instanceof Error ? error.message : String(error))
    } finally {
      setMemoryLoading(false)
    }
  }, [])

  const openMemory = () => {
    setMemoryOpen(true)
    if (memory === EMPTY_MEMORY && !memoryLoading) void loadMemory()
  }

  const connectionLabel = useMemo(() => ({
    connecting: '正在连接',
    connected: '在线',
    reconnecting: '正在重连',
  })[connection], [connection])

  const currentMemoryTab = MEMORY_TABS.find((tab) => tab.key === memoryTab)!
  const CurrentMemoryIcon = currentMemoryTab.icon

  return (
    <div className="app-shell">
      <div className={`mobile-scrim ${sidebarOpen ? 'visible' : ''}`} onClick={() => setSidebarOpen(false)} />

      <aside className={`sidebar ${sidebarOpen ? 'open' : ''}`} aria-label="会话导航">
        <div className="brand">
          <div className="brand-mark"><Bot size={21} strokeWidth={2.2} /></div>
          <div>
            <strong>MyClaw</strong>
            <span>个人智能助手</span>
          </div>
          <button className="icon-button mobile-only" onClick={() => setSidebarOpen(false)} aria-label="关闭会话栏">
            <PanelLeftClose size={19} />
          </button>
        </div>

        <button className="new-chat-button" onClick={startNewConversation}>
          <Plus size={18} />
          新建对话
        </button>

        <button
          className={`monitor-nav-button ${view === 'monitor' ? 'active' : ''}`}
          onClick={() => { setView('monitor'); setSidebarOpen(false) }}
        >
          <Activity size={18} />
          运行监控
        </button>

        <div className="sidebar-section-title">
          <span>最近对话</span>
          <button
            className="icon-button subtle"
            onClick={() => void refreshSessions()}
            aria-label="刷新会话"
            title="刷新会话"
          >
            <RefreshCw size={15} />
          </button>
        </div>

        <nav className="session-list" aria-label="历史会话">
          {sessionsLoading && <div className="sidebar-note">正在载入会话…</div>}
          {sessionsError && <div className="sidebar-note error-text">{sessionsError}</div>}
          {!sessionsLoading && !sessionsError && sessions.length === 0 && (
            <div className="sidebar-empty">
              <MessageSquare size={22} />
              <span>还没有历史对话</span>
            </div>
          )}
          {sessions.map((session) => (
            <button
              key={session.key}
              className={`session-card ${active.sessionKey === session.key ? 'active' : ''}`}
              onClick={() => void selectSession(session)}
              aria-current={active.sessionKey === session.key ? 'page' : undefined}
            >
              <span className="session-card-title">{session.title || '未命名对话'}</span>
              <span className="session-card-preview">{session.preview || '暂无内容'}</span>
              <span className="session-card-meta">
                <span>{session.channel.toUpperCase()}</span>
                <span>{formatRelativeTime(session.updated_at)}</span>
              </span>
            </button>
          ))}
        </nav>

        <div className="sidebar-footer">
          <span className={`connection-dot ${connection}`} />
          <div>
            <strong>{connectionLabel}</strong>
            <span>{active.sessionKey || '等待第一条消息'}</span>
          </div>
        </div>
      </aside>

      {view === 'monitor' ? <MonitoringPage onBack={() => setView('chat')} /> : <main className="chat-panel">
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
            <button className="memory-button" onClick={openMemory}>
              <BrainCircuit size={18} />
              <span>查看记忆</span>
              <ChevronRight size={15} />
            </button>
          </div>
        </header>

        <section className="messages" aria-live="polite" aria-busy={conversationLoading}>
          {conversationLoading && <div className="loading-state">正在载入对话…</div>}
          {!conversationLoading && messages.length === 0 && (
            <div className="welcome-state">
              <div className="welcome-orbit">
                <div className="welcome-logo"><Bot size={33} /></div>
              </div>
              <span className="welcome-kicker"><Sparkles size={14} /> 随时为你工作</span>
              <h2>今天想一起完成什么？</h2>
              <p>我能记住重要信息，也会保留每一次对话。直接输入问题，开始新的协作。</p>
              <div className="prompt-suggestions">
                <button onClick={() => setDraft('帮我梳理一下今天的工作计划')}>梳理今天的工作计划</button>
                <button onClick={() => setDraft('总结一下你对我的了解')}>总结你对我的了解</button>
                <button onClick={() => setDraft('和我一起分析一个问题')}>一起分析一个问题</button>
              </div>
            </div>
          )}

          {messages.map((message) => (
            <article key={message.id} className={`message-row ${message.role}`}>
              {message.role !== 'status' && message.role !== 'error' && message.role !== 'tool_group' && (
                <div className="message-avatar" aria-hidden="true">
                  {message.role === 'assistant' ? <Bot size={17} /> : <UserRound size={17} />}
                </div>
              )}
              <div className="message-content">
                {message.role === 'assistant' ? markdown(message.content) : message.role === 'tool_group'
                  ? <ToolActivityGroup tools={message.tools || []} />
                  : message.content}
              </div>
            </article>
          ))}
          {isSending && messages.at(-1)?.role === 'user' && (
            <div className="thinking" aria-label="助手正在思考"><span /><span /><span /></div>
          )}
          <div ref={messagesEndRef} />
        </section>

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
      </main>}

      {memoryOpen && (
        <>
          <div className="drawer-scrim" onClick={() => setMemoryOpen(false)} />
          <aside className="memory-drawer" aria-label="长期记忆" aria-modal="true">
            <div className="memory-header">
              <div>
                <span className="eyebrow">长期上下文</span>
                <h2>MyClaw 记忆</h2>
              </div>
              <div className="memory-header-actions">
                <button className="icon-button" onClick={() => void loadMemory()} aria-label="重新加载记忆">
                  <RefreshCw size={17} className={memoryLoading ? 'spin' : ''} />
                </button>
                <button className="icon-button" onClick={() => setMemoryOpen(false)} aria-label="关闭记忆面板">
                  <X size={19} />
                </button>
              </div>
            </div>

            <div className="memory-tabs" role="tablist" aria-label="记忆分类">
              {MEMORY_TABS.map((tab) => {
                const Icon = tab.icon
                return (
                  <button
                    key={tab.key}
                    role="tab"
                    aria-selected={memoryTab === tab.key}
                    className={memoryTab === tab.key ? 'active' : ''}
                    onClick={() => setMemoryTab(tab.key)}
                  >
                    <Icon size={16} />
                    {tab.label}
                  </button>
                )
              })}
            </div>

            <div className="memory-content" role="tabpanel">
              {memoryLoading && <div className="memory-placeholder">正在读取记忆…</div>}
              {memoryError && <div className="memory-placeholder error-text">{memoryError}</div>}
              {!memoryLoading && !memoryError && memory[memoryTab] && (
                <div className="memory-markdown">{markdown(memory[memoryTab])}</div>
              )}
              {!memoryLoading && !memoryError && !memory[memoryTab] && (
                <div className="memory-empty">
                  <CurrentMemoryIcon size={28} />
                  <h3>{currentMemoryTab.label}还是空的</h3>
                  <p>随着使用和记忆整理，这里会逐步积累相关内容。</p>
                </div>
              )}
            </div>
            <div className="memory-footer">只读视图 · 内容来自本地工作区</div>
          </aside>
        </>
      )}
    </div>
  )
}

function ToolActivityGroup({ tools }: { tools: ToolActivity[] }) {
  const running = tools.some((tool) => tool.state === 'running')
  return <details className="tool-activity">
    <summary>
      <Terminal size={15} />
      <span>{running ? '正在调用工具' : '已调用工具'} · {tools.length} 项</span>
      <small>{running ? '运行中' : '已完成'}</small>
    </summary>
    <div className="tool-activity-list">
      {tools.map((tool) => (
        <div className="tool-activity-item" key={tool.id}>
          <span className={tool.state} aria-label={tool.state === 'running' ? '运行中' : '已完成'} />
          <div><strong>{tool.name}</strong><pre><code>{formatToolInvocation(tool)}</code></pre></div>
        </div>
      ))}
    </div>
  </details>
}

export default App
