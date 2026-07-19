import { Activity, Bot, MessageSquare, PanelLeftClose, Plus, RefreshCw } from 'lucide-react'
import type { ConnectionState, SessionSummary } from '../../shared/types/gateway'

type Props = {
  activeSessionKey: string | null
  connection: ConnectionState
  connectionLabel: string
  error: string
  loading: boolean
  onClose: () => void
  onNewConversation: () => void
  onRefresh: () => Promise<void>
  onSelect: (session: SessionSummary) => Promise<void>
  onShowMonitoring: () => void
  open: boolean
  sessions: SessionSummary[]
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

export default function SessionSidebar(props: Props) {
  return (
    <aside className={`sidebar ${props.open ? 'open' : ''}`} aria-label="会话导航">
      <div className="brand">
        <div className="brand-mark"><Bot size={21} strokeWidth={2.2} /></div>
        <div><strong>MyClaw</strong><span>个人智能助手</span></div>
        <button className="icon-button mobile-only" onClick={props.onClose} aria-label="关闭会话栏">
          <PanelLeftClose size={19} />
        </button>
      </div>

      <button className="new-chat-button" onClick={props.onNewConversation}>
        <Plus size={18} />新建对话
      </button>
      <button className="monitor-nav-button" onClick={() => { props.onShowMonitoring(); props.onClose() }}>
        <Activity size={18} />运行监控
      </button>

      <div className="sidebar-section-title">
        <span>最近对话</span>
        <button className="icon-button subtle" onClick={() => void props.onRefresh()} aria-label="刷新会话" title="刷新会话">
          <RefreshCw size={15} />
        </button>
      </div>

      <nav className="session-list" aria-label="历史会话">
        {props.loading && <div className="sidebar-note">正在载入会话…</div>}
        {props.error && <div className="sidebar-note error-text">{props.error}</div>}
        {!props.loading && !props.error && props.sessions.length === 0 && (
          <div className="sidebar-empty"><MessageSquare size={22} /><span>还没有历史对话</span></div>
        )}
        {props.sessions.map((session) => (
          <button
            key={session.key}
            className={`session-card ${props.activeSessionKey === session.key ? 'active' : ''}`}
            onClick={() => void props.onSelect(session)}
            aria-current={props.activeSessionKey === session.key ? 'page' : undefined}
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
        <span className={`connection-dot ${props.connection}`} />
        <div><strong>{props.connectionLabel}</strong><span>{props.activeSessionKey || '等待第一条消息'}</span></div>
      </div>
    </aside>
  )
}
