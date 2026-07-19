import type { RefObject } from 'react'
import { Bot, Sparkles, Terminal, UserRound } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import type { ToolActivity, ViewMessage } from '../../shared/types/gateway'

type Props = {
  endRef: RefObject<HTMLDivElement>
  isSending: boolean
  loading: boolean
  messages: ViewMessage[]
  onSuggestion: (value: string) => void
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
      {tools.map((tool) => {
        const invocation = tool.name === 'exec' && typeof tool.arguments.cmd === 'string'
          ? tool.arguments.cmd
          : `${tool.name}${Object.keys(tool.arguments).length ? ` ${JSON.stringify(tool.arguments, null, 2)}` : ''}` || tool.fallback
        return <div className="tool-activity-item" key={tool.id}>
          <span className={tool.state} aria-label={tool.state === 'running' ? '运行中' : '已完成'} />
          <div><strong>{tool.name}</strong><pre><code>{invocation}</code></pre></div>
        </div>
      })}
    </div>
  </details>
}

export default function MessageList({ endRef, isSending, loading, messages, onSuggestion }: Props) {
  return <section className="messages" aria-live="polite" aria-busy={loading}>
    {loading && <div className="loading-state">正在载入对话…</div>}
    {!loading && messages.length === 0 && (
      <div className="welcome-state">
        <div className="welcome-orbit"><div className="welcome-logo"><Bot size={33} /></div></div>
        <span className="welcome-kicker"><Sparkles size={14} /> 随时为你工作</span>
        <h2>今天想一起完成什么？</h2>
        <p>我能记住重要信息，也会保留每一次对话。直接输入问题，开始新的协作。</p>
        <div className="prompt-suggestions">
          <button onClick={() => onSuggestion('帮我梳理一下今天的工作计划')}>梳理今天的工作计划</button>
          <button onClick={() => onSuggestion('总结一下你对我的了解')}>总结你对我的了解</button>
          <button onClick={() => onSuggestion('和我一起分析一个问题')}>一起分析一个问题</button>
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
          {message.role === 'assistant'
            ? <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>{message.content}</ReactMarkdown>
            : message.role === 'tool_group'
              ? <ToolActivityGroup tools={message.tools || []} />
              : message.content}
        </div>
      </article>
    ))}
    {isSending && messages.at(-1)?.role === 'user' && (
      <div className="thinking" aria-label="助手正在思考"><span /><span /><span /></div>
    )}
    <div ref={endRef} />
  </section>
}
