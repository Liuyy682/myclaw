import { useCallback, useEffect, useRef, useState } from 'react'
import { BrainCircuit, RefreshCw, Sparkles, UserRound, X } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { getMemory } from '../../shared/api/gateway'
import type { MemoryPayload } from '../../shared/types/gateway'

const EMPTY_MEMORY: MemoryPayload = { memory: '', user: '', soul: '' }
const MEMORY_TABS = [
  { key: 'memory' as const, label: '项目记忆', icon: BrainCircuit },
  { key: 'user' as const, label: '用户信息', icon: UserRound },
  { key: 'soul' as const, label: '助手设定', icon: Sparkles },
]

export default function MemoryDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const [memory, setMemory] = useState<MemoryPayload>(EMPTY_MEMORY)
  const [memoryTab, setMemoryTab] = useState<keyof MemoryPayload>('memory')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const closeButtonRef = useRef<HTMLButtonElement>(null)
  const previouslyFocusedRef = useRef<HTMLElement | null>(null)
  const onCloseRef = useRef(onClose)
  onCloseRef.current = onClose

  const loadMemory = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      setMemory(await getMemory())
    } catch (requestError) {
      setError(requestError instanceof Error ? requestError.message : String(requestError))
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => {
    if (open && memory === EMPTY_MEMORY && !loading) void loadMemory()
  }, [loadMemory, loading, memory, open])

  useEffect(() => {
    if (!open) return
    previouslyFocusedRef.current = document.activeElement instanceof HTMLElement ? document.activeElement : null
    const focusable = 'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        onCloseRef.current()
        return
      }
      if (event.key !== 'Tab') return
      const elements = Array.from(document.querySelectorAll<HTMLElement>(`.memory-drawer ${focusable}`))
        .filter((element) => !element.hasAttribute('disabled'))
      if (elements.length === 0) return
      const first = elements[0]
      const last = elements[elements.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first.focus()
      }
    }
    document.addEventListener('keydown', handleKeyDown)
    const frame = requestAnimationFrame(() => closeButtonRef.current?.focus())
    return () => {
      cancelAnimationFrame(frame)
      document.removeEventListener('keydown', handleKeyDown)
      previouslyFocusedRef.current?.focus()
      previouslyFocusedRef.current = null
    }
  }, [open])

  if (!open) return null
  const currentTab = MEMORY_TABS.find((tab) => tab.key === memoryTab)!
  const CurrentIcon = currentTab.icon

  return <>
    <div className="drawer-scrim" onClick={onClose} />
    <aside className="memory-drawer" role="dialog" aria-label="长期记忆" aria-modal="true">
      <div className="memory-header">
        <div><span className="eyebrow">长期上下文</span><h2>MyClaw 记忆</h2></div>
        <div className="memory-header-actions">
          <button className="icon-button" onClick={() => void loadMemory()} aria-label="重新加载记忆">
            <RefreshCw size={17} className={loading ? 'spin' : ''} />
          </button>
          <button ref={closeButtonRef} className="icon-button" onClick={onClose} aria-label="关闭记忆面板"><X size={19} /></button>
        </div>
      </div>

      <div className="memory-tabs" role="tablist" aria-label="记忆分类">
        {MEMORY_TABS.map((tab) => {
          const Icon = tab.icon
          return <button
            key={tab.key}
            role="tab"
            aria-selected={memoryTab === tab.key}
            className={memoryTab === tab.key ? 'active' : ''}
            onClick={() => setMemoryTab(tab.key)}
          ><Icon size={16} />{tab.label}</button>
        })}
      </div>

      <div className="memory-content" role="tabpanel">
        {loading && <div className="memory-placeholder">正在读取记忆…</div>}
        {error && <div className="memory-placeholder error-text">{error}</div>}
        {!loading && !error && memory[memoryTab] && (
          <div className="memory-markdown">
            <ReactMarkdown remarkPlugins={[remarkGfm]} skipHtml>{memory[memoryTab]}</ReactMarkdown>
          </div>
        )}
        {!loading && !error && !memory[memoryTab] && (
          <div className="memory-empty">
            <CurrentIcon size={28} />
            <h3>{currentTab.label}还是空的</h3>
            <p>随着使用和记忆整理，这里会逐步积累相关内容。</p>
          </div>
        )}
      </div>
      <div className="memory-footer">只读视图 · 内容来自本地工作区</div>
    </aside>
  </>
}
