import { useState } from 'react'
import ChatPage from '../features/chat/ChatPage'
import MonitoringPage from '../features/monitoring/MonitoringPage'

export default function App() {
  const [view, setView] = useState<'chat' | 'monitor'>('chat')

  if (view === 'monitor') {
    return (
      <div className="app-shell">
        <MonitoringPage onBack={() => setView('chat')} />
      </div>
    )
  }

  return <ChatPage onShowMonitoring={() => setView('monitor')} />
}
