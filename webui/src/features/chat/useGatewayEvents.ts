import { useCallback, useEffect, useRef, useState } from 'react'
import { openEventStream } from '../../shared/api/gateway'
import type { ConnectionState, GatewayEvent, ToolActivity, ViewMessage } from '../../shared/types/gateway'

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

export function useGatewayEvents(chatId: string, refreshSessions: () => Promise<void>) {
  const [messages, setMessages] = useState<ViewMessage[]>([])
  const [isSending, setIsSending] = useState(false)
  const [connection, setConnection] = useState<ConnectionState>('connecting')
  const pendingAskRequestIdsRef = useRef(new Set<string>())
  const answeredAskRequestIdsRef = useRef(new Set<string>())

  const handleGatewayEvent = useCallback((event: GatewayEvent) => {
    const requestId = event.id || event.metadata?.request_id || chatId
    const assistantId = `assistant:${requestId}`
    const moveAssistantAfterAnswer = answeredAskRequestIdsRef.current.has(requestId)

    if (event.type === 'message_delta') {
      setMessages((current) => {
        const index = current.findIndex((item) => item.id === assistantId)
        if (index < 0) return [...current, { id: assistantId, role: 'assistant', content: event.content }]
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
      setMessages((current) => [...current, {
        id: `${event.type}:${requestId}:${current.length}`,
        role,
        content: event.content,
      }])
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
  }, [chatId, refreshSessions])

  useEffect(() => {
    setConnection('connecting')
    return openEventStream(chatId, {
      onEvent: handleGatewayEvent,
      onOpen: () => setConnection('connected'),
      onError: () => {
        setConnection('reconnecting')
        setIsSending(false)
      },
    })
  }, [chatId, handleGatewayEvent])

  const markPendingAsAnswered = useCallback(() => {
    for (const requestId of pendingAskRequestIdsRef.current) answeredAskRequestIdsRef.current.add(requestId)
  }, [])

  return { connection, isSending, markPendingAsAnswered, messages, setIsSending, setMessages }
}
