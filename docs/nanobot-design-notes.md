# Nanobot Design Notes For The MVP

This MVP keeps only the part of nanobot that matters for a first working
personal assistant: the main run loop. It intentionally leaves tools, channels,
message buses, memory compaction, subagents, and web UI for later.

## Useful Nanobot Ideas

- Nanobot separates product orchestration from execution. `AgentLoop` owns
  sessions, channels, tools, and persistence, while `AgentRunner` owns the
  repeatable model loop. The MVP mirrors this by keeping `Agent` focused on
  history and run flow, and `LLMProvider` focused on model I/O.
- Conversation messages are the source of truth. Nanobot builds provider input
  from saved history, appends model/tool results, then persists the new turn.
  The MVP uses the same shape with OpenAI-compatible messages:
  `system`, `user`, and `assistant`.
- Each turn follows a small loop: build context, request the model, decide
  whether the turn is finished, append the result. For MVP, there are no tool
  calls, so one model response finishes the turn.
- Errors should become visible conversation state. Nanobot preserves partial
  turn context around interruptions; the MVP keeps the triggering user message
  and appends a readable assistant error.

## MVP Boundaries

- No channel abstraction. The CLI calls `Agent.run()` directly.
- No tools. `max_turns` exists as an extension point but defaults to one model
  call.
- No durable session manager. History lives in the `Agent` instance.
- No third-party runtime dependency. The OpenAI-compatible provider uses the
  Python standard library.
