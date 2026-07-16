# MyClaw Gateway API

The Gateway is a local HTTP service started with:

```bash
python -m myclaw gateway
```

The default base URL is `http://127.0.0.1:8765`. All JSON responses use UTF-8.

## Send a message

`POST /api/messages`

Request body:

```json
{
  "content": "Hello",
  "chat_id": "direct",
  "session_key": "gateway:direct"
}
```

- `content` is required and must contain non-whitespace text.
- `chat_id` is optional and defaults to `direct`. It selects the SSE stream that receives the reply.
- `session_key` is optional. When present, the turn continues that saved session; when absent, the Gateway uses `gateway:<chat_id>`.

Accepted response (`202`):

```json
{
  "id": "request-id",
  "trace_id": "32-character-trace-id",
  "chat_id": "direct",
  "accepted": true
}
```

Validation failures return `400` with `{ "error": "..." }`. `trace_id`
correlates the accepted request, SSE events, structured logs, and execution trace.

## Stream events

`GET /api/events?chat_id=direct`

The response is an SSE stream (`text/event-stream`). Each event is sent as one `data:` line containing JSON:

```json
{
  "type": "message_delta",
  "id": "request-id",
  "chat_id": "direct",
  "content": "partial text",
  "terminal": false,
  "metadata": {
    "request_id": "request-id",
    "trace_id": "32-character-trace-id",
    "session_key": "gateway:direct"
  }
}
```

Supported event types:

| Type | Meaning | Terminal |
| --- | --- | --- |
| `message_delta` | Incremental assistant text | No |
| `message` | Complete assistant reply | Yes |
| `tool_progress` | Progress emitted while a tool runs | No |
| `control` | Response to a control command | Usually yes |
| `error` | Turn or runtime error | Yes |

Clients must connect with the same `chat_id` used by `POST /api/messages`.

## List sessions

`GET /api/sessions`

Response (`200`):

```json
{
  "sessions": [
    {
      "key": "gateway:direct",
      "channel": "gateway",
      "title": "Example conversation",
      "preview": "First message",
      "created_at": "2026-07-14T10:00:00",
      "updated_at": "2026-07-14T10:01:00",
      "message_count": 2
    }
  ]
}
```

Sessions are ordered by `updated_at`, newest first.

## Read a session

`GET /api/sessions?key=gateway%3Adirect`

Response (`200`):

```json
{
  "key": "gateway:direct",
  "title": "Example conversation",
  "messages": [
    { "role": "user", "content": "Hello" },
    { "role": "assistant", "content": "Hi" }
  ]
}
```

Only user and assistant messages with non-empty content are returned. A missing key returns `404`.

## Read long-term memory

`GET /api/memory`

Response (`200`):

```json
{
  "memory": "# Memory\n...",
  "user": "# User\n...",
  "soul": "# Soul\n..."
}
```

The fields correspond to `MEMORY.md`, `USER.md`, and `SOUL.md` in the active workspace memory directory. Missing files are represented by empty strings. The endpoint is read-only and does not expose compacted `history.jsonl` entries.

## Observability

Observability data is metadata-only by default and is stored in
`observability/observability.db` under the active workspace.

### Summary

`GET /api/observability/summary?window=24h`

Returns request totals, running/error counts, success rate, P50/P95 duration,
P95 queue wait, LLM/tool counts, provider-supplied token totals, and an hourly
trend series. `window` accepts positive hour/day values up to 30 days, such as
`1h`, `24h`, or `7d`.

### Traces

`GET /api/observability/traces`

Optional filters are `window`, `status`, `kind`, `session_key`, `before`, and
`limit`. The default limit is 50 and the maximum is 200. The response contains
`traces` plus `next_before` for pagination.

`GET /api/observability/traces/{trace_id}` returns one root `trace`, its ordered
`spans`, and correlated `logs`. A missing trace returns `404`.

### Logs

`GET /api/observability/logs`

Optional filters are `window`, `level`, `component`, `trace_id`, `query`,
`before`, and `limit`. The default limit is 200 and the maximum is 500.
Observability endpoints are read-only and return `503` when collection is disabled.

## Errors and methods

Unsupported methods return `405`; unknown paths return `404`. Error responses use:

```json
{ "error": "message" }
```
