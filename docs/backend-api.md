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
  "chat_id": "direct",
  "accepted": true
}
```

Validation failures return `400` with `{ "error": "..." }`.

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

## Errors and methods

Unsupported methods return `405`; unknown paths return `404`. Error responses use:

```json
{ "error": "message" }
```
