# myclaw

Minimal personal assistant agent MVP.

Run one turn:

```bash
python -m myclaw "hello"
python -m myclaw --session work "hello"
```

Run interactively:

```bash
python -m myclaw
```

Interactive commands:

- `/resume` lists CLI sessions with their generated titles.
- `/resume <name>` switches to the named session, creating it if needed.
- `/new` creates and switches to a new session.
- `/clear` clears the current session history.
- `/status` shows whether the current session is idle, running, or queued.
- `/stop` cancels the current in-flight turn and keeps recovery checkpoint state.

Interactive mode starts in a new generated session unless `--session` is provided.

Without `OPENAI_API_KEY`, the CLI uses a local fake provider so the MVP runs offline.
For a real OpenAI-compatible endpoint, create `.env` in the project root:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

Existing shell environment variables take priority over `.env` values.
