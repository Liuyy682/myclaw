# myclaw

Minimal personal assistant agent MVP.

Run one turn:

```bash
python -m myclaw "hello"
```

Run interactively:

```bash
python -m myclaw
```

Without `OPENAI_API_KEY`, the CLI uses a local fake provider so the MVP runs offline.
For a real OpenAI-compatible endpoint, create `.env` in the project root:

```env
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
```

Existing shell environment variables take priority over `.env` values.
