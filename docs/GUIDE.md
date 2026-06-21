# User Guide

> Everything Kairos can do for you — features, use cases, and example prompts.

**← [Back to README](../README.md)** · [Tool Guide](TOOL_GUIDE.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Contributing](CONTRIBUTING.md) · [Diagrams HTML](architecture.html) · [Diagrams PDF](kairos_architecture.pdf)

> For exact parameters, actions, and failure modes of every tool mentioned below, see the [Tool Guide](TOOL_GUIDE.md).

---

## Channels

### Telegram Bot
Your always-on interface. Works over mobile data, in noisy environments, or when you can't speak.

- **Authenticated** — locked to your numeric Telegram user ID
- **Typing indicator** — shows "typing..." while Kairos thinks
- **Long response splitting** — splits at paragraph boundaries, never mid-sentence
- **Reminders mid-conversation** — proactive pushes via the `send_message` tool

<p align="center">
  <img src="images/telegram_bot_start.png" alt="Telegram bot" width="400">
</p>

### Web UI
Browser-based chat at `http://localhost:8000`.

- **Streaming** — tokens arrive as they're generated via WebSocket
- **Any device** — works from phone, tablet, or laptop on your LAN
- **Remote access** — use [Tailscale](https://tailscale.com) for secure access from anywhere

<p align="center">
  <img src="images/home.png" alt="Kairos WebUI" width="600">
</p>

### Email Channel
- **Proactive morning briefing** — daily HTML briefings containing your pending tasks and events are delivered directly to your email inbox via Gmail SMTP
- **Verification diagnostics** — connection checks run at application boot to ensure the SMTP endpoint is configured properly

---

## Features

### 📋 Task Management

Add, view, update, and complete tasks with priority and due dates.

```
"Add a task to finish the API integration by Friday"
"What tasks do I have?"
"What's my highest priority task?"
"Mark the deployment task as done"
"Add a high priority task: review PR #42"
"Move my DSA practice task to next Monday"
```

Tasks live in SQLite (title, due_date, status, project, priority). When you ask about work or to-dos, open tasks are automatically injected into the prompt context, and the `tasks` tool gives the LLM full create/update/complete/search/stats capability mid-conversation.

<p align="center">
  <img src="images/add_task.png" alt="Adding a task" width="600">
</p>

<p align="center">
  <img src="images/ask_tasks.png" alt="Querying tasks" width="600">
</p>

---

### 📅 Calendar & Events

Manage your schedule with full event support, synced to your real Google Calendar.

```
"What do I have tomorrow?"
"Add a meeting with the team at 3pm on Thursday"
"Search for any dentist appointments on my calendar"
"Change my meeting tomorrow to be at 4pm"
"Delete my dental appointment next Tuesday"
"What calendars do I have access to?"
```

Backed by the `google_calendar` tool: list, search, create, update, delete events, and list all calendars you have access to. Invite emails are sent automatically when you ask Kairos to add attendees.

---

### 🔥 Habit Tracking

Track daily habits with streak counting, backfilling, and a 30-day consistency view.

```
"Did I work out today?"
"Mark gym as done"
"How's my reading streak?"
"What habits am I tracking?"
"Mark DSA practice as done"
"I forgot to log meditation yesterday, add it"
"Show me my stats for gym"
```

Each habit tracks: name, last completion date, current streak, target frequency (daily, weekdays, 3x/week, 5x/week, weekly), and a full checkin log used to compute streaks accurately even after backfills or corrections.

---

### 💰 Spending Tracker

Log expenses, set budgets, and view category breakdowns — manual logging only, no bank/UPI sync.

```
"I spent 500 on groceries at BigBasket"
"How much have I spent on food this month?"
"Log 1200 for rent"
"What's my spending breakdown?"
"Set my food budget to 5000 for this month"
"Am I over budget on food?"
```

Entries include amount, category, merchant, date, and notes. The `spending` tool also supports budgets (per-category or overall, per month, with over-budget/approaching-limit warnings) and period reports (`this_month`, `last_month`, `last_7_days`, `last_30_days`, or an explicit month).

---

### 🔍 Web Search

Real-time web search with pluggable backends, retry/backoff, and a circuit breaker per backend.

```
"What's the latest news on GPT-5?"
"Search for best laptops under 80k"
"What's the weather in Delhi?"
"Who won the match yesterday?"
```

Switch backends in `.env` — DuckDuckGo (free, default), Brave, Tavily, or Serper. See [Tool Guide → web_search](tools/web_search.md) for full config and failure-mode details.

<p align="center">
  <img src="images/chat_search.png" alt="Web search results" width="600">
</p>

---

### 📅 Google Calendar
Synchronize and manage your primary Google Calendar directly through natural text commands. See [Calendar & Events](#-calendar--events) above and [Tool Guide → google_calendar](tools/google_calendar.md).

---

### 📧 Gmail — Reading

Read unread email headers, search your inbox, and pull full message bodies via secure IMAP.

```
"Do I have any unread emails?"
"Check my inbox alerts"
"What are the subjects of my latest unread emails?"
"Search my inbox for the GitHub deployment failure"
"Read that email about the invoice"
```

The `check_gmail` tool scans recent unread emails and returns sender, subject, and date without downloading attachments or bloated bodies. Pulling a full body or marking something read is a follow-up call using the UID from the first result. Requires `GMAIL_USER` and `GMAIL_APP_PASSWORD` in `.env`.

---

### ✉️ Gmail — Sending, Replying & Organizing

Compose, reply, forward, delete, archive, move, and manage your inbox — the write-side companion to inbox reading, using the same Gmail App Password.

```
"Reply to that email and say I'll review it tomorrow"
"Reply all to the project thread"
"Forward the invoice to my accountant"
"Send an email to alice@example.com about the meeting moving to 4pm"
"Move that email to spam"
"Archive the newsletter emails"
"Delete that promotional email"
"Save a draft thanking them, I'll send it later"
"What folders/labels do I have in Gmail?"
```

Powered by `gmail_actions`. Reply/reply-all/forward preserve proper email threading (`In-Reply-To`/`References` headers) automatically. **Permanent deletion is irreversible** — Kairos should always confirm with you before using it; regular `delete` moves to Trash and is recoverable. See [Tool Guide → gmail_actions](tools/gmail_actions.md) for the full action list.

---

### 📈 Finance

Stock, ETF, and cryptocurrency pricing and history — no API key required (Yahoo Finance).

```
"What's Apple's stock price?"
"How's Bitcoin doing today?"
"Show me Tesla's price history for the last month"
"What's Reliance Industries trading at?"
"Find the ticker for Infosys"
```

Returns current price, change vs. previous close, day's range, volume, 52-week range, moving averages, market cap, P/E ratio, and dividend yield for the `quote` action; OHLCV candle history for `history`; ticker lookup for `search`. Supports US tickers, Indian NSE/BSE stocks (`.NS`/`.BO` suffix), and crypto (auto-converted to `-USD` pairs). See [Tool Guide → finance](tools/finance.md).

---

### ☀️ Weather

Current conditions, daily forecasts, and hour-by-hour outlooks for any location worldwide — no API key required (Open-Meteo).

```
"What's the weather like right now?"
"Will it rain tomorrow in Mumbai?"
"Give me a 7-day forecast for New Delhi"
"What's the hourly forecast for tonight?"
"Should I bring an umbrella?"
```

Returns temperature, feels-like, humidity, wind (speed/direction/gusts), UV index, visibility, and precipitation for current conditions; min/max temp, condition, precipitation chance, and sunrise/sunset for forecasts. Supports metric and imperial units. See [Tool Guide → weather](tools/weather.md).

---

### 🗒️ Notes

Deliberately saved, titled, taggable notes — distinct from the automatic conversation memory below.

```
"Note this: use Redis for the hot-data cache"
"Save a note titled 'Meeting recap' with what we just discussed"
"Search my notes for caching"
"What notes do I have tagged 'kairos'?"
"Find notes related to speeding up reads"  (semantic search)
"Link that note to my API integration task"
```

Supports fast keyword search (FTS5) and meaning-based semantic search (reuses the same embedding pipeline as conversation memory), plus optional linking to a task or habit. Use notes when you want something explicitly saved and recallable by title or tag — use plain conversation when you just want Kairos to remember something said in passing. See [Tool Guide → notes](tools/notes.md).

---

### 🧠 Semantic Memory

Kairos remembers past conversations via vector embeddings — separate from the explicit `notes` tool above.

```
"What did I say about the project last week?"
"Remember that I prefer dark mode"
"What was that restaurant name I mentioned?"
"What did we discuss about the API?"
```

Every conversation turn is embedded and stored. When relevant, past turns are retrieved via cosine similarity and injected into the prompt. Trivial chitchat is skipped to avoid memory pollution.

---

### ☀️ Morning Briefing

Automated daily briefing sent directly to your email inbox.

- Summarises your open tasks and upcoming events
- Composed by the LLM — natural, concise, energetic
- Delivered as a clean HTML email via Gmail SMTP
- Configurable time: `BRIEFING_HOUR` and `BRIEFING_MINUTE` in `.env`
- Timezone-aware (defaults to `Asia/Kolkata`)

---

### 💬 Conversation Continuity

Kairos maintains context within and across sessions.

- Last 8 turns included in every prompt
- Per-session history stored as JSON files
- Sessions auto-compact at 20 turns (configurable)
- Your profile and preferences are prepended to every single prompt

---

## Use Cases

### 🎓 Student / Learner
- Track DSA practice streaks and study habits
- Manage assignment deadlines with priority levels
- Morning briefings on what's due today
- Search for coding concepts, documentation, research papers
- Log daily study sessions as habits
- Track spending on courses and books, save reference notes

### 💼 Professional / Builder
- Manage project tasks across multiple repos
- Get reminded about meetings and deadlines
- Quick web search during deep work (without leaving your editor)
- Track spending on tools and infrastructure
- Triage and reply to email without opening Gmail
- Check stock/crypto prices for a quick portfolio glance

### 🏋️ Health & Routine
- Track gym sessions, diet, and meditation as habits
- Morning briefing includes routine reminders
- Streak tracking keeps you accountable
- Ask about your progress over time

### 🧠 Personal Knowledge
- Ask Kairos anything — it remembers past conversations
- Build a personal knowledge base through natural interaction and explicit notes
- Recall facts, decisions, and discussions from weeks ago
- "What did I decide about X?" actually works

### 📱 On-the-go
- Message your Telegram bot from anywhere
- Full task/event/email management from your phone
- Check weather and finance without switching apps
- Works over mobile data — no need for Wi-Fi

---

## Tips

- **Keep `profile.md` under 2KB** — it's prepended to every prompt. Bloating it wastes tokens and slows down responses.
- **Use Telegram for quick tasks** — "add task: review PR" is faster than opening a browser.
- **Check your streaks daily** — "how are my habits?" gives you a quick accountability check.
- **Let the classifier do its job** — you don't need to specify complexity. Just ask naturally.
- **Morning briefings work best when you actually have tasks and events** — seed your database with real data.
- **Always confirm before permanent deletes** — Kairos should ask before calling `gmail_actions` with `delete_permanent`; if it doesn't, say so explicitly.
- **Notes vs. memory** — say "note this" or "save this" when you want something explicitly titled and searchable by tag; otherwise, just mention things naturally and semantic memory picks it up.