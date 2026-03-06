# AI News WhatsApp Alert - Development Reference

## Project Overview
Automated daily AI news alert system that fetches AI-related news, ranks articles by importance using AI, and sends the top story to WhatsApp.

## Architecture
```
main.py                  # Entry point - orchestrates the workflow
src/
├── news_fetcher.py      # Fetches AI news from NewsAPI (last 24 hours)
├── news_ranker.py       # Ranks articles using GPT-4o-mini or simple fallback
└── whatsapp_sender.py   # Sends messages via Twilio WhatsApp API
.github/workflows/
└── daily-news.yml       # GitHub Actions - runs daily at 10:00 AM CST (16:00 UTC)
```

## Key Dependencies
- `requests` - NewsAPI HTTP calls
- `openai` - GPT-4o-mini for article ranking
- `twilio` - WhatsApp message delivery
- `python-dotenv` - Environment variable loading

## Environment Variables
All stored in `.env` (local) and GitHub Secrets (CI/CD):
- `NEWS_API_KEY` - from newsapi.org
- `TWILIO_ACCOUNT_SID` - Twilio console
- `TWILIO_AUTH_TOKEN` - Twilio console
- `TWILIO_WHATSAPP_NUMBER` - format: `whatsapp:+14155238886`
- `YOUR_WHATSAPP_NUMBER` - format: `whatsapp:+14083944615`
- `OPENAI_API_KEY` - optional, enables AI ranking

## Common Issues
1. **"Invalid From and To pair"** - WhatsApp numbers missing `whatsapp:` prefix or country code (`+1` for US)
2. **Messages not received** - User must join Twilio sandbox first by texting join code to Twilio number
3. **Missing modules** - Run `pip install -r requirements.txt`

## Testing
```bash
# Local test
python main.py

# Trigger GitHub Actions workflow
gh workflow run daily-news.yml

# Check workflow status
gh run list --limit 1

# View failed logs
gh run view <run-id> --log-failed
```

## News Search Query
Searches for: artificial intelligence, AI, machine learning, ChatGPT, OpenAI, Google AI, Claude, LLM

## Schedule
Daily at 10:00 AM CST (16:00 UTC) via GitHub Actions cron: `0 16 * * *`
