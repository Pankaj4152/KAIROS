

# Normalizer
LOCAL_USER_ID = "local"  # since we only have one user, we can hardcode this

# Session
# A session with no activity for this long is considered expired.
SESSION_TIMEOUT   = 60 * 60   # seconds before an inactive session expires
PURGE_PROBABILITY = 0.02      # 2% chance of running cleanup on each resolve call


# Telegram
TELEGRAM_MAX_LENGTH = 4096  # Telegram's message character limit
TYPING_REFRESH_SECS = 4     # how often to refresh Telegram typing indicators (must be <5s)
MIN_MESSAGE_INTERVAL = 1.0  # ignore duplicate/rapid messages within this window (seconds)