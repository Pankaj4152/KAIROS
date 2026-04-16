import os

from dotenv import load_dotenv

load_dotenv()



# Normalizer
LOCAL_USER_ID = "local"  # since we only have one user, we can hardcode this

# Session
# A session with no activity for this long is considered expired.
SESSION_TIMEOUT   = 60 * 60   # seconds before an inactive session expires
PURGE_PROBABILITY = 0.02      # 2% chance of running cleanup on each resolve call


# Telegram
TELEGRAM_MAX_LENGTH = 4096  # Telegram's message character limit
TYPING_REFRESH_SECS = 2     # how often to refresh Telegram typing indicators (must be <5s)
MIN_MESSAGE_INTERVAL = 1.0  # ignore duplicate/rapid messages within this window (seconds)


# Logging
LOG_FORMAT       = "%(asctime)s  %(name)-20s  %(levelname)-7s  %(message)s"
LOG_DATE_FORMAT  = "%Y-%m-%d %H:%M:%S"
LOG_MAX_BYTES    = 5 * 1024 * 1024   # 5 MB per log file before rotation
LOG_BACKUP_COUNT = 3                 # keep 3 rotated backups


# Classifier
CLASSIFIER_TIMEOUT = float(os.getenv("CLASSIFIER_TIMEOUT", "10"))


# Orchestrator / tool-loop resilience
# TOOL_CALL_MAX_RETRIES = int(os.getenv("TOOL_CALL_MAX_RETRIES", "2"))
# TOOL_CALL_RETRY_BASE_DELAY = float(os.getenv("TOOL_CALL_RETRY_BASE_DELAY", "0.5"))
# TOOL_CIRCUIT_BREAKER_FAILURE_THRESHOLD = int(os.getenv("TOOL_CIRCUIT_BREAKER_FAILURE_THRESHOLD", "3"))
# TOOL_CIRCUIT_BREAKER_COOLDOWN_SECONDS = int(os.getenv("TOOL_CIRCUIT_BREAKER_COOLDOWN_SECONDS", "60"))
MAX_TOOL_ROUNDS = int(os.getenv("MAX_TOOL_ROUNDS", "5"))