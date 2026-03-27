"""Global configuration for sweep parallelism."""
import os

# Leave 4 cores for OS / IDE / browser — use the rest.
MAX_WORKERS = max(1, (os.cpu_count() or 1) - 4)
