import os

MAX_WORKERS = max(1, os.cpu_count() - 4)
