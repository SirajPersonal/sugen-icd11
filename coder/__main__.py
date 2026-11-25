"""``python -m coder IN.jsonl OUT.jsonl`` (the Docker entrypoint)."""
import sys

from .pipeline import main

sys.exit(main())
