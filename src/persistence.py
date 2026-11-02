"""Small persistence helpers for short Windows file-sharing locks."""
from pathlib import Path
import time


def replace_with_retry(source, target):
    """Atomically replace a file, tolerating a brief PermissionError only.

    Six attempts wait at most 0.3 seconds in total. Permanent permission errors
    and all other failures remain visible to the caller; cleanup stays theirs.
    """
    source = Path(source)
    for attempt in range(6):
        try:
            return source.replace(target)
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(.02 * (attempt + 1))
