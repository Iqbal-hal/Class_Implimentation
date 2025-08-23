import sys
import io

class DualLogger:
    """
    Writes to both terminal (stdout) and a log file.
    Uses UTF-8 with replacement for unencodable characters to avoid
    UnicodeEncodeError when writing emojis or other non-CP1252 glyphs.
    """
    def __init__(self, logfile_path):
        self.terminal = sys.stdout
        # open file in append mode with utf-8 and replace errors
        self.log_file = open(logfile_path, "a", encoding="utf-8", errors="replace")

    def write(self, message):
        # write to terminal as-is
        try:
            self.terminal.write(message)
        except Exception:
            # best-effort: fallback to sys.__stdout__
            try:
                sys.__stdout__.write(message)
            except Exception:
                pass

        # write to log file (utf-8, with replacement already set on open)
        try:
            self.log_file.write(message)
        except Exception:
            # defensive fallback: ensure we write a safe string
            safe = message.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
            try:
                self.log_file.write(safe)
            except Exception:
                # last resort: write nothing
                pass

    def flush(self):
        try:
            self.terminal.flush()
        except Exception:
            pass
        try:
            self.log_file.flush()
        except Exception:
            pass

    def close(self):
        try:
            self.log_file.close()
        except Exception:
            pass




