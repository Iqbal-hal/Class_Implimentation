import sys
import io

class DualLogger:
    """
    Quick fix version: Writes to both terminal (stdout) and a log file using UTF-8.
    Handles console encoding issues by providing ASCII fallbacks for emojis.
    """
    def __init__(self, logfile_path):
        # keep a reference to the original stdout for terminal writes
        self.terminal = sys.stdout
        # open file in append mode with utf-8 and replace errors
        self.log_file = open(logfile_path, "a", encoding="utf-8", errors="replace")
        
    def _console_safe(self, message):
        """Convert emojis and special characters to console-safe alternatives when needed."""
        # If console supports UTF, keep emojis as-is
        enc = getattr(self.terminal, 'encoding', None)
        if isinstance(enc, str) and enc.lower().startswith('utf'):
            return message

        # Fallback replacements for non-UTF consoles
        replacements = {
            '📊': '[CHART]', '💰': '[MONEY]', '📈': '[UP]', '📉': '[DOWN]',
            '🎯': '[TARGET]', '🚀': '[ROCKET]', '💎': '[DIAMOND]', '✅': '[OK]',
            '⚠️': '[WARNING]', '🎉': '[PARTY]', '💹': '[TRADING]', '💵': '[CASH]',
            '📝': '[NOTE]', '🖥️': '[SCREEN]', '🔥': '[HOT]', '⭐': '[STAR]',
            '₹': 'Rs.', '€': 'EUR', '$': 'USD', '£': 'GBP',
            '→': '->', '←': '<-', '↑': 'UP', '↓': 'DOWN',
            '•': '*', '◦': 'o', '■': '[BLOCK]', '□': '[BOX]',
            '…': '...', '–': '-', '—': '--', '≥': '>=', '≤': '<=', '±': '+/-'
        }

        safe_message = message
        for emoji, replacement in replacements.items():
            safe_message = safe_message.replace(emoji, replacement)
        return safe_message

    def write(self, message):
        # Write to terminal with emoji replacement for console compatibility
        try:
            console_message = self._console_safe(message)
            self.terminal.write(console_message)
        except UnicodeEncodeError:
            # Fallback: encode with error handling
            try:
                safe = message.encode('ascii', errors='replace').decode('ascii')
                self.terminal.write(safe)
            except Exception:
                try:
                    sys.__stdout__.write(str(message))
                except Exception:
                    pass
        except Exception:
            # Final fallback to original stdout
            try:
                sys.__stdout__.write(str(message))
            except Exception:
                pass

        # Write to log file (preserve original UTF-8 with emojis)
        try:
            self.log_file.write(message)
        except Exception:
            # defensive fallback: ensure we write a safe string
            try:
                safe = message.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                self.log_file.write(safe)
            except Exception:
                # last resort: ignore the write error
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