"""Deterministic secret scrubber and timezone / schedule constants.

Ensures no credentials, tokens, account identifiers, PANs, or phone numbers
ever reach logs, stdout, reports, or LLM prompts, while preserving benign domain
keys such as `side`, `calendar_discrepancy_warnings`, `span`, and `candidates_considered`.
"""

from __future__ import annotations

import datetime as dt
import os
import re
import stat
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

# Settled 6 Nov 2026: Five approved scheduled run times per market day (IST)
APPROVED_RUN_TIMES_IST: tuple[dt.time, ...] = (
    dt.time(10, 15, tzinfo=IST),
    dt.time(11, 15, tzinfo=IST),
    dt.time(12, 15, tzinfo=IST),
    dt.time(13, 15, tzinfo=IST),
    dt.time(14, 15, tzinfo=IST),
)

# Short or ambiguous tokens matched ONLY as whole snake_case/camelCase identifier segments
# (prevents "sid" redacting "side"/"considered" or "pan" redacting "discrepancy"/"span").
_SENSITIVE_EXACT_SEGMENTS: frozenset[str] = frozenset(
    {
        "secret",
        "secrets",
        "token",
        "tokens",
        "password",
        "passwd",
        "pwd",
        "apikey",
        "sid",
        "account",
        "ucc",
        "pan",
        "phone",
        "mobile",
        "authorization",
        "bearer",
        "cookie",
        "cookies",
        "mpin",
        "totp",
        "otp",
    }
)

# Unambiguous multi-word key substrings
_SENSITIVE_KEY_PHRASES: tuple[str, ...] = (
    "api_key",
    "consumer_key",
    "access_key",
    "private_key",
    "secret_key",
    "session_token",
    "access_token",
    "refresh_token",
    "auth_token",
    "client_id",
    "account_id",
    "account_no",
    "account_number",
)

_IDENTIFIER_SPLIT_RE = re.compile(r"[^a-z0-9]+")

# Regex patterns for Indian PAN, 10-digit mobile numbers, Bearer tokens, and long hex/base64 tokens
_PAN_RE = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b")
_INDIAN_MOBILE_RE = re.compile(r"(?<![0-9])(?:\+91[\-\s]?)?[6-9][0-9]{9}(?![0-9])")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9\-._~+/]+=*")
_LONG_TOKEN_RE = re.compile(r"\b(?:sk-|ghp_|gho_|agy-h-|eyJ)[A-Za-z0-9\-._~+/]{10,}\b")


def is_sensitive_key_name(key: str) -> bool:
    """Return True if a dictionary key name represents a credential or PII field."""
    lowered = key.strip().lower()
    if not lowered:
        return False
    if any(phrase in lowered for phrase in _SENSITIVE_KEY_PHRASES):
        return True
    segments = {seg for seg in _IDENTIFIER_SPLIT_RE.split(lowered) if seg}
    return bool(segments & _SENSITIVE_EXACT_SEGMENTS)


def ensure_ist(timestamp: dt.datetime) -> dt.datetime:
    """Validate and convert any timezone-aware datetime to Asia/Kolkata (IST).

    Naive datetimes are rejected to enforce strict IST-awareness throughout.
    """
    if timestamp.tzinfo is None or timestamp.tzinfo.utcoffset(timestamp) is None:
        raise ValueError(f"Naive datetime is prohibited; must be IST-aware: {timestamp!r}")
    return timestamp.astimezone(IST)


def is_approved_schedule_slot(
    timestamp_ist: dt.datetime,
    tolerance_seconds: int = 300,
    holiday_dates: frozenset | set | None = None,
    special_sessions: dict | None = None,
) -> bool:
    """Return True if timestamp_ist is within an approved run window.

    Calendar-aware: a weekday listed in `holiday_dates` (verified NSE FO holiday calendar)
    is rejected. A weekend/holiday date is approved only when `special_sessions` maps its ISO
    date to (open_HH:MM, close_HH:MM) from a verified NSE circular, and then only within that
    actual session interval - never before the session opens.
    """
    ts = ensure_ist(timestamp_ist)
    d = ts.date().isoformat()
    if special_sessions and d in special_sessions:
        open_s, close_s = special_sessions[d]
        oh, om = (int(x) for x in open_s.split(":"))
        ch, cm = (int(x) for x in close_s.split(":"))
        open_dt = dt.datetime.combine(ts.date(), dt.time(oh, om), tzinfo=IST)
        close_dt = dt.datetime.combine(ts.date(), dt.time(ch, cm), tzinfo=IST)
        return open_dt <= ts <= close_dt
    if ts.weekday() >= 5:
        return False
    if holiday_dates and (d in holiday_dates or ts.date() in holiday_dates):
        return False
    for slot in APPROVED_RUN_TIMES_IST:
        slot_dt = dt.datetime.combine(ts.date(), slot.replace(tzinfo=None), tzinfo=IST)
        if abs((ts - slot_dt).total_seconds()) <= tolerance_seconds:
            return True
    return False


class SecretScrubber:
    """Deterministic scrubber removing secret values and PII from text and dicts."""

    def __init__(self, explicit_secrets: list[str] | None = None) -> None:
        self._explicit_secrets = [
            s
            for s in (explicit_secrets or [])
            if s and len(s.strip()) >= 4 and s != "PLACEHOLDER_DO_NOT_COMMIT"
        ]

    def scrub_text(self, text: str) -> str:
        if not text:
            return text
        out = text
        for secret in self._explicit_secrets:
            out = out.replace(secret, "[REDACTED_SECRET]")
        out = _BEARER_RE.sub("Bearer [REDACTED_TOKEN]", out)
        out = _LONG_TOKEN_RE.sub("[REDACTED_TOKEN]", out)
        out = _PAN_RE.sub("[REDACTED_PAN]", out)
        out = _INDIAN_MOBILE_RE.sub("[REDACTED_PHONE]", out)
        return out

    def scrub_obj(self, obj: Any) -> Any:
        """Recursively scrub a JSON-like object (dict, list, str, primitive)."""
        if isinstance(obj, str):
            return self.scrub_text(obj)
        if isinstance(obj, dict):
            cleaned: dict[str, Any] = {}
            for k, v in obj.items():
                key_str = str(k)
                if is_sensitive_key_name(key_str):
                    cleaned[key_str] = "[REDACTED_KEY_VALUE]"
                else:
                    cleaned[key_str] = self.scrub_obj(v)
            return cleaned
        if isinstance(obj, (list, tuple)):
            return [self.scrub_obj(item) for item in obj]
        return obj


def load_chmod600_env_file(file_path: str | Path, repo_root: Path | None = None) -> dict[str, str]:
    """Load key=value pairs from an external chmod-600 file.

    Raises PermissionError if the file is inside the repository or has group/other permissions.
    """
    path = Path(file_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Secret env file not found: {path}")

    if repo_root is not None:
        resolved_repo = repo_root.expanduser().resolve()
        try:
            path.relative_to(resolved_repo)
            raise PermissionError(
                f"Secret file {path} must reside OUTSIDE the repository ({resolved_repo})."
            )
        except ValueError:
            pass

    st = os.stat(path)
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o077:
        raise PermissionError(
            f"Secret file {path} has insecure permissions {oct(mode)}; required chmod 600 (0o600)."
        )

    env_vars: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        env_vars[key.strip()] = val.strip().strip("'").strip('"')
    return env_vars
