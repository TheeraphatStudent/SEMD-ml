from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Optional
from urllib.parse import SplitResult, urlsplit, urlunsplit

import idna


_ALLOWED_SCHEMES = {"http", "https"}
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9._~-]+$")
_DEFAULT_PORTS = {"http": 80, "https": 443}


@dataclass(frozen=True)
class NormalizedURL:
    raw_url: str
    normalized_url: Optional[str]
    scheme: Optional[str]
    hostname: Optional[str]
    port: Optional[int]
    had_explicit_port: bool
    path: str
    query: str
    registered_domain: Optional[str]
    is_valid: bool
    error: Optional[str] = None


def normalize_url(url: object) -> NormalizedURL:
    raw = "" if url is None else str(url)
    stripped = raw.strip()
    if not stripped:
        return NormalizedURL(
            raw_url=raw,
            normalized_url=None,
            scheme=None,
            hostname=None,
            port=None,
            had_explicit_port=False,
            path="",
            query="",
            registered_domain=None,
            is_valid=False,
            error="empty_url",
        )

    candidate = stripped if "://" in stripped else f"http://{stripped}"

    try:
        parsed = urlsplit(candidate)
    except ValueError as exc:
        return _invalid(raw, f"parse_error:{exc}")

    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        return _invalid(raw, f"unsupported_scheme:{scheme or 'missing'}")

    try:
        hostname = parsed.hostname
        port = parsed.port
        had_explicit_port = parsed.port is not None
    except ValueError as exc:
        return _invalid(raw, f"invalid_port:{exc}")

    if not hostname:
        return _invalid(raw, "missing_hostname")

    hostname = hostname.strip().rstrip(".")
    if not hostname:
        return _invalid(raw, "missing_hostname")

    try:
        ascii_hostname = idna.encode(hostname, uts46=True).decode("ascii").lower()
    except idna.IDNAError as exc:
        return _invalid(raw, f"invalid_hostname:{exc}")

    if not _is_valid_hostname(ascii_hostname):
        return _invalid(raw, "invalid_hostname")

    netloc = ascii_hostname
    if port is not None and _DEFAULT_PORTS.get(scheme) != port:
        netloc = f"{ascii_hostname}:{port}"

    path = parsed.path or ""
    query = parsed.query or ""

    normalized = urlunsplit(
        SplitResult(
            scheme=scheme,
            netloc=netloc,
            path=path,
            query=query,
            fragment="",
        )
    )

    return NormalizedURL(
        raw_url=raw,
        normalized_url=normalized,
        scheme=scheme,
        hostname=ascii_hostname,
        port=port,
        had_explicit_port=had_explicit_port,
        path=path,
        query=query,
        registered_domain=extract_registered_domain(ascii_hostname),
        is_valid=True,
        error=None,
    )


def extract_registered_domain(hostname: Optional[str]) -> Optional[str]:
    if not hostname:
        return None
    if _looks_like_ip(hostname):
        return hostname

    labels = [label for label in hostname.lower().split(".") if label]
    if len(labels) <= 2:
        return ".".join(labels) or None

    common_second_level = {
        "ac.uk",
        "co.id",
        "co.il",
        "co.in",
        "co.jp",
        "co.kr",
        "co.nz",
        "co.th",
        "co.uk",
        "com.au",
        "com.br",
        "com.cn",
        "com.mx",
        "gov.uk",
        "net.au",
        "org.au",
        "org.uk",
    }
    suffix = ".".join(labels[-2:])
    if suffix in common_second_level and len(labels) >= 3:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def _invalid(raw: str, error: str) -> NormalizedURL:
    return NormalizedURL(
        raw_url=raw,
        normalized_url=None,
        scheme=None,
        hostname=None,
        port=None,
        had_explicit_port=False,
        path="",
        query="",
        registered_domain=None,
        is_valid=False,
        error=error,
    )


def _is_valid_hostname(hostname: str) -> bool:
    if len(hostname) > 253:
        return False
    if _looks_like_ip(hostname):
        return True
    labels = hostname.split(".")
    return all(
        label
        and len(label) <= 63
        and not label.startswith("-")
        and not label.endswith("-")
        and _HOSTNAME_RE.match(label)
        for label in labels
    )


def _looks_like_ip(hostname: str) -> bool:
    parts = hostname.split(".")
    if len(parts) != 4:
        return False
    for part in parts:
        if not part.isdigit():
            return False
        value = int(part)
        if value < 0 or value > 255:
            return False
    return True
