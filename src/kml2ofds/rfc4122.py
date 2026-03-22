"""
RFC 4122 UUID validation for optional network identifiers.
"""

from __future__ import annotations

import uuid

NETWORK_ID_INVALID_MESSAGE = (
    "Network ID must be a valid UUID"
    "Leave the field empty to auto-generate one."
)


def network_id_validation_error(network_id: str | None) -> str | None:
    """
    If network_id is empty or whitespace, return None (caller may auto-generate).

    If non-empty, return None when the value is an RFC 4122 UUID string,
    otherwise return NETWORK_ID_INVALID_MESSAGE.
    """
    if network_id is None:
        return None
    s = network_id.strip()
    if not s:
        return None
    try:
        parsed = uuid.UUID(s)
    except ValueError:
        return NETWORK_ID_INVALID_MESSAGE
    if parsed.variant != uuid.RFC_4122:
        return NETWORK_ID_INVALID_MESSAGE
    return None
