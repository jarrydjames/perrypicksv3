from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests


def post_message(
    webhook_url: str,
    *,
    content: Optional[str] = None,
    embeds: Optional[List[Dict[str, Any]]] = None,
    username: Optional[str] = None,
) -> None:
    payload: Dict[str, Any] = {}
    if content:
        payload["content"] = content
    if embeds:
        payload["embeds"] = embeds
    if username:
        payload["username"] = username

    response = requests.post(webhook_url, json=payload, timeout=10)
    response.raise_for_status()
