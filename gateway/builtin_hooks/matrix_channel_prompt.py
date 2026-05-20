"""Inject channel_prompts from config.yaml into Matrix room messages.

The Matrix adapter doesn't natively call resolve_channel_prompt()
(unlike Slack, Mattermost, etc.). This hook fills that gap by reading
channel_prompts from the Matrix platform config and injecting them
into the event's channel_prompt field before dispatch.

Enabled by adding to config.yaml:
  hermes:
    enabled_hooks:
      - matrix_channel_prompt

And configure prompts under:
  matrix:
    channel_prompts:
      "!room:id:server": |
        your prompt here
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def pre_gateway_dispatch(event: dict, config: dict) -> dict:
    """Inject channel_prompt for Matrix group messages."""

    source = event.get("source", {})
    platform = source.get("platform")
    chat_type = source.get("chat_type")
    chat_id = source.get("chat_id")

    if platform != "matrix":
        return event
    if chat_type != "group":
        return event
    if not chat_id:
        return event

    # Read channel_prompts from the matrix platform config.
    matrix_cfg = config.get("matrix", {}) if isinstance(config, dict) else {}
    prompts = matrix_cfg.get("channel_prompts", {})
    if not isinstance(prompts, dict):
        return event

    prompt = prompts.get(chat_id)
    if not prompt:
        return event

    prompt_str = str(prompt).strip()
    if not prompt_str:
        return event

    logger.debug(
        "matrix_channel_prompt: injecting prompt for %s (%d chars)",
        chat_id,
        len(prompt_str),
    )
    event["channel_prompt"] = prompt_str
    return event
