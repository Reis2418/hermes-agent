"""Generic opt-in fail-closed private inbound route policy (``gateway.fail_closed_consumer_routes``).

The default ``post_gateway_admission`` consumer is deliberately fail-open (#129958): a plugin that
is unloaded, raises, times out, is suppressed or simply absent lets the message proceed to the
ordinary agent turn. For a dedicated single-purpose bot that ingests PRIVATE data (an operator
registers exact ``{platform, scope_id, user_id}`` routes), fail-open leaks the message body and a
full transcript/agent turn to whatever else serves the profile. This module flips the semantics
for those routes ONLY, as an optional core prerequisite a deployment opts into — the plugin alone
is not fail-closed without this gate, and shipping it still needs separate deployment review.

Contract for a matched, authorized inbound message (checked against the ORIGINAL route even when
a prior ``pre_gateway_dispatch`` hook rewrote the event):

1. Offer ``post_gateway_admission`` BEFORE pending approvals, busy preemption/steering,
   command/slash dispatch, FIFO admission, transcript append and any agent/provider call.
2. If a callback returns ``{"action": "handled", "reply": ...}``, that reply is the answer.
3. Otherwise — hook absent, raises, times out or is suppressed by the bounded-worker path, or
   returns nothing handled, or the reply is blank — return the static refusal below. NEVER fall
   through to the agent.
4. A previously-admitted protected message replaying (startup replay or durable-FIFO execution) is
   refused without re-running the hook or the agent.

Routing-profile safety: under multiplexing the offer is made in the ROUTED profile's plugin scope
(same scope rule as the fail-open consumer). A routed profile with no registered plugin therefore
refuses its own matched traffic — safe by construction; no unsupported-multiplex overclaim.

Malformed config refuses gateway startup: every boot path builds the config through
``GatewayConfig.from_dict``/the dataclass constructor, and parsing raises ``ValueError`` here
(wildcards, blank/non-string IDs, missing keys, non-dict entries are all fatal — a half-effective
privacy policy is worse than none).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

#: Static refusal returned for every refused protected message. Deliberately content-free: it
#: confirms nothing about the private plugin's state and echoes nothing of the message.
FAIL_CLOSED_REFUSAL_MESSAGE = (
    "Private ingress could not confirm handling. Nothing was delivered to the ordinary agent. "
    "A private consumer may already have saved the message; check its state before retrying."
)

_REQUIRED_KEYS = ("platform", "scope_id", "user_id")


@dataclass(frozen=True)
class FailClosedRoute:
    """One exact-match private route: platform value + scope id + user id, all mandatory, non-empty."""

    platform: str
    scope_id: str
    user_id: str

    def describe(self) -> str:
        """Stable pseudonymous route label; never log a phone, guild or account ID."""
        scope = hashlib.sha256(self.scope_id.encode()).hexdigest()[:12]
        user = hashlib.sha256(self.user_id.encode()).hexdigest()[:12]
        return f"{self.platform}/{scope}/{user}"


def _exact_id(value: Any, key: str, index: int) -> str:
    """A mandatory nonempty, non-wildcard string ID; anything else refuses startup."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"gateway.fail_closed_consumer_routes[{index}].{key!r} must be a non-empty string "
            f"(exact match only; wildcards and blanks are refused)"
        )
    if "*" in value or "?" in value:
        raise ValueError(
            f"gateway.fail_closed_consumer_routes[{index}].{key!r} contains a wildcard; this is an "
            f"exact allowlist — every id must be concrete"
        )
    return value.strip()


def parse_fail_closed_consumer_routes(raw: Any) -> list[FailClosedRoute]:
    """Validate/coerce the operator list into ``FailClosedRoute``s; raises ``ValueError``.

    Absent → the dataclass default ``[]`` (no-op). An explicit None or wrong shape is malformed.
    Unknown keys are fatal too: a typo'd key (``chat_id``) would silently widen nothing, but it
    means the operator's intent was not parsed — fail loudly rather than half-enforce.
    """
    if raw == []:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            "gateway.fail_closed_consumer_routes must be a list of "
            "{platform, scope_id, user_id} dicts"
        )
    routes: list[FailClosedRoute] = []
    for index, entry in enumerate(raw):
        if isinstance(entry, FailClosedRoute):
            entry = {key: getattr(entry, key) for key in _REQUIRED_KEYS}
        if not isinstance(entry, dict):
            raise ValueError(
                f"gateway.fail_closed_consumer_routes[{index}] must be a dict with "
                f"{', '.join(_REQUIRED_KEYS)}"
            )
        extra = sorted(set(entry) - set(_REQUIRED_KEYS))
        if extra:
            raise ValueError(
                f"gateway.fail_closed_consumer_routes[{index}] has unsupported keys {extra}; "
                "the policy is an exact {platform, scope_id, user_id} allowlist"
            )
        platform = _exact_id(entry.get("platform"), "platform", index)
        try:
            from gateway.config import Platform

            platform = Platform(platform).value
        except ValueError:
            raise ValueError(
                f"gateway.fail_closed_consumer_routes[{index}].platform {platform!r} is not a "
                "known platform"
            ) from None
        routes.append(FailClosedRoute(
            platform=platform,
            scope_id=_exact_id(entry.get("scope_id"), "scope_id", index),
            user_id=_exact_id(entry.get("user_id"), "user_id", index),
        ))
    return routes


def source_route_identity(source: Any) -> tuple:
    """Freeze routing-critical values before a pre-dispatch hook sees the mutable source."""
    return (
        getattr(getattr(source, "platform", None), "value", None),
        getattr(source, "scope_id", None), getattr(source, "chat_id", None),
        getattr(source, "chat_type", None), getattr(source, "thread_id", None),
        getattr(source, "parent_chat_id", None),
        getattr(source, "user_id", None), getattr(source, "profile", None),
    )


def match_fail_closed_route(routes: list[FailClosedRoute], source: Any) -> Optional[FailClosedRoute]:
    """The exact route matching *source* (original identity), or None.

    ``scope_id`` matches the platform-neutral scope (Discord guild / Slack workspace / Signal chat)
    falling back to ``chat_id`` — DM-shaped platforms (WhatsApp, Telegram DMs) carry no guild, so
    the chat IS the scope. Every component must match exactly; a source with no user id never
    matches (an unattributable message is refused elsewhere, not silently widened).
    """
    if not routes or source is None:
        return None
    platform = getattr(getattr(source, "platform", None), "value", None)
    if not platform or not getattr(source, "user_id", None):
        return None
    scope = getattr(source, "scope_id", None) or getattr(source, "chat_id", None)
    for route in routes:
        if route.platform == platform and route.user_id == source.user_id and route.scope_id == scope:
            return route
    return None


def _refuse(reason: str, route: FailClosedRoute) -> tuple[bool, str]:
    """One log line (route identity + reason, never message content) and the static refusal."""
    logger.warning(
        "fail-closed private ingress: refused (%s) for route %s; the ordinary agent turn is "
        "blocked by gateway.fail_closed_consumer_routes",
        reason, route.describe(),
    )
    return True, FAIL_CLOSED_REFUSAL_MESSAGE


async def offer_fail_closed_admission(
    runner: Any, event: Any, source: Any, session_key: Optional[str], route: FailClosedRoute
) -> tuple[bool, Optional[str]]:
    """The fail-closed counterpart of ``run_post_admission_hook``: ``(handled, reply)`` where
    EVERY non-handled outcome — absent hook, raise, timeout/suppression, None/unhandled result,
    blank reply, replay, refused profile resolution — returns the static refusal. Only an
    affirmative ``{\"action\": \"handled\", \"reply\": str}`` returns the plugin's own reply."""
    from gateway.session_ingress import executing_admission
    if (executing_admission.get()
            or getattr(event, "_hermes_startup_restore_replay", False)
            or getattr(event, "_fifo_only", False)):
        # A previously admitted protected row or queued slash is being replayed.
        # Do not offer it a second time or let it reach the ordinary agent.
        return _refuse("replay of a previously-admitted protected message", route)

    try:
        results = await _offer_in_routed_scope(runner, event, source, session_key)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # health: allow BLE001 -- fail-CLOSED by design: any dispatch/setup error refuses the message
        logger.warning("fail-closed private ingress: hook dispatch failed (%s)",
                       type(exc).__name__)
        return _refuse("post_gateway_admission dispatch error", route)

    if results is None:
        return _refuse("no post_gateway_admission consumer registered in the routed profile", route)
    for result in results:
        if isinstance(result, dict) and result.get("action") == "handled":
            reply = result.get("reply")
            if isinstance(reply, str) and reply.strip():
                logger.info(
                    "fail-closed private ingress: plugin handled the message for route %s",
                    route.describe(),
                )
                return True, reply
            return _refuse("handled without a usable reply", route)
    # No affirmative handled result: the callback raised (isolated, dropped from results), was
    # skipped for timeout/suppression by the bounded-worker path, or declined. Fail closed.
    return _refuse("hook absent, raised, timed out, was suppressed, or returned nothing handled", route)


async def _offer_in_routed_scope(
    runner: Any, event: Any, source: Any, session_key: Optional[str]
) -> Optional[list]:
    """Fire ``post_gateway_admission`` under the message's routed profile scope.

    Returns the result list, or ``None`` when the scope has no consumer at all (so the caller can
    tell "absent" from "ran, nothing handled"). ``RuntimeStoreError``/resolution errors propagate
    to the caller and become refusals.
    """
    from gateway.run_inbound_consumer import _routed_scope
    async with _routed_scope(runner, source):
        from hermes_cli.lifecycle import ainvoke_hook, has_hook
        if not has_hook("post_gateway_admission"):
            return None
        return await ainvoke_hook(
            "post_gateway_admission",
            session_key=session_key,
            platform=source.platform.value if source.platform else "",
            source=source.to_dict(),
            message_id=event.message_id,
            text=event.text or "",
        )
