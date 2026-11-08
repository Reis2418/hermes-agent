"""Generic opt-in fail-closed private inbound route policy (``gateway.fail_closed_consumer_routes``).

An exact-allowlist of {platform, scope_id, user_id} routes whose inbound messages are offered to
``post_gateway_admission`` BEFORE pending approvals, busy preemption/steering, command/slash
dispatch, FIFO admission and any agent turn. If the hook is absent, raises, times out, is
suppressed or returns nothing handled, the message gets a static refusal and NEVER falls through
to the ordinary agent. Absent config is a no-op; malformed config refuses gateway startup.

This is a core prerequisite for fail-closed private ingress plugins — the plugin alone is not
fail-closed without it. Not Journal-specific: no plugin import, no hardcoded profile.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.event import MessageEvent
from gateway.private_ingress import FAIL_CLOSED_REFUSAL_MESSAGE
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_constants import get_hermes_home

HOOK = "post_gateway_admission"

ROUTE = {"platform": "whatsapp", "scope_id": "155500000000000000@s.whatsapp.net",
         "user_id": "155500000000000000@s.whatsapp.net"}


def _register(callback) -> None:
    """Register *callback* on the ACTIVE home's plugin manager (what a loaded plugin does)."""
    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    manager._discovered = True  # no on-disk plugins in the test home; skip discovery
    manager._hooks.setdefault(HOOK, []).append(callback)


def _runner(*, routes=None, multiplex=False):
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig(
        platforms={Platform.WHATSAPP: PlatformConfig(enabled=True)},
        multiplex_profiles=multiplex,
        fail_closed_consumer_routes=routes if routes is not None else [],
    )
    adapter = SimpleNamespace(send=AsyncMock())
    runner.adapters = {Platform.WHATSAPP: adapter}
    runner.pairing_store = MagicMock()
    runner.session_store = MagicMock()
    runner._is_user_authorized_for_source = lambda _s: True
    runner._admit_bot_message_for_source = lambda _s: True
    runner._delivery_adapter_for = lambda _s: adapter
    runner._intake_adapter_for = lambda _s: adapter
    runner._rescue_orphaned_overflow = lambda _k, _a: None
    runner._enqueue_fifo = MagicMock()
    runner._handle_message_with_agent = AsyncMock(return_value="agent-ok")
    runner._run_post_turn_hooks = AsyncMock()
    runner._session_key_for_source = lambda _s: "agent:main:whatsapp:dm"
    return runner


def _event(text="private journal line", profile=None, scope_id=None, user_id=None,
           chat_id=None, chat_type="dm"):
    source = SessionSource(
        platform=Platform.WHATSAPP,
        user_id=user_id or ROUTE["user_id"],
        chat_id=chat_id or ROUTE["scope_id"],
        scope_id=scope_id if scope_id is not None else ROUTE["scope_id"],
        user_name="owner", chat_type=chat_type, profile=profile,
    )
    return MessageEvent(text=text, message_id="m1", source=source)


@pytest.fixture(autouse=True)
def _allow_all(monkeypatch):
    for key in ("WHATSAPP_ALLOWED_USERS", "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("WHATSAPP_ALLOWED_USERS", "*")


def _sealed_past_gate(runner):
    """Everything downstream of the fail-closed gate that could reach the agent must trip loudly.
    The estop gate sits above the fail-closed gate and only ever returns a pause notice (never an
    agent turn), so it stays a passthrough."""
    async def _aseal(*_a, **_k):
        raise AssertionError("private-ingress gate leaked past a downstream stage")

    def _seal(*_a, **_k):
        raise AssertionError("private-ingress gate leaked past a downstream stage")

    runner._hm_estop_gate = lambda *_a, **_k: None
    for name in ("_hm_pending_reply_intercepts", "_hm_authority_admission",
                 "_hm_handle_running_session_message", "_hm_dispatch_idle_commands"):
        setattr(runner, name, _aseal)
    runner._claim_active_session_slot = _seal
    runner._is_session_running = lambda _k: False
    # If the gate ever falls through, the real agent path is the loudest possible signal.
    runner._handle_message_with_agent = AsyncMock(side_effect=AssertionError(
        "fall-through to the agent on a protected route"))


# ── config validation: absent → no-op, malformed → refuse startup ──────────────

def test_absent_config_is_a_noop():
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    # The CLI registry and presence-sensitive gateway loader must agree:
    # absent means no policy; a registered empty list does not enable one.
    assert DEFAULT_CONFIG["gateway"]["fail_closed_consumer_routes"] == []
    cfg = GatewayConfig.from_dict({"platforms": {"whatsapp": {"enabled": True}}})
    assert cfg.fail_closed_consumer_routes == []


def test_valid_config_parses_to_exact_rules():
    cfg = GatewayConfig.from_dict({"fail_closed_consumer_routes": [dict(ROUTE)]})
    assert len(cfg.fail_closed_consumer_routes) == 1
    rule = cfg.fail_closed_consumer_routes[0]
    assert rule.platform == "whatsapp" and rule.scope_id == ROUTE["scope_id"]
    assert rule.user_id == ROUTE["user_id"]
    # Round-trips through to_dict for status/export without secrets.
    assert GatewayConfig.from_dict(cfg.to_dict()).fail_closed_consumer_routes == [rule]


def test_programmatic_route_object_cannot_bypass_validation():
    from gateway.private_ingress import FailClosedRoute
    with pytest.raises(ValueError):
        GatewayConfig(fail_closed_consumer_routes=[
            FailClosedRoute(platform="whatsapp", scope_id="*", user_id="owner")])


@pytest.mark.parametrize("bad", [
    "not-a-list", None, {}, False, 0, "",
    [{"platform": "whatsapp", "scope_id": "s", "user_id": ""}],        # empty id
    [{"platform": "whatsapp", "scope_id": "   ", "user_id": "u"}],      # whitespace-only id
    [{"platform": "whatsapp", "scope_id": "s"}],                        # missing user_id
    [{"platform": "*", "scope_id": "s", "user_id": "u"}],               # wildcard platform
    [{"platform": "whatsapp", "scope_id": "*", "user_id": "u"}],        # wildcard scope
    [{"platform": "whatsapp", "scope_id": "s", "user_id": "*"}],        # wildcard user
    ["whatsapp/s/u"],                                                    # not a dict
    [{"platform": 7, "scope_id": "s", "user_id": "u"}],                  # non-string id
    [{"platform": "whatsapp", "scope_id": "s", "user_id": "u", "chat_id": "x"}],  # typo'd key
])
def test_malformed_config_refuses_gateway_startup(bad):
    # GatewayConfig.from_dict is the single validation seam the boot path
    # (load_gateway_config -> GatewayRunner.__init__) uses: an exception there stops the gateway.
    with pytest.raises(ValueError):
        GatewayConfig.from_dict({"fail_closed_consumer_routes": bad})


def test_malformed_yaml_config_refuses_load_gateway_config(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    (home / "config.yaml").write_text(
        "gateway:\n  fail_closed_consumer_routes: oops\n", encoding="utf-8")
    from gateway.config import load_gateway_config
    from gateway.run import _profile_runtime_scope

    with _profile_runtime_scope(home), pytest.raises(ValueError):
        load_gateway_config()


@pytest.mark.parametrize("platform", ["local", "api_server", "webhook", "relay", "msgraph_webhook"])
def test_non_messaging_ingress_platform_refuses_privacy_policy(platform):
    # These surfaces do not offer a normal external _handle_message turn
    # before native admission or agent execution; accepting a rule would
    # incorrectly claim that its text is protected by this gateway gate.
    with pytest.raises(ValueError, match="unsupported"):
        GatewayConfig.from_dict({"fail_closed_consumer_routes": [
            {"platform": platform, "scope_id": "exact-scope", "user_id": "exact-user"}]})


@pytest.mark.asyncio
async def test_secondary_profile_private_route_policy_is_rejected_at_load(tmp_path, monkeypatch):
    """Only the default gateway owns the route match; a secondary policy may not silently no-op."""
    from gateway.run import MultiplexConfigError
    from gateway import run_startup
    from hermes_cli import env_loader

    home = tmp_path / "h"
    secondary = home / "profiles" / "beta"
    secondary.mkdir(parents=True)
    (home / "config.yaml").write_text("gateway:\n  multiplex_profiles: true\n")
    (secondary / "config.yaml").write_text(
        "gateway:\n  fail_closed_consumer_routes:\n"
        f"    - platform: {ROUTE['platform']}\n"
        f"      scope_id: {ROUTE['scope_id']}\n"
        f"      user_id: {ROUTE['user_id']}\n"
    )
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(env_loader, "hydrate_profile_secret_sources", lambda _h: None)
    monkeypatch.setattr(run_startup, "recover_left_core_in", lambda *_a, **_kw: None)
    runner = object.__new__(GatewayRunner)
    runner._subscribe_plugin_rewire = MagicMock()
    runner._register_config_hooks = MagicMock()
    runner._snapshot_profile_busy_modes = MagicMock()

    with pytest.raises(MultiplexConfigError, match="default profile"):
        await runner._load_secondary_profile_config("beta", secondary)


# ── gate ordering + fail-closed outcomes ───────────────────────────────────────

@pytest.mark.asyncio
async def test_plugin_present_and_handled_returns_plugin_reply_before_everything_else():
    _register(lambda **_k: {"action": "handled", "reply": "journal plugin took it"})
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)

    out = await runner._handle_message(_event())
    assert out == "journal plugin took it"
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_hook_absent_refuses_and_never_falls_through():
    runner = _runner(routes=[dict(ROUTE)])  # nothing registered for the hook
    _sealed_past_gate(runner)

    out = await runner._handle_message(_event())
    assert out == FAIL_CLOSED_REFUSAL_MESSAGE
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_hook_raising_refuses_instead_of_failing_open():
    def broken(**_k):
        raise RuntimeError("plugin bug")

    _register(broken)
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)

    assert await runner._handle_message(_event()) == FAIL_CLOSED_REFUSAL_MESSAGE
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_hook_returning_none_or_unhandled_refuses():
    _register(lambda **_k: None)
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)
    assert await runner._handle_message(_event()) == FAIL_CLOSED_REFUSAL_MESSAGE

    _register(lambda **_k: {"action": "ignored"})
    runner2 = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner2)
    assert await runner2._handle_message(_event("second")) == FAIL_CLOSED_REFUSAL_MESSAGE


@pytest.mark.asyncio
async def test_pre_dispatch_skip_does_not_log_private_reason_or_route(caplog):
    from hermes_cli.plugins import get_plugin_manager
    private_text = "my-private-journal-synthetic-token-409"
    manager = get_plugin_manager()
    manager._discovered = True
    manager._hooks.setdefault("pre_gateway_dispatch", []).append(
        lambda **_kw: {"action": "skip", "reason": private_text})
    runner = _runner(routes=[dict(ROUTE)])
    with caplog.at_level("INFO"):
        assert await runner._handle_message(_event(private_text)) is None
    assert private_text not in caplog.text
    assert ROUTE["user_id"] not in caplog.text
    assert ROUTE["scope_id"] not in caplog.text


@pytest.mark.asyncio
async def test_callback_partial_write_error_does_not_lie_or_log_private_text(caplog):
    private_text = "my-private-journal-synthetic-token-408"
    writes = []
    def committed_then_failed(**kw):
        writes.append(kw["text"])
        raise RuntimeError("callback failed after saving " + kw["text"])
    _register(committed_then_failed)
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)
    with caplog.at_level("DEBUG"):
        reply = await runner._handle_message(_event(private_text))
    assert writes == [private_text]
    assert reply == FAIL_CLOSED_REFUSAL_MESSAGE
    assert "may already" in reply
    assert "not processed" not in reply
    assert private_text not in caplog.text
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_hook_handled_without_reply_still_refuses_privately():
    _register(lambda **_k: {"action": "handled"})
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)
    assert await runner._handle_message(_event()) == FAIL_CLOSED_REFUSAL_MESSAGE


@pytest.mark.asyncio
async def test_callback_timeout_suppression_refuses(monkeypatch):
    """The bounded-worker path SKIPS a timed-out/suppressed callback: either an empty result
    list (skip) or a raised asyncio.TimeoutError (caller-side wait_for cancel). Both must refuse,
    where the default pipeline would fail open to the agent."""
    from hermes_cli import lifecycle

    real = lifecycle.ainvoke_hook
    outcomes = []

    async def timeout_dispatch(name, **kwargs):
        if name == HOOK:
            outcomes.append("timed-out")
            raise TimeoutError()  # the awaited bounded call is cancelled/times out
        return await real(name, **kwargs)

    _register(lambda **_k: None)  # a consumer IS registered; only its dispatch times out
    monkeypatch.setattr(lifecycle, "ainvoke_hook", timeout_dispatch)
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)
    assert await runner._handle_message(_event()) == FAIL_CLOSED_REFUSAL_MESSAGE
    assert outcomes == ["timed-out"]

    # And the pure-skip shape: hook "ran", collected no result (suppressed) -> refusal, no fall-through.
    async def suppressed_dispatch(name, **kwargs):
        if name == HOOK:
            return []
        return await real(name, **kwargs)

    monkeypatch.setattr(lifecycle, "ainvoke_hook", suppressed_dispatch)
    runner2 = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner2)
    assert await runner2._handle_message(_event("second")) == FAIL_CLOSED_REFUSAL_MESSAGE


@pytest.mark.asyncio
async def test_private_slash_text_never_reaches_shared_command_metrics(monkeypatch):
    from hermes_cli.observability import shared_metrics_events
    observed = []
    monkeypatch.setattr(shared_metrics_events, "record_gateway_slash_command",
                        lambda event: observed.append(event.text))
    _register(lambda **_kw: {"action": "handled", "reply": "plugin"})
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)
    assert await runner._handle_message(_event("/privateword")) == "plugin"
    assert observed == []


@pytest.mark.asyncio
async def test_busy_and_command_content_still_gate_first():
    """A slash-looking message and a plain approval-shaped 'yes' from a matched route hit the
    gate BEFORE command dispatch and busy preemption."""
    _register(lambda **_k: {"action": "handled", "reply": "plugin"})
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)

    assert await runner._handle_message(_event("/approve 12")) == "plugin"
    assert await runner._handle_message(_event("yes")) == "plugin"


@pytest.mark.asyncio
async def test_pre_dispatch_cannot_skip_private_gate_by_setting_route_rejected():
    from hermes_cli.plugins import get_plugin_manager
    manager = get_plugin_manager()
    manager._discovered = True
    def mark_rejected(**kw):
        kw["event"].source.profile_route_rejected = True
        return {"action": "allow"}
    manager._hooks.setdefault("pre_gateway_dispatch", []).append(mark_rejected)
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)
    assert await runner._handle_message(_event()) == FAIL_CLOSED_REFUSAL_MESSAGE
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_pre_dispatch_rewrite_cannot_escape_the_route():
    """Even when a pre_gateway_dispatch hook rewrote the event, the ORIGINAL source route match
    holds and the fail-closed offer still happens."""
    from hermes_cli import lifecycle
    from gateway.session_ingress_context import carry_callback

    fired = []
    _register(lambda **kw: fired.append(kw["text"]) or {"action": "handled", "reply": "plugin"})

    real = lifecycle.ainvoke_hook

    async def patched(name, **kwargs):
        if name == "pre_gateway_dispatch":
            # Rewrite the text in place on the event the way the real hook chain does.
            kwargs["event"].text = "REWRITTEN"
            return [{"action": "allow"}]
        return await real(name, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(lifecycle, "ainvoke_hook", patched)
    try:
        runner = _runner(routes=[dict(ROUTE)])
        _sealed_past_gate(runner)
        out = await runner._handle_message(_event("original private text"))
        assert out == "plugin"
        assert fired == ["REWRITTEN"]  # plugin sees the rewritten text; the route still matched
    finally:
        monkeypatch.undo()


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["scope_id", "user_id", "profile", "thread_id",
                                     "parent_chat_id", "chat_type", "source"])
async def test_original_protected_route_cannot_escape_source_rewrite(changed):
    from hermes_cli.plugins import get_plugin_manager
    calls = []
    manager = get_plugin_manager()
    manager._discovered = True
    def rewrite_source(event, **_kwargs):
        if changed == "source":
            from dataclasses import replace
            event.source = replace(event.source, scope_id="other-scope")
        else:
            setattr(event.source, changed, "other-scope")
        return {"action": "allow"}
    manager._hooks.setdefault("pre_gateway_dispatch", []).append(rewrite_source)
    _register(lambda **_kw: calls.append("offered") or
              {"action": "handled", "reply": "plugin"})
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)
    out = await runner._handle_message(_event())
    assert out == FAIL_CLOSED_REFUSAL_MESSAGE
    assert calls == []
    runner._handle_message_with_agent.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_slash_replay_refuses_without_offering_private_hook():
    calls = []
    _register(lambda **_kw: calls.append("offered") or
              {"action": "handled", "reply": "plugin"})
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)
    event = _event("/approve token")
    vars(event)["_fifo_only"] = True  # replay marker is set dynamically by FIFO dispatch
    assert await runner._handle_message(event) == FAIL_CLOSED_REFUSAL_MESSAGE
    assert calls == []


@pytest.mark.asyncio
async def test_startup_replay_refuses_without_rerunning_hook_or_agent():
    _register(lambda **_k: pytest.fail("replay must not re-offer the hook"))
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)
    event = _event()
    event._hermes_startup_restore_replay = True
    assert await runner._handle_message(event) == FAIL_CLOSED_REFUSAL_MESSAGE


@pytest.mark.asyncio
async def test_executing_admission_replay_refuses_without_rerunning_hook():
    from gateway.session_ingress import executing_admission

    _register(lambda **_k: pytest.fail("executing admission must not re-offer the hook"))
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)
    token = executing_admission.set(True)
    try:
        assert await runner._handle_message(_event()) == FAIL_CLOSED_REFUSAL_MESSAGE
    finally:
        executing_admission.reset(token)


# ── non-matching traffic keeps default behavior ────────────────────────────────

@pytest.mark.asyncio
async def test_unmatched_user_on_same_scope_is_ordinary_agent_traffic():
    _register(lambda **_k: None)  # inert plugin: default fail-open path must reach the agent
    runner = _runner(routes=[dict(ROUTE)])
    out = await runner._handle_message(_event(user_id="999@s.whatsapp.net",
                                              chat_id="999@s.whatsapp.net",
                                              scope_id="999@s.whatsapp.net"))
    assert out == "agent-ok"


@pytest.mark.asyncio
async def test_wrong_scope_on_the_same_user_refuses_nothing_and_reaches_agent():
    runner = _runner(routes=[dict(ROUTE)])
    out = await runner._handle_message(_event(scope_id="111-other@s.whatsapp.net",
                                              chat_id="111-other@s.whatsapp.net"))
    assert out == "agent-ok"


@pytest.mark.asyncio
async def test_matching_route_outside_the_expected_category_still_refuses():
    """Dedicated single-purpose instance: an exact (platform, scope, user) match outside the
    plugin's own chat/category is ALSO refused — documented safety trade-off."""
    runner = _runner(routes=[dict(ROUTE)])
    _sealed_past_gate(runner)
    event = _event("random chatter", chat_type="group")
    assert await runner._handle_message(event) == FAIL_CLOSED_REFUSAL_MESSAGE


@pytest.mark.asyncio
async def test_no_routes_configured_keeps_existing_fail_open_behavior():
    def broken(**_k):
        raise RuntimeError("plugin bug")

    _register(broken)
    runner = _runner(routes=[])  # policy absent: untouched v0.23 fail-open semantics

    assert await runner._handle_message(_event()) == "agent-ok"


# ── routed-profile safety ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multiplex_matched_route_offers_hook_in_routed_profile_scope(tmp_path):
    launch, routed = tmp_path / "launch", tmp_path / "profiles" / "beta"
    launch.mkdir()
    routed.mkdir(parents=True)
    fired = []

    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(routed):
        _register(lambda **_k: fired.append(get_hermes_home()) or
                  {"action": "handled", "reply": "beta plugin"})

    runner = _runner(routes=[dict(ROUTE)], multiplex=True)
    runner._resolve_profile_home_for_source = lambda _source: routed
    _sealed_past_gate(runner)
    with _profile_runtime_scope(launch):
        out = await runner._handle_message(_event(profile="beta"))
    assert out == "beta plugin"
    assert fired and fired[0] == routed  # offered in the ROUTED profile's plugin scope


@pytest.mark.asyncio
async def test_multiplex_unavailable_hook_in_routed_scope_refuses(tmp_path):
    launch, routed = tmp_path / "launch", tmp_path / "profiles" / "beta"
    launch.mkdir()
    routed.mkdir(parents=True)

    runner = _runner(routes=[dict(ROUTE)], multiplex=True)
    runner._resolve_profile_home_for_source = lambda _source: routed
    _sealed_past_gate(runner)
    from gateway.run import _profile_runtime_scope
    with _profile_runtime_scope(launch):
        # beta has no post_gateway_admission plugin registered: refuse, never run the agent.
        assert await runner._handle_message(_event(profile="beta")) == FAIL_CLOSED_REFUSAL_MESSAGE
    runner._handle_message_with_agent.assert_not_awaited()
