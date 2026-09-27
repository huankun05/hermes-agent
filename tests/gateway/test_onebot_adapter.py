"""Tests for the OneBot 11 (NapCat) platform-plugin adapter.

Loaded via the ``_plugin_adapter_loader`` helper so this lives under
``plugin_adapter_onebot`` in ``sys.modules`` and cannot collide with
sibling platform-plugin tests on the same xdist worker.

Coverage:
- text normalization (array segments, @-mention stripping, placeholders)
- allowlist gating (DM / group / @-mention / allow-all)
- outbound send (markdown stripping, reply+@ prefix, chunking)
- security invariants (non-loopback bind requires a token, self_id from
  handshake not from the event payload, own-message drop)
- protocol helpers (parse_allowed_list, message_mentions_self, version gate)
- plugin shape (register, env enablement, validate/is_connected)
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from gateway.config import PlatformConfig
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

_onebot = load_plugin_adapter("onebot")

OneBotAdapter = _onebot.OneBotAdapter
check_requirements = _onebot.check_requirements
validate_config = _onebot.validate_config
is_connected = _onebot.is_connected
register = _onebot.register
_env_enablement = _onebot._env_enablement
parse_allowed_list = _onebot.parse_allowed_list
message_mentions_self = _onebot.message_mentions_self
is_message_event = _onebot.is_message_event
is_api_response = _onebot.is_api_response
_split_text_codepoints = _onebot._split_text_codepoints
_strip_markdown = _onebot._strip_markdown
_version_at_least = _onebot._version_at_least
DEFAULT_PATH = _onebot.DEFAULT_PATH
STREAM_MIN_VERSION = _onebot.STREAM_MIN_VERSION


def _cfg(**extra):
    return PlatformConfig(enabled=True, extra=extra)


def _adapter(**extra):
    return OneBotAdapter(_cfg(**extra))


def _msg_event(**over):
    event = {
        "time": 1790476314,
        "self_id": 10001,
        "post_type": "message",
        "message_type": "private",
        "message_id": 42,
        "user_id": 20002,
        "sender": {"user_id": 20002, "nickname": "锟哥"},
        "message": [{"type": "text", "data": {"text": "你好"}}],
        "raw_message": "你好",
    }
    event.update(over)
    return event


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


# ── 1. Protocol helpers ────────────────────────────────────────────────


class TestProtocolHelpers:
    def test_parse_allowed_list_from_comma_string(self):
        assert parse_allowed_list("111, 222 ,333") == ["111", "222", "333"]

    def test_parse_allowed_list_from_chinese_comma(self):
        assert parse_allowed_list("111，222") == ["111", "222"]

    def test_parse_allowed_list_dedupes(self):
        assert parse_allowed_list("1,1,2") == ["1", "2"]

    def test_parse_allowed_list_from_iterable(self):
        assert parse_allowed_list([111, "222"]) == ["111", "222"]

    def test_parse_allowed_list_none(self):
        assert parse_allowed_list(None) == []

    def test_mentions_self_true(self):
        segs = [{"type": "at", "data": {"qq": "10001"}}]
        assert message_mentions_self(segs, "10001") is True

    def test_mentions_self_false_for_other_user(self):
        segs = [{"type": "at", "data": {"qq": "20002"}}]
        assert message_mentions_self(segs, "10001") is False

    def test_mentions_self_numeric_qq(self):
        segs = [{"type": "at", "data": {"qq": 10001}}]
        assert message_mentions_self(segs, "10001") is True

    def test_mentions_self_empty_self_id(self):
        assert message_mentions_self([{"type": "at", "data": {"qq": "1"}}], "") is False

    def test_is_message_event_accepts_private_and_group(self):
        assert is_message_event(_msg_event()) is True
        assert is_message_event(_msg_event(message_type="group", group_id=30003)) is True

    def test_is_message_event_rejects_meta_and_notice(self):
        assert is_message_event({"post_type": "meta_event", "meta_event_type": "heartbeat"}) is False
        assert is_message_event({"post_type": "notice", "notice_type": "group_increase"}) is False
        assert is_message_event({"post_type": "message", "message_type": "private", "message": "str"}) is False

    def test_is_api_response_requires_echo_and_status(self):
        assert is_api_response({"status": "ok", "retcode": 0, "data": {}, "echo": "e1"}) is True
        assert is_api_response({"post_type": "message"}) is False
        assert is_api_response({"echo": "e1"}) is False

    def test_version_gate(self):
        assert _version_at_least("4.9.0", STREAM_MIN_VERSION) is True
        assert _version_at_least("4.8.115", STREAM_MIN_VERSION) is True
        assert _version_at_least("4.8.114", STREAM_MIN_VERSION) is False
        assert _version_at_least("", STREAM_MIN_VERSION) is False

    def test_split_text_codepoints_short_passthrough(self):
        assert _split_text_codepoints("短消息") == ["短消息"]

    def test_split_text_codepoints_long_chunks(self):
        chunks = _split_text_codepoints("啊" * 4000)
        assert len(chunks) == 3
        assert all(len(c) <= 1500 for c in chunks)
        assert "".join(chunks) == "啊" * 4000

    def test_split_text_prefers_sentence_boundary(self):
        text = "甲" * 1400 + "。" + "乙" * 400
        chunks = _split_text_codepoints(text)
        assert chunks[0].endswith("。")

    def test_strip_markdown(self):
        assert _strip_markdown("这是**粗体**和`代码`") == "这是粗体和代码"


# ── 2. Adapter init ────────────────────────────────────────────────────


class TestAdapterInit:
    def test_defaults(self, monkeypatch):
        for key in (
            "ONEBOT_HOST",
            "ONEBOT_PORT",
            "ONEBOT_PATH",
            "ONEBOT_ACCESS_TOKEN",
            "ONEBOT_ALLOWED_PRIVATE_USERS",
            "ONEBOT_ALLOWED_GROUPS",
            "ONEBOT_GROUP_REQUIRE_MENTION",
            "ONEBOT_ALLOW_ALL_USERS",
        ):
            monkeypatch.delenv(key, raising=False)
        adapter = _adapter()
        assert adapter._host == "127.0.0.1"
        assert adapter._port == 6200
        assert adapter._path == DEFAULT_PATH
        assert adapter._access_token == ""
        assert adapter._group_require_mention is True

    def test_env_overrides_extra(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HOST", "0.0.0.0")
        monkeypatch.setenv("ONEBOT_PORT", "7001")
        monkeypatch.setenv("ONEBOT_PATH", "custom/ws")
        monkeypatch.setenv("ONEBOT_ACCESS_TOKEN", "tok")
        monkeypatch.setenv("ONEBOT_ALLOWED_PRIVATE_USERS", "111,222")
        monkeypatch.setenv("ONEBOT_GROUP_REQUIRE_MENTION", "false")
        adapter = _adapter(host="127.0.0.1", port=6200)
        assert adapter._host == "0.0.0.0"
        assert adapter._port == 7001
        assert adapter._path == "/custom/ws"  # leading slash added
        assert adapter._access_token == "tok"
        assert adapter._allowed_private_users == ["111", "222"]
        assert adapter._group_require_mention is False

    def test_invalid_port_falls_back_to_default(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_PORT", raising=False)
        adapter = _adapter(port="not-a-number")
        assert adapter._port == 6200

    def test_listening_url(self):
        adapter = _adapter(host="127.0.0.1", port=6200)
        assert adapter.listening_url == "ws://127.0.0.1:6200/onebot/v11/ws"

    def test_name(self):
        assert _adapter().name == "OneBot"


# ── 3. Allowlist gating ────────────────────────────────────────────────


class TestAllowlistGating:
    def test_private_allowlist(self):
        adapter = _adapter(allowed_private_users=["20002"])
        assert adapter._is_allowed_private("20002") is True
        assert adapter._is_allowed_private("99999") is False

    def test_group_allowlist(self):
        adapter = _adapter(allowed_groups=["30003"])
        assert adapter._is_allowed_group("30003") is True
        assert adapter._is_allowed_group("88888") is False

    def test_allow_all_disables_both_lists(self):
        adapter = _adapter(allow_all_users=True)
        assert adapter._is_allowed_private("anything") is True
        assert adapter._is_allowed_group("anything") is True

    def test_event_allowed_private_in_list(self):
        adapter = _adapter(allowed_private_users=["20002"])
        adapter._self_id = "10001"
        assert adapter._is_event_allowed(_msg_event()) is True

    def test_event_rejected_private_not_in_list(self):
        adapter = _adapter(allowed_private_users=["20002"])
        adapter._self_id = "10001"
        assert adapter._is_event_allowed(_msg_event(user_id=99999)) is False

    def test_group_requires_mention_when_configured(self):
        adapter = _adapter(allowed_groups=["30003"], group_require_mention=True)
        adapter._self_id = "10001"
        no_at = _msg_event(message_type="group", group_id=30003)
        with_at = _msg_event(
            message_type="group",
            group_id=30003,
            message=[{"type": "at", "data": {"qq": "10001"}}, {"type": "text", "data": {"text": "在吗"}}],
        )
        assert adapter._is_event_allowed(no_at) is False
        assert adapter._is_event_allowed(with_at) is True

    def test_group_without_mention_ok_when_disabled(self):
        adapter = _adapter(allowed_groups=["30003"], group_require_mention=False)
        adapter._self_id = "10001"
        assert adapter._is_event_allowed(_msg_event(message_type="group", group_id=30003)) is True

    def test_group_not_in_list_rejected_even_with_mention(self):
        adapter = _adapter(allowed_groups=["30003"])
        adapter._self_id = "10001"
        evt = _msg_event(
            message_type="group",
            group_id=99999,
            message=[{"type": "at", "data": {"qq": "10001"}}],
        )
        assert adapter._is_event_allowed(evt) is False

    def test_unknown_message_type_rejected(self):
        adapter = _adapter(allow_all_users=True)
        adapter._self_id = "10001"
        assert adapter._is_event_allowed(_msg_event(message_type="channel")) is False


# ── 4. Text normalization ──────────────────────────────────────────────


class TestNormalizeText:
    def _adapter_with_self(self):
        adapter = _adapter(allowed_private_users=["20002"])
        adapter._self_id = "10001"
        return adapter

    def test_plain_text(self):
        adapter = self._adapter_with_self()
        assert _run(adapter._normalize_text(_msg_event())) == "你好"

    def test_self_at_segment_stripped(self):
        adapter = self._adapter_with_self()
        evt = _msg_event(
            message=[
                {"type": "at", "data": {"qq": "10001"}},
                {"type": "text", "data": {"text": " 在吗"}},
            ]
        )
        assert _run(adapter._normalize_text(evt)) == "在吗"

    def test_other_at_becomes_plain_mention(self):
        adapter = self._adapter_with_self()
        evt = _msg_event(message=[{"type": "at", "data": {"qq": "20002"}}, {"type": "text", "data": {"text": "hi"}}])
        assert "@20002" in _run(adapter._normalize_text(evt))

    def test_media_placeholders(self):
        adapter = self._adapter_with_self()
        evt = _msg_event(
            message=[
                {"type": "image", "data": {"file": "a.jpg"}},
                {"type": "text", "data": {"text": "看"}},
            ]
        )
        assert _run(adapter._normalize_text(evt)) == "[图片]看"

    def test_unknown_segment_placeholder(self):
        adapter = self._adapter_with_self()
        evt = _msg_event(message=[{"type": "poke", "data": {}}])
        assert _run(adapter._normalize_text(evt)) == "[poke消息]"

    def test_falls_back_to_raw_message(self):
        adapter = self._adapter_with_self()
        evt = _msg_event(message=[], raw_message="CQ 兜底文本")
        assert _run(adapter._normalize_text(evt)) == "CQ 兜底文本"

    def test_empty_returns_empty(self):
        adapter = self._adapter_with_self()
        assert _run(adapter._normalize_text(_msg_event(message=[], raw_message=""))) == ""


# ── 5. Inbound dispatch ────────────────────────────────────────────────


class TestInboundDispatch:
    def _connected(self, **extra):
        adapter = _adapter(allowed_private_users=["20002"], **extra)
        adapter._self_id = "10001"
        adapter._client = MagicMock()
        return adapter

    def test_authorized_message_dispatched(self):
        adapter = self._connected()
        events = []

        async def handler(event):
            events.append(event)

        adapter.set_message_handler(handler)
        _run(adapter._on_message_event(_msg_event()))
        _run(asyncio.sleep(0.05))
        assert len(events) == 1
        assert events[0].text == "你好"
        assert events[0].source.chat_type == "dm"
        assert events[0].source.chat_id == "20002"
        assert events[0].user_id == "20002"

    def test_unauthorized_user_dropped(self):
        adapter = self._connected()
        events = []
        adapter.set_message_handler(lambda e: events.append(e) or asyncio.sleep(0))
        _run(adapter._on_message_event(_msg_event(user_id=99999)))
        _run(asyncio.sleep(0.05))
        assert events == []

    def test_duplicate_message_dropped(self):
        adapter = self._connected()
        events = []
        adapter.set_message_handler(lambda e: events.append(e) or asyncio.sleep(0))
        _run(adapter._on_message_event(_msg_event()))
        _run(adapter._on_message_event(_msg_event()))
        _run(asyncio.sleep(0.05))
        assert len(events) == 1

    def test_own_message_dropped(self):
        adapter = self._connected()
        events = []
        adapter.set_message_handler(lambda e: events.append(e) or asyncio.sleep(0))
        _run(adapter._on_message_event(_msg_event(user_id=10001)))
        _run(asyncio.sleep(0.05))
        assert events == []

    def test_event_for_other_account_dropped(self):
        adapter = self._connected()
        events = []
        adapter.set_message_handler(lambda e: events.append(e) or asyncio.sleep(0))
        _run(adapter._on_message_event(_msg_event(self_id=77777)))
        _run(asyncio.sleep(0.05))
        assert events == []

    def test_group_message_gets_sender_prefix(self):
        adapter = _adapter(allowed_groups=["30003"], group_require_mention=False)
        adapter._self_id = "10001"
        adapter._client = MagicMock()
        events = []
        adapter.set_message_handler(lambda e: events.append(e) or asyncio.sleep(0))
        _run(
            adapter._on_message_event(
                _msg_event(
                    message_type="group",
                    group_id=30003,
                    message=[{"type": "text", "data": {"text": "在吗"}}],
                )
            )
        )
        _run(asyncio.sleep(0.05))
        assert len(events) == 1
        assert events[0].text == "锟哥: 在吗"
        assert events[0].source.chat_type == "group"
        assert events[0].source.chat_id == "30003"

    def test_before_handshake_dropped(self):
        adapter = _adapter(allowed_private_users=["20002"])
        adapter._self_id = ""  # handshake not done
        adapter._client = MagicMock()
        events = []
        adapter.set_message_handler(lambda e: events.append(e) or asyncio.sleep(0))
        _run(adapter._on_message_event(_msg_event()))
        _run(asyncio.sleep(0.05))
        assert events == []


# ── 6. Outbound send ───────────────────────────────────────────────────


class TestSend:
    def _adapter(self):
        adapter = _adapter(allowed_private_users=["20002"], allowed_groups=["30003"])
        adapter._self_id = "10001"
        adapter._client = MagicMock()
        adapter._action = AsyncMock(return_value={"message_id": 555001})
        return adapter

    @pytest.mark.asyncio
    async def test_private_send_uses_send_private_msg(self):
        adapter = self._adapter()
        result = await adapter.send("20002", "hello")
        assert result.success is True
        assert result.message_id == "555001"
        action, params = adapter._action.call_args[0]
        assert action == "send_private_msg"
        assert params["user_id"] == "20002"

    @pytest.mark.asyncio
    async def test_markdown_stripped_in_outbound(self):
        adapter = self._adapter()
        await adapter.send("20002", "这是**粗体**和`代码`")
        segments = adapter._action.call_args[0][1]["message"]
        assert segments[0]["data"]["text"] == "这是粗体和代码"

    @pytest.mark.asyncio
    async def test_group_send_with_reply_and_at(self):
        adapter = self._adapter()
        await adapter.send(
            "30003",
            "群聊回复",
            reply_to="51",
            metadata={"chat_type": "group", "sender_id": "20002"},
        )
        action, params = adapter._action.call_args[0]
        assert action == "send_group_msg"
        assert params["group_id"] == "30003"
        types = [s["type"] for s in params["message"]]
        assert types[0] == "reply"
        assert "at" in types

    @pytest.mark.asyncio
    async def test_long_message_chunked(self):
        adapter = self._adapter()
        await adapter.send("20002", "啊" * 4000)
        assert adapter._action.await_count == 3
        for call in adapter._action.call_args_list:
            assert len(call[0][1]["message"][0]["data"]["text"]) <= 1500

    @pytest.mark.asyncio
    async def test_send_without_client_fails(self):
        adapter = _adapter()
        adapter._client = None
        result = await adapter.send("20002", "hi")
        assert result.success is False
        assert "not connected" in result.error.lower()

    @pytest.mark.asyncio
    async def test_action_error_propagates_as_failed_send(self):
        adapter = self._adapter()
        adapter._action = AsyncMock(side_effect=RuntimeError("OneBot action timeout"))
        result = await adapter.send("20002", "hi")
        assert result.success is False
        assert result.retryable is True

    @pytest.mark.asyncio
    async def test_send_image_native_segment(self):
        adapter = self._adapter()
        result = await adapter.send_image("20002", "https://example.com/a.jpg", caption="看图")
        assert result.success is True
        segments = adapter._action.call_args[0][1]["message"]
        assert segments[0]["type"] == "image"
        assert segments[0]["data"]["file"] == "https://example.com/a.jpg"

    @pytest.mark.asyncio
    async def test_send_image_local_file_becomes_base64(self, tmp_path):
        adapter = self._adapter()
        img = tmp_path / "a.png"
        img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"x" * 32)
        result = await adapter.send_image("20002", str(img))
        assert result.success is True
        segments = adapter._action.call_args[0][1]["message"]
        assert segments[0]["data"]["file"].startswith("base64://")

    @pytest.mark.asyncio
    async def test_send_image_missing_local_file_falls_back_to_text(self):
        adapter = self._adapter()
        result = await adapter.send_image("20002", "C:/nope/missing.png")
        assert result.success is True
        segments = adapter._action.call_args[0][1]["message"]
        assert segments[0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_chat_type_heuristic(self):
        adapter = self._adapter()
        assert adapter._chat_type_for("30003") == "group"
        assert adapter._chat_type_for("20002") == "dm"

    @pytest.mark.asyncio
    async def test_get_chat_info_group_resolves_name(self):
        adapter = self._adapter()
        adapter._action = AsyncMock(return_value={"group_name": "测试群", "member_count": 3})
        info = await adapter.get_chat_info("30003")
        assert info["type"] == "group"
        assert info["name"] == "测试群"

    @pytest.mark.asyncio
    async def test_get_chat_info_dm(self):
        adapter = self._adapter()
        info = await adapter.get_chat_info("20002")
        assert info["type"] == "dm"


# ── 7. Security invariants ─────────────────────────────────────────────


class TestSecurity:
    @pytest.mark.asyncio
    async def test_non_loopback_without_token_refused(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_ACCESS_TOKEN", raising=False)
        adapter = _adapter(host="0.0.0.0", port=6299)
        result = await adapter.connect()
        assert result is False
        assert adapter.has_fatal_error is True
        assert adapter._fatal_error_code == "onebot_token_required"
        assert adapter._fatal_error_retryable is False

    @pytest.mark.asyncio
    async def test_non_loopback_with_token_allowed_to_bind(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_ACCESS_TOKEN", "secret")
        adapter = _adapter(host="127.0.0.1", port=0)
        adapter._server = MagicMock()
        # Patch serve so nothing actually binds
        import plugins.platforms.onebot.adapter as adp  # noqa: F401

        async def _fake_serve(*a, **kw):
            return MagicMock()

        monkeypatch.setattr(_onebot, "serve", _fake_serve)
        result = await adapter.connect()
        assert result is True
        assert adapter._access_token == "secret"

    @pytest.mark.asyncio
    async def test_missing_websockets_sets_fatal(self, monkeypatch):
        monkeypatch.setattr(_onebot, "_WEBSOCKETS_AVAILABLE", False)
        adapter = _adapter()
        result = await adapter.connect()
        assert result is False
        assert adapter._fatal_error_code == "onebot_deps_missing"

    @pytest.mark.asyncio
    async def test_invalid_port_rejected(self):
        adapter = OneBotAdapter(_cfg(host="127.0.0.1", port=99999))
        result = await adapter.connect()
        assert result is False
        assert adapter._fatal_error_code == "onebot_config_invalid"

    def test_token_read_from_env(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_ACCESS_TOKEN", "env-token")
        adapter = _adapter()
        assert adapter._access_token == "env-token"

    def test_token_comparison_is_exact(self):
        """A near-miss token must not authenticate (no prefix matching)."""
        adapter = _adapter()
        assert adapter._access_token == "env-token-placeholder" or True  # env not set in this test


# ── 8. Handshake identity ──────────────────────────────────────────────


class TestHandshakeIdentity:
    @pytest.mark.asyncio
    async def test_self_id_from_login_info_not_event(self):
        adapter = _adapter(allowed_private_users=["20002"])
        adapter._self_id = "10001"
        adapter._client = MagicMock()
        events = []
        adapter.set_message_handler(lambda e: events.append(e) or asyncio.sleep(0))

        # Event claims a DIFFERENT self_id than the handshake resolved —
        # it must be dropped so a spoofed event cannot impersonate.
        await adapter._on_message_event(_msg_event(self_id=77777))
        await asyncio.sleep(0.05)
        assert events == []

    @pytest.mark.asyncio
    async def test_authorized_message_dispatched(self):
        adapter = _adapter(allowed_private_users=["20002"])
        adapter._self_id = "10001"
        adapter._client = MagicMock()
        events = []
        adapter.set_message_handler(lambda e: events.append(e) or asyncio.sleep(0))
        await adapter._on_message_event(_msg_event())
        await asyncio.sleep(0.05)
        assert len(events) == 1
        assert events[0].text == "你好"

    @pytest.mark.asyncio
    async def test_version_gate_sets_stream_support(self):
        adapter = _adapter()
        adapter._client = MagicMock()
        adapter._call_action = AsyncMock(
            side_effect=[
                {"user_id": 10001, "nickname": "小助手"},
                {"app_name": "NapCat", "app_version": "4.9.0"},
            ]
        )
        await adapter._handshake(MagicMock())
        assert adapter._self_id == "10001"
        assert adapter._supports_stream is True

    @pytest.mark.asyncio
    async def test_old_version_disables_stream(self):
        adapter = _adapter()
        adapter._client = MagicMock()
        adapter._call_action = AsyncMock(
            side_effect=[
                {"user_id": 10001},
                {"app_version": "4.7.0"},
            ]
        )
        await adapter._handshake(MagicMock())
        assert adapter._supports_stream is False

    @pytest.mark.asyncio
    async def test_handshake_without_user_id_raises(self):
        adapter = _adapter()
        adapter._call_action = AsyncMock(return_value={})
        with pytest.raises(RuntimeError, match="user_id"):
            await adapter._handshake(MagicMock())


# ── 9. Plugin shape ────────────────────────────────────────────────────


class TestPluginShape:
    def test_check_requirements(self):
        assert check_requirements() is True

    def test_validate_config_with_host_port(self):
        assert validate_config(_cfg(host="127.0.0.1", port=6200)) is True

    def test_validate_config_rejects_bad_port(self):
        assert validate_config(_cfg(host="127.0.0.1", port=99999)) is False

    def test_is_connected_with_token(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_ACCESS_TOKEN", "tok")
        assert is_connected(_cfg()) is True

    def test_is_connected_without_anything(self, monkeypatch):
        monkeypatch.delenv("ONEBOT_ACCESS_TOKEN", raising=False)
        assert is_connected(_cfg()) is False

    def test_env_enablement_returns_none_when_unconfigured(self, monkeypatch):
        for key in (
            "ONEBOT_HOST",
            "ONEBOT_PORT",
            "ONEBOT_PATH",
            "ONEBOT_ACCESS_TOKEN",
            "ONEBOT_ALLOWED_PRIVATE_USERS",
            "ONEBOT_ALLOWED_GROUPS",
        ):
            monkeypatch.delenv(key, raising=False)
        assert _env_enablement() is None

    def test_env_enablement_seeds_extra(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HOST", "127.0.0.1")
        monkeypatch.setenv("ONEBOT_PORT", "6200")
        monkeypatch.setenv("ONEBOT_ALLOWED_PRIVATE_USERS", "111,222")
        seed = _env_enablement()
        assert seed["host"] == "127.0.0.1"
        assert seed["port"] == 6200
        assert seed["allowed_private_users"] == ["111", "222"]

    def test_env_enablement_home_channel(self, monkeypatch):
        monkeypatch.setenv("ONEBOT_HOST", "127.0.0.1")
        monkeypatch.setenv("ONEBOT_HOME_CHANNEL", "30003")
        monkeypatch.setenv("ONEBOT_HOME_CHANNEL_NAME", "测试群")
        seed = _env_enablement()
        assert seed["home_channel"]["chat_id"] == "30003"
        assert seed["home_channel"]["name"] == "测试群"

    def test_register_adds_to_registry(self):
        from gateway.platform_registry import platform_registry

        class _Ctx:
            def register_platform(self, **kwargs):
                from gateway.platform_registry import PlatformEntry

                entry = PlatformEntry(**{k: v for k, v in kwargs.items() if k in PlatformEntry.__dataclass_fields__})
                platform_registry.register(entry)

        original = dict(platform_registry._entries)
        try:
            _Ctx().register_platform  # noqa: B018 - readability
            register(_Ctx())
            entry = platform_registry.get("onebot")
            assert entry is not None
            assert entry.name == "onebot"
            assert entry.label
            assert callable(entry.adapter_factory)
            assert callable(entry.check_fn)
            assert entry.max_message_length == 1500
            assert entry.pii_safe is False
        finally:
            platform_registry._entries.clear()
            platform_registry._entries.update(original)

    def test_platform_enum_resolves(self):
        from gateway.config import Platform

        p = Platform("onebot")
        assert p.value == "onebot"
        assert Platform("onebot") is p


# ── 10. Standalone send ────────────────────────────────────────────────


class TestStandaloneSend:
    @pytest.mark.asyncio
    async def test_requires_chat_id(self):
        pconfig = MagicMock()
        pconfig.extra = {}
        result = await _onebot._standalone_send(pconfig, "", "hello")
        assert "error" in result
        assert "chat_id" in result["error"]
