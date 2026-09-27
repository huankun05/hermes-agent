"""OneBot 11 (NapCat) platform adapter (Hermes plugin).

Connects Hermes to QQ through NapCat's OneBot 11 reverse-WebSocket
endpoint.  NapCat logs in to a real QQ account, translates QQ traffic
into OneBot 11 events, and CONNECTS TO US (reverse WebSocket) — so this
adapter runs an asyncio WebSocket SERVER and no inbound firewall rule,
public IP, or port mapping is required.

Protocol: https://283375.github.io/onebot_v11_vitepress/

Message flow::

    QQ user ─ QQ servers ─ NapCat (logged-in QQ account)
                                  │  OneBot 11 JSON over reverse WebSocket
                                  ▼
                          OneBotAdapter (this module)
                                  │  MessageEvent
                                  ▼
                          Hermes agent core (full tools, memory, skills)

Configuration (config.yaml, under ``platforms.onebot``)::

    platforms:
      onebot:
        enabled: true
        extra:
          host: "127.0.0.1"        # listen address (loopback recommended)
          port: 6200               # listen port
          path: "/onebot/v11/ws"   # WebSocket path NapCat must dial
          access_token: ""         # shared secret (REQUIRED off-loopback)
          allowed_private_users: ["12345678"]   # QQ numbers allowed in DM
          allowed_groups: ["87654321"]          # group numbers allowed
          group_require_mention: true           # only respond when @-ed

Environment variables (env wins over ``extra``; non-secret settings only
— a token is a secret and lives in ``.env``)::

    ONEBOT_HOST                  listen address (default 127.0.0.1)
    ONEBOT_PORT                  listen port (default 6200)
    ONEBOT_PATH                  WebSocket path (default /onebot/v11/ws)
    ONEBOT_ACCESS_TOKEN          shared secret (secret: .env only)
    ONEBOT_ALLOWED_PRIVATE_USERS comma-separated QQ numbers (DM allowlist)
    ONEBOT_ALLOWED_GROUPS        comma-separated group numbers
    ONEBOT_GROUP_REQUIRE_MENTION "true" requires @bot in groups
    ONEBOT_ALLOW_ALL_USERS       "true" disables both allowlists (dev only)
    ONEBOT_HOME_CHANNEL          default chat_id for cron delivery
    ONEBOT_HOME_CHANNEL_NAME     human label for the home channel

Security notes
--------------
* Loopback listening with no token is safe only because the OS keeps
  127.0.0.1 local.  ANY non-loopback bind REQUIRES a token, enforced in
  ``connect()`` — otherwise any process that can reach the port could
  inject messages as the bot.
* DM and group allowlists are separate.  A group must be in
  ``allowed_groups`` AND (when ``group_require_mention`` is on) the
  message must @ the bot.
* ``self_id`` is resolved from NapCat's ``get_login_info`` at connect
  time, never from the event payload, so a spoofed event cannot claim to
  be another account.  Messages authored by the bot itself are dropped.

Media: inbound images/records/files are downloaded via OneBot actions
into the Hermes media cache and attached to the event (``media_urls``);
outbound images/records are sent natively (base64 for small files).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # websockets is a hard Hermes dependency (pyproject [dependencies])
    import websockets
    from websockets.asyncio.server import ServerConnection, serve
    _WEBSOCKETS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised via monkeypatched flag
    websockets = None  # type: ignore[assignment]
    ServerConnection = Any  # type: ignore[assignment,misc]
    serve = None  # type: ignore[assignment]
    _WEBSOCKETS_AVAILABLE = False

from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
)

from agent.secret_scope import UnscopedSecretError as _UnscopedSecretError
from agent.secret_scope import get_secret as _scoped_get_secret

logger = logging.getLogger(__name__)


def _get_scoped_secret(name: str, default: str = "") -> str:
    """Scope-aware secret read with the default-profile startup fallback.

    Same pattern as the bundled platform plugins (ntfy ``NTFY_TOKEN``,
    slack ``SLACK_APP_TOKEN``): secondary profiles resolve under a profile
    secret scope and a scoped miss returns the default, while the default
    profile sends unscoped and falls back to ``os.environ``.
    """
    try:
        val = _scoped_get_secret(name, None)
    except _UnscopedSecretError:
        val = os.getenv(name)
    if val is None:
        return default
    return str(val)


# ── Protocol constants ──────────────────────────────────────────────────

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 6200
DEFAULT_PATH = "/onebot/v11/ws"

#: NapCat >= 4.8.115 exposes the Stream API used for large media.
STREAM_MIN_VERSION = "4.8.115"

#: QQ text is chunked around ~1500 codepoints per message (NapCat-side
#: hard limit is much higher, but long walls of text get rate-limited).
MAX_MESSAGE_CODEPOINTS = 1500

#: Inbound files above this size are NOT inlined as base64 (memory).
MAX_INBOUND_BASE64_BYTES = 8 * 1024 * 1024

#: Inbound media cache lifetime.
MEDIA_CACHE_TTL_HOURS = 24
MEDIA_CACHE_MAX_FILES = 200

#: OneBot API actions are awaited with this timeout.
ACTION_TIMEOUT_SECONDS = 30.0

#: Reconnect backoff ladder (seconds) for the accept loop.
RECONNECT_BACKOFF = [1.0, 2.0, 5.0, 10.0, 30.0]

#: Dedup window for inbound message ids (seconds).
DEDUP_WINDOW_SECONDS = 600
DEDUP_MAX_SIZE = 1000

_LOOPBACK_HOSTS = {"", "127.0.0.1", "localhost", "::1", "::ffff:127.0.0.1"}


def _onebot_id(value: Any) -> str:
    """Normalize a OneBot id (number or string) to a non-empty string."""
    if isinstance(value, bool):
        return ""
    if isinstance(value, (int, str)):
        text = str(value).strip()
        return text
    return ""


def _is_loopback_host(host: str) -> bool:
    normalized = (host or "").strip().lower().strip("[]")
    return normalized in _LOOPBACK_HOSTS


def _split_text_codepoints(text: str, limit: int = MAX_MESSAGE_CODEPOINTS) -> List[str]:
    """Split text into QQ-sized chunks, preferring sentence boundaries."""
    source = (text or "").strip()
    if not source:
        return []
    if len(source) <= limit:
        return [source]

    chunks: List[str] = []
    rest = source
    boundary = re.compile(r"[。！？!?；;…\n]")
    while len(rest) > limit:
        window = rest[:limit]
        split_at = -1
        for i in range(len(window) - 1, int(limit * 0.55), -1):
            if boundary.match(window[i]):
                split_at = i + 1
                break
        if split_at < 0:
            space = window.rfind(" ")
            split_at = space if space > int(limit * 0.5) else limit
        chunks.append(rest[:split_at].strip())
        rest = rest[split_at:].lstrip()
    if rest.strip():
        chunks.append(rest.strip())
    return [c for c in chunks if c]


def _strip_markdown(text: str) -> str:
    """QQ renders raw markdown literally; keep it readable in plain text."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"(?<!\w)\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return text


def _version_at_least(version: str, minimum: str) -> bool:
    def _parts(value: str) -> Tuple[int, ...]:
        return tuple(int(p) for p in re.findall(r"\d+", value or "")[:3] or [0])
    return _parts(version) >= _parts(minimum)


def check_requirements() -> bool:
    """Passive dependency probe — never installs anything."""
    return _WEBSOCKETS_AVAILABLE


def validate_config(config) -> bool:
    """A host/port is enough to start listening; NapCat may dial later."""
    if not _WEBSOCKETS_AVAILABLE:
        return False
    extra = getattr(config, "extra", {}) or {}
    host = os.getenv("ONEBOT_HOST") or extra.get("host") or DEFAULT_HOST
    port_raw = os.getenv("ONEBOT_PORT") or extra.get("port") or DEFAULT_PORT
    try:
        port = int(port_raw)
    except (TypeError, ValueError):
        return False
    return bool(host) and 0 < port < 65536


def is_connected(config) -> bool:
    """True when a NapCat client is actually connected (not just listening)."""
    extra = getattr(config, "extra", {}) or {}
    if os.getenv("ONEBOT_ACCESS_TOKEN") or extra.get("access_token"):
        return True
    return bool(os.getenv("ONEBOT_HOST") or extra.get("host"))


# ── OneBot 11 event helpers ─────────────────────────────────────────────


def is_message_event(event: Any) -> bool:
    """True for an inbound ``post_type == "message"`` event (not meta/notice)."""
    if not isinstance(event, dict):
        return False
    return (
        event.get("post_type") == "message"
        and event.get("message_type") in ("private", "group")
        and isinstance(event.get("message"), list)
    )


def is_api_response(payload: Any) -> bool:
    """True for an OneBot API response frame (carries ``echo``)."""
    return isinstance(payload, dict) and "echo" in payload and "status" in payload


def segment_text(segment: Any) -> str:
    """Text content of a single array-format segment."""
    if not isinstance(segment, dict):
        return ""
    kind = segment.get("type")
    data = segment.get("data") if isinstance(segment.get("data"), dict) else {}
    if kind == "text":
        return str(data.get("text") or "")
    if kind == "markdown":
        return str(data.get("markdown") or data.get("content") or "")
    return ""


def message_mentions_self(segments: Any, self_id: str) -> bool:
    """True when an ``at`` segment targets the bot (or @everyone)."""
    if not self_id or not isinstance(segments, list):
        return False
    for segment in segments:
        if not isinstance(segment, dict) or segment.get("type") != "at":
            continue
        target = _onebot_id((segment.get("data") or {}).get("qq"))
        if target and target == self_id:
            return True
    return False


def parse_allowed_list(raw: Any) -> List[str]:
    """Parse a comma/space-separated allowlist into clean id strings."""
    if raw is None:
        return []
    if isinstance(raw, (list, tuple, set)):
        items: List[Any] = list(raw)
    else:
        items = re.split(r"[,，\s]+", str(raw))
    out: List[str] = []
    for item in items:
        value = _onebot_id(item)
        if value and value not in out:
            out.append(value)
    return out


class OneBotAdapter(BasePlatformAdapter):
    """OneBot 11 (NapCat) adapter — reverse WebSocket server.

    NapCat dials US, so there is no outbound connection and no public
    endpoint.  A single NapCat client is served at a time; a second
    connection for the same QQ account replaces the first (matching
    NapCat's reconnect semantics), while a DIFFERENT account is rejected
    so two bots cannot silently hijack one listener.
    """

    # QQ renders literal markdown; the gateway's code-block fallback would
    # look broken, so claim the plain-text presentation path.
    supports_code_blocks = False
    # Long replies are chunked natively in send().
    splits_long_messages = True
    # Typing indicators exist in OneBot but are group-only and noisy.
    supports_status_text = False

    MAX_MESSAGE_CODEPOINTS = MAX_MESSAGE_CODEPOINTS

    def __init__(self, config: PlatformConfig):
        platform = Platform("onebot")
        super().__init__(config=config, platform=platform)

        extra = getattr(config, "extra", {}) or {}

        self._host: str = (
            os.getenv("ONEBOT_HOST") or extra.get("host") or DEFAULT_HOST
        ).strip()
        port_raw = os.getenv("ONEBOT_PORT") or extra.get("port") or DEFAULT_PORT
        try:
            self._port = int(port_raw)
        except (TypeError, ValueError):
            self._port = DEFAULT_PORT
        self._path: str = (
            os.getenv("ONEBOT_PATH") or extra.get("path") or DEFAULT_PATH
        ).strip() or DEFAULT_PATH
        if not self._path.startswith("/"):
            self._path = "/" + self._path

        self._access_token: str = _get_scoped_secret("ONEBOT_ACCESS_TOKEN", "").strip()

        self._allowed_private_users: List[str] = parse_allowed_list(
            os.getenv("ONEBOT_ALLOWED_PRIVATE_USERS") or extra.get("allowed_private_users")
        )
        self._allowed_groups: List[str] = parse_allowed_list(
            os.getenv("ONEBOT_ALLOWED_GROUPS") or extra.get("allowed_groups")
        )
        require_mention_env = os.getenv("ONEBOT_GROUP_REQUIRE_MENTION")
        if require_mention_env is not None:
            self._group_require_mention = require_mention_env.strip().lower() in (
                "1",
                "true",
                "yes",
                "on",
            )
        else:
            self._group_require_mention = bool(extra.get("group_require_mention", True))
        self._allow_all_users = os.getenv("ONEBOT_ALLOW_ALL_USERS", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ) or bool(extra.get("allow_all_users", False))

        # Runtime state
        self._server = None  # websockets.asyncio.server.Server
        self._client: Optional[ServerConnection] = None
        self._client_lock = asyncio.Lock()
        self._accept_task: Optional[asyncio.Task] = None
        self._self_id: str = ""
        self._nickname: str = ""
        self._app_version: str = ""
        self._supports_stream = False

        # OneBot action call bookkeeping: echo -> Future
        self._pending: Dict[str, asyncio.Future] = {}
        self._echo_counter = 0

        # Dedup: message_id -> expiry
        self._seen_messages: Dict[str, float] = {}

        # Per-chat serial queue so two rapid messages in one chat do not
        # interleave their agent turns (mirrors the NapCat adapter).
        self._chat_queues: Dict[str, float] = {}

        # Local lock so two Hermes profiles cannot both bind the port.
        self._lock_key: Optional[str] = None

        self._media_dir = self._resolve_media_dir()

    # ── Naming ──────────────────────────────────────────────────────────

    @property
    def name(self) -> str:
        return "OneBot"

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_media_dir() -> Path:
        """Inbound media cache under HERMES_HOME/media/onebot (profile-safe)."""
        try:
            from hermes_cli.paths import get_hermes_home

            base = Path(get_hermes_home()) / "media" / "onebot"
        except Exception:
            base = Path(os.environ.get("HERMES_HOME", "")) / "media" / "onebot"
        try:
            base.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        return base

    def _prune_media_cache(self) -> None:
        """Best-effort TTL/size cap on the inbound media cache."""
        try:
            files = sorted(
                (p for p in self._media_dir.iterdir() if p.is_file()),
                key=lambda p: p.stat().st_mtime,
            )
        except OSError:
            return
        cutoff = time.time() - MEDIA_CACHE_TTL_HOURS * 3600
        for path in files:
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
            except OSError:
                continue
        remaining = [p for p in self._media_dir.iterdir() if p.is_file()]
        for path in remaining[: max(0, len(remaining) - MEDIA_CACHE_MAX_FILES)]:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                continue

    def _is_allowed_private(self, user_id: str) -> bool:
        if self._allow_all_users:
            return True
        return user_id in self._allowed_private_users

    def _is_allowed_group(self, group_id: str) -> bool:
        if self._allow_all_users:
            return True
        return group_id in self._allowed_groups

    def _is_event_allowed(self, event: Dict[str, Any]) -> bool:
        """DM allowlist / group allowlist + @-mention gate."""
        message_type = event.get("message_type")
        if message_type == "private":
            return self._is_allowed_private(_onebot_id(event.get("user_id")))
        if message_type == "group":
            group_id = _onebot_id(event.get("group_id"))
            if not self._is_allowed_group(group_id):
                return False
            if self._group_require_mention:
                return message_mentions_self(event.get("message"), self._self_id)
            return True
        return False

    def _is_duplicate(self, message_id: str) -> bool:
        now = time.time()
        if len(self._seen_messages) > DEDUP_MAX_SIZE:
            cutoff = now - DEDUP_WINDOW_SECONDS
            self._seen_messages = {
                k: v for k, v in self._seen_messages.items() if v > cutoff
            }
        if message_id in self._seen_messages:
            return True
        self._seen_messages[message_id] = now
        return False

    # ── Connection lifecycle ────────────────────────────────────────────

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start the reverse-WebSocket listener and wait for NapCat.

        Returns True as soon as the listener is bound (NapCat may connect
        later); the actual account identity is resolved on connect.
        """
        if not _WEBSOCKETS_AVAILABLE:
            logger.warning(
                "[%s] websockets not installed. Run: pip install websockets", self.name
            )
            self._set_fatal_error(
                "onebot_deps_missing",
                "websockets package is required for the OneBot adapter",
                retryable=False,
            )
            return False

        if not self._host:
            self._set_fatal_error(
                "onebot_config_missing",
                "ONEBOT_HOST must be set",
                retryable=False,
            )
            return False
        if not (0 < self._port < 65536):
            self._set_fatal_error(
                "onebot_config_invalid",
                f"ONEBOT_PORT must be 1-65535 (got {self._port})",
                retryable=False,
            )
            return False

        # Non-loopback binds accept connections from the whole network, so a
        # missing token would let anyone inject messages as the bot.
        if not _is_loopback_host(self._host) and not self._access_token:
            self._set_fatal_error(
                "onebot_token_required",
                (
                    f"Listening on non-loopback host {self._host} requires "
                    "ONEBOT_ACCESS_TOKEN — any host that can reach the port "
                    "could otherwise inject messages as this bot."
                ),
                retryable=False,
            )
            logger.error("[%s] Refusing to bind %s without a token", self.name, self._host)
            return False

        # Machine-local port lock (same pattern as the IRC adapter).
        try:
            from gateway.status import acquire_scoped_lock, release_scoped_lock

            lock_key = f"{self._host}:{self._port}{self._path}"
            acquired, _holder = acquire_scoped_lock("onebot", lock_key)
            if not acquired:
                self._set_fatal_error(
                    "onebot_lock_conflict",
                    f"Another local Hermes is already listening on {lock_key}",
                    retryable=False,
                )
                return False
            self._lock_key = lock_key
        except ImportError:
            self._lock_key = None  # status module unavailable (e.g. tests)

        try:
            self._server = await serve(
                self._handle_connection,
                self._host,
                self._port,
                # NapCat sends no Origin header; default (same-origin) is fine.
                max_size=64 * 1024 * 1024,
                ping_interval=30,
                ping_timeout=30,
                close_timeout=5,
            )
        except Exception as exc:
            logger.error("[%s] Failed to listen on %s:%s — %s", self.name, self._host, self._port, exc)
            self._release_lock()
            self._set_fatal_error("onebot_listen_failed", str(exc), retryable=True)
            return False

        self._mark_connected()
        logger.info(
            "[%s] Listening on ws://%s:%s%s — waiting for NapCat to connect",
            self.name,
            self._host,
            self._port,
            self._path,
        )
        return True

    async def disconnect(self) -> None:
        """Stop listening and drop the NapCat connection."""
        self._running = False
        self._mark_disconnected()

        client, self._client = self._client, None
        if client is not None:
            try:
                await client.close(code=1001, reason="adapter shutting down")
            except Exception:
                pass

        server, self._server = self._server, None
        if server is not None:
            try:
                server.close()
                await server.wait_closed()
            except Exception:
                pass

        # Reject any action still in flight so callers do not hang.
        for future in list(self._pending.values()):
            if not future.done():
                future.set_exception(
                    asyncio.CancelledError("OneBot adapter disconnected")
                )
        self._pending.clear()

        self._self_id = ""
        self._nickname = ""
        self._app_version = ""
        self._supports_stream = False
        self._seen_messages.clear()
        self._chat_queues.clear()
        self._release_lock()
        logger.info("[%s] Disconnected", self.name)

    def _release_lock(self) -> None:
        if not self._lock_key:
            return
        try:
            from gateway.status import release_scoped_lock

            release_scoped_lock("onebot", self._lock_key)
        except Exception:
            pass
        self._lock_key = None

    # ── WebSocket server side ───────────────────────────────────────────

    async def _handle_connection(self, websocket: ServerConnection) -> None:
        """Serve one NapCat client for the lifetime of its connection."""
        request_path = "/"
        try:
            request_path = websocket.request.path if websocket.request else "/"
        except Exception:
            pass
        if request_path != self._path:
            logger.warning(
                "[%s] Rejecting connection on %s (expected %s)",
                self.name,
                request_path,
                self._path,
            )
            await websocket.close(code=1008, reason="wrong path")
            return

        if self._access_token:
            header = ""
            try:
                header = websocket.request.headers.get("Authorization", "") or ""
            except Exception:
                header = ""
            supplied = header[7:].strip() if header.lower().startswith("bearer ") else header.strip()
            if supplied != self._access_token:
                logger.warning("[%s] Rejecting connection with bad access token", self.name)
                await websocket.close(code=1008, reason="unauthorized")
                return

        async with self._client_lock:
            previous = self._client
            if previous is not None and previous is not websocket:
                # NapCat reconnecting: the newer connection wins. A different
                # account is refused so two bots cannot share one listener.
                try:
                    previous_self = self._self_id
                except Exception:
                    previous_self = ""
                header_self = ""
                try:
                    header_self = _onebot_id(
                        websocket.request.headers.get("X-Self-ID")
                    )
                except Exception:
                    header_self = ""
                if previous_self and header_self and previous_self != header_self:
                    logger.warning(
                        "[%s] Refusing second QQ account %s (already serving %s)",
                        self.name,
                        header_self,
                        previous_self,
                    )
                    await websocket.close(code=1008, reason="another account connected")
                    return
                try:
                    await previous.close(code=1001, reason="replaced by reconnect")
                except Exception:
                    pass
            self._client = websocket
            self._self_id = ""
            self._nickname = ""
            self._app_version = ""

        # The handshake must run CONCURRENTLY with the receive loop: the
        # API responses that resolve ``get_login_info`` are dispatched by
        # that loop, so awaiting the handshake before starting it would
        # deadlock until the action timeout fires.
        handshake_task = asyncio.create_task(self._handshake(websocket))
        try:
            async for raw in websocket:
                if not self._running:
                    break
                try:
                    payload = json.loads(raw)
                except (ValueError, TypeError):
                    logger.debug("[%s] Ignoring malformed frame", self.name)
                    continue
                if is_api_response(payload):
                    self._resolve_action(payload)
                    continue
                if is_message_event(payload):
                    await self._on_message_event(payload)
        except Exception as exc:  # connection dropped / protocol error
            logger.info("[%s] NapCat connection ended: %s", self.name, exc)
        finally:
            if not handshake_task.done():
                handshake_task.cancel()
                try:
                    await handshake_task
                except (asyncio.CancelledError, Exception):
                    pass
            async with self._client_lock:
                if self._client is websocket:
                    self._client = None
            self._self_id = ""
            for future in list(self._pending.values()):
                if not future.done():
                    future.set_exception(
                        asyncio.CancelledError("OneBot connection closed")
                    )
            self._pending.clear()
            logger.info("[%s] NapCat disconnected — still listening for reconnect", self.name)

    async def _handshake(self, websocket: ServerConnection) -> None:
        """Resolve the bot identity before accepting any traffic."""
        login = await self._call_action(websocket, "get_login_info")
        self_id = _onebot_id((login or {}).get("user_id"))
        if not self_id:
            raise RuntimeError("NapCat get_login_info returned no user_id")

        header_self = ""
        try:
            header_self = _onebot_id(websocket.request.headers.get("X-Self-ID"))
        except Exception:
            header_self = ""
        if header_self and header_self != self_id:
            raise RuntimeError(
                f"NapCat X-Self-ID {header_self} != get_login_info {self_id}"
            )

        version = await self._call_action(websocket, "get_version_info") or {}
        app_version = str(version.get("app_version") or "")
        self._supports_stream = _version_at_least(app_version, STREAM_MIN_VERSION)

        async with self._client_lock:
            self._self_id = self_id
            self._nickname = str((login or {}).get("nickname") or "")
            self._app_version = app_version

        logger.info(
            "[%s] NapCat connected: %s (%s) v%s%s",
            self.name,
            self._self_id,
            self._nickname or "no nickname",
            app_version or "?",
            "" if self._supports_stream else f" — media streaming needs {STREAM_MIN_VERSION}+",
        )

    # ── OneBot action calls ─────────────────────────────────────────────

    def _next_echo(self) -> str:
        self._echo_counter += 1
        return f"h{self._echo_counter}-{uuid.uuid4().hex[:8]}"

    def _resolve_action(self, payload: Dict[str, Any]) -> None:
        echo = str(payload.get("echo") or "")
        future = self._pending.pop(echo, None)
        if future is None or future.done():
            return
        status = str(payload.get("status") or "")
        retcode = payload.get("retcode")
        if status == "failed" or (isinstance(retcode, int) and retcode != 0):
            wording = (
                payload.get("wording")
                or payload.get("message")
                or f"OneBot action failed (retcode={retcode})"
            )
            future.set_exception(RuntimeError(str(wording)))
            return
        future.set_result(payload.get("data"))

    async def _call_action(
        self,
        websocket: ServerConnection,
        action: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = ACTION_TIMEOUT_SECONDS,
    ) -> Any:
        """Invoke an OneBot API action and await its response frame."""
        if websocket is None:
            raise RuntimeError("OneBot client is not connected")
        echo = self._next_echo()
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        self._pending[echo] = future
        try:
            await websocket.send(
                json.dumps(
                    {"action": action, "params": params or {}, "echo": echo},
                    ensure_ascii=False,
                )
            )
        except Exception as exc:
            self._pending.pop(echo, None)
            raise RuntimeError(f"OneBot send failed for {action}: {exc}") from exc
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as exc:
            self._pending.pop(echo, None)
            raise RuntimeError(f"OneBot action timeout: {action}") from exc

    async def _action(self, action: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """Action call against the CURRENT NapCat client."""
        async with self._client_lock:
            client = self._client
        if client is None:
            raise RuntimeError("NapCat is not connected")
        return await self._call_action(client, action, params)

    # ── Inbound processing ──────────────────────────────────────────────

    async def _on_message_event(self, event: Dict[str, Any]) -> None:
        """Normalize one OneBot message event and dispatch it to the agent."""
        if not self._self_id:
            logger.debug("[%s] Dropping event before handshake completed", self.name)
            return

        event_self = _onebot_id(event.get("self_id"))
        if event_self and event_self != self._self_id:
            logger.debug("[%s] Dropping event for another account %s", self.name, event_self)
            return
        sender_id = _onebot_id(event.get("user_id"))
        if sender_id and sender_id == self._self_id:
            return  # our own message echoed back

        message_id = _onebot_id(event.get("message_id")) or uuid.uuid4().hex
        if self._is_duplicate(message_id):
            return
        if not self._is_event_allowed(event):
            logger.debug(
                "[%s] Dropping unauthorized %s message from %s",
                self.name,
                event.get("message_type"),
                sender_id,
            )
            return

        text = await self._normalize_text(event)
        if not text:
            return

        is_group = event.get("message_type") == "group"
        group_id = _onebot_id(event.get("group_id"))
        chat_id = group_id if is_group else sender_id
        if not chat_id:
            return

        sender = event.get("sender") if isinstance(event.get("sender"), dict) else {}
        sender_name = str(sender.get("card") or sender.get("nickname") or "").strip()

        if is_group and sender_name:
            text = f"{sender_name}: {text}"

        source = self.build_source(
            chat_id=chat_id,
            chat_name=group_id if is_group else (sender_name or sender_id),
            chat_type="group" if is_group else "dm",
            user_id=sender_id,
            user_name=sender_name or sender_id,
        )

        media_urls, media_types = await self._download_inbound_media(event)

        ts_raw = event.get("time")
        try:
            timestamp = (
                datetime.fromtimestamp(int(ts_raw), tz=timezone.utc)
                if ts_raw
                else datetime.now(tz=timezone.utc)
            )
        except (ValueError, OSError, TypeError):
            timestamp = datetime.now(tz=timezone.utc)

        message_event = MessageEvent(
            text=text,
            message_type=MessageType.TEXT,
            source=source,
            user_id=sender_id,
            user_name=sender_name or sender_id,
            message_id=message_id,
            raw_message=event,
            timestamp=timestamp,
            media_urls=media_urls,
            media_types=media_types,
            metadata={
                "onebot_message_type": event.get("message_type"),
                "onebot_group_id": group_id or None,
                "onebot_self_id": self._self_id,
                "onebot_raw_message": event.get("raw_message"),
            },
        )

        await self._dispatch_serialized(chat_id, message_event)

    async def _dispatch_serialized(self, chat_id: str, event: MessageEvent) -> None:
        """Serialize agent turns per chat (base class spawns its own task)."""
        previous = self._chat_queues.get(chat_id)
        if previous is not None and not previous.done():
            # Drop the older queued message — a newer one supersedes it.
            previous.cancel()
        task = asyncio.create_task(self.handle_message(event))
        self._chat_queues[chat_id] = task

        def _cleanup(t: asyncio.Task) -> None:
            if self._chat_queues.get(chat_id) is t:
                self._chat_queues.pop(chat_id, None)

        task.add_done_callback(_cleanup)

    async def _normalize_text(self, event: Dict[str, Any]) -> str:
        """Flatten array-format segments into plain text (CQ fallback)."""
        segments = event.get("message")
        parts: List[str] = []
        if isinstance(segments, list):
            for segment in segments:
                text = segment_text(segment)
                if text:
                    parts.append(text)
                    continue
                kind = segment.get("type") if isinstance(segment, dict) else None
                if kind == "at":
                    target = _onebot_id((segment.get("data") or {}).get("qq"))
                    if target == self._self_id:
                        continue  # strip the leading @bot
                    parts.append(f" @{target} ")
                elif kind == "face":
                    parts.append("[QQ表情]")
                elif kind in ("image", "mface"):
                    parts.append("[图片]")
                elif kind == "record":
                    parts.append("[语音]")
                elif kind == "file":
                    parts.append("[文件]")
                elif kind == "video":
                    parts.append("[视频]")
                elif kind == "reply":
                    parts.append("[回复]")
                elif kind == "json":
                    parts.append("[JSON卡片]")
                elif kind == "forward":
                    parts.append("[合并转发]")
                else:
                    parts.append(f"[{kind or '未知'}消息]")
        text = "".join(parts).strip()
        if not text:
            text = str(event.get("raw_message") or "").strip()
        return text

    async def _download_inbound_media(
        self, event: Dict[str, Any]
    ) -> Tuple[List[str], List[str]]:
        """Download image/record/file segments into the media cache.

        Returns ``(local_paths, media_types)`` for ``MessageEvent``. Any
        failure degrades to the ``[图片]`` placeholder already emitted by
        ``_normalize_text`` — media is never worth failing a turn over.
        """
        segments = event.get("message")
        if not isinstance(segments, list):
            return [], []

        targets: List[Tuple[str, Dict[str, Any], str]] = []
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            kind = segment.get("type")
            if kind not in ("image", "record", "file", "video"):
                continue
            data = segment.get("data") if isinstance(segment.get("data"), dict) else {}
            if data.get("url"):
                targets.append((kind, data, "url"))
            elif data.get("file"):
                targets.append((kind, data, "file"))
        if not targets:
            return [], []

        paths: List[str] = []
        types: List[str] = []
        for kind, data, mode in targets:
            try:
                local = await self._fetch_media(kind, data, mode)
            except Exception as exc:
                logger.warning("[%s] Inbound %s download failed: %s", self.name, kind, exc)
                continue
            if local:
                paths.append(local)
                types.append(kind)
        if paths:
            self._prune_media_cache()
        return paths, types

    async def _fetch_media(self, kind: str, data: Dict[str, Any], mode: str) -> Optional[str]:
        """Fetch one media segment to a local cache file."""
        suffix = {
            "image": ".jpg",
            "record": ".mp3",
            "file": ".bin",
            "video": ".mp4",
        }.get(kind, ".bin")
        name = data.get("name") or data.get("file")
        if isinstance(name, str) and Path(name).suffix:
            suffix = Path(name).suffix
        dest = self._media_dir / f"{int(time.time() * 1000)}-{uuid.uuid4().hex[:8]}{suffix}"

        if mode == "url":
            url = str(data.get("url") or "")
            if not url:
                return None
            return await self._http_download(url, dest)
        # file id / local path on the NapCat host
        file_ref = str(data.get("file") or "")
        if not file_ref:
            return None
        if file_ref.startswith("file://"):
            return await self._http_download(file_ref[7:], dest)
        if file_ref.startswith(("http://", "https://")):
            return await self._http_download(file_ref, dest)
        if file_ref.startswith("base64://"):
            payload = file_ref[len("base64://"):]
            raw = base64.b64decode(payload)
            dest.write_bytes(raw)
            return str(dest)
        if Path(file_ref).is_file():  # NapCat on the same machine
            dest.write_bytes(Path(file_ref).read_bytes())
            return str(dest)
        # Remote file id — ask NapCat to resolve it to a URL/path.
        try:
            resolved = await self._action("get_file", {"file_id": file_ref})
        except Exception:
            resolved = None
        if isinstance(resolved, dict):
            url = str(resolved.get("url") or "")
            local_path = str(resolved.get("file") or "")
            if url:
                return await self._http_download(url, dest)
            if local_path and Path(local_path).is_file():
                dest.write_bytes(Path(local_path).read_bytes())
                return str(dest)
        return None

    async def _http_download(self, url: str, dest: Path) -> Optional[str]:
        try:
            import httpx
        except ImportError:  # pragma: no cover - httpx is a hard dependency
            raise RuntimeError("httpx is required to download OneBot media")
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
        return str(dest)

    # ── Outbound messaging ──────────────────────────────────────────────

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a text message (chunked) to a QQ user or group."""
        metadata = metadata or {}
        if not self._client:
            return SendResult(success=False, error="NapCat is not connected")

        chat_type = metadata.get("chat_type") or self._chat_type_for(chat_id)
        reply_message_id = reply_to or metadata.get("reply_to_message_id")

        plain = _strip_markdown(content)
        chunks = _split_text_codepoints(plain) or [""]

        last_id: Optional[str] = None
        for index, chunk in enumerate(chunks):
            segments: List[Dict[str, Any]] = []
            if index == 0 and chat_type == "group":
                if reply_message_id:
                    segments.append({"type": "reply", "data": {"id": reply_message_id}})
                sender_id = metadata.get("sender_id")
                if sender_id:
                    segments.append({"type": "at", "data": {"qq": sender_id}})
                    segments.append({"type": "text", "data": {"text": " "}})
            segments.append({"type": "text", "data": {"text": chunk}})

            action = "send_group_msg" if chat_type == "group" else "send_private_msg"
            params: Dict[str, Any] = {"message": segments}
            if chat_type == "group":
                params["group_id"] = chat_id
            else:
                params["user_id"] = chat_id
            try:
                data = await self._action(action, params)
                last_id = _onebot_id((data or {}).get("message_id")) or last_id
            except Exception as exc:
                return SendResult(success=False, error=str(exc), retryable=True)
            if index < len(chunks) - 1:
                await asyncio.sleep(0.5)

        return SendResult(success=True, message_id=last_id or str(int(time.time() * 1000)))

    async def send_image(
        self,
        chat_id: str,
        image_url: str,
        caption: Optional[str] = None,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send an image natively (base64 for local files, URL otherwise)."""
        metadata = metadata or {}
        if not self._client:
            return SendResult(success=False, error="NapCat is not connected")
        chat_type = metadata.get("chat_type") or self._chat_type_for(chat_id)

        file_ref = await self._image_ref_for_send(image_url)
        if not file_ref:
            return await self.send(
                chat_id, f"{caption}\n{image_url}" if caption else image_url,
                reply_to=reply_to, metadata=metadata,
            )

        segments: List[Dict[str, Any]] = [{"type": "image", "data": {"file": file_ref}}]
        if caption:
            segments.append({"type": "text", "data": {"text": f" {caption}"}})

        action = "send_group_msg" if chat_type == "group" else "send_private_msg"
        params: Dict[str, Any] = {"message": segments}
        if chat_type == "group":
            params["group_id"] = chat_id
        else:
            params["user_id"] = chat_id
        try:
            data = await self._action(action, params)
        except Exception as exc:
            return SendResult(success=False, error=str(exc), retryable=True)
        message_id = _onebot_id((data or {}).get("message_id")) or str(int(time.time() * 1000))
        return SendResult(success=True, message_id=message_id)

    async def _image_ref_for_send(self, image_url: str) -> Optional[str]:
        """Local path → base64 (≤8 MiB) or file path; remote URL → URL."""
        path: Optional[Path] = None
        if image_url.startswith("file://"):
            path = Path(image_url[7:])
        elif re.match(r"^[a-zA-Z]:[\\/]", image_url) or image_url.startswith("/"):
            path = Path(image_url)
        if path is not None:
            if not path.is_file():
                return None
            if path.stat().st_size <= MAX_INBOUND_BASE64_BYTES:
                encoded = base64.b64encode(path.read_bytes()).decode("ascii")
                return f"base64://{encoded}"
            return str(path)
        if image_url.startswith(("http://", "https://", "base64://")):
            return image_url
        return None

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """OneBot has no portable typing indicator — no-op."""
        return None

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Group membership vs private chat, resolved live when possible."""
        info: Dict[str, Any] = {"name": chat_id, "type": self._chat_type_for(chat_id)}
        if info["type"] == "group" and self._client:
            try:
                data = await self._action(
                    "get_group_info", {"group_id": chat_id, "no_cache": False}
                )
                if isinstance(data, dict):
                    info["name"] = str(data.get("group_name") or chat_id)
                    info["member_count"] = data.get("member_count")
            except Exception:
                pass
        return info

    def _chat_type_for(self, chat_id: str) -> str:
        """Heuristic fallback when the caller did not pass a chat_type."""
        if chat_id in self._allowed_groups:
            return "group"
        return "dm"

    # ── Status helpers ──────────────────────────────────────────────────

    @property
    def listening_url(self) -> str:
        host = self._host or DEFAULT_HOST
        return f"ws://{host}:{self._port}{self._path}"


# ── Plugin registration ─────────────────────────────────────────────────


def _env_enablement() -> Optional[dict]:
    """Seed ``PlatformConfig.extra`` from env vars at config-load time.

    Returns ``None`` when OneBot is not minimally configured so the
    registry skips auto-enabling.
    """
    host = os.getenv("ONEBOT_HOST", "").strip()
    port = os.getenv("ONEBOT_PORT", "").strip()
    path = os.getenv("ONEBOT_PATH", "").strip()
    token = _get_scoped_secret("ONEBOT_ACCESS_TOKEN").strip()
    private_users = os.getenv("ONEBOT_ALLOWED_PRIVATE_USERS", "").strip()
    groups = os.getenv("ONEBOT_ALLOWED_GROUPS", "").strip()
    if not (host or port or path or token or private_users or groups):
        return None

    seed: Dict[str, Any] = {
        "host": host or DEFAULT_HOST,
        "port": int(port) if port.isdigit() else DEFAULT_PORT,
        "path": path or DEFAULT_PATH,
    }
    if token:
        seed["access_token"] = token
    if private_users:
        seed["allowed_private_users"] = parse_allowed_list(private_users)
    if groups:
        seed["allowed_groups"] = parse_allowed_list(groups)
    require_mention = os.getenv("ONEBOT_GROUP_REQUIRE_MENTION", "").strip().lower()
    if require_mention:
        seed["group_require_mention"] = require_mention in ("1", "true", "yes", "on")
    home = os.getenv("ONEBOT_HOME_CHANNEL", "").strip()
    if home:
        seed["home_channel"] = {
            "chat_id": home,
            "name": os.getenv("ONEBOT_HOME_CHANNEL_NAME", home),
        }
    return seed


async def _standalone_send(
    pconfig,
    chat_id: str,
    message: str,
    *,
    thread_id: Optional[str] = None,
    media_files: Optional[List[str]] = None,
    force_document: bool = False,
) -> Dict[str, Any]:
    """Out-of-process send for cron / send_message_tool fallbacks.

    Opens a short-lived reverse-WebSocket listener, waits for NapCat to
    dial in, sends the message, and disconnects. Used when the gateway
    runner is not in this process (``hermes cron`` standalone).
    """
    if not _WEBSOCKETS_AVAILABLE:
        return {"error": "onebot standalone send: websockets not installed"}

    extra = getattr(pconfig, "extra", {}) or {}
    host = (os.getenv("ONEBOT_HOST") or extra.get("host") or DEFAULT_HOST).strip()
    port_raw = os.getenv("ONEBOT_PORT") or extra.get("port") or DEFAULT_PORT
    try:
        port = int(port_raw)
    except (TypeError, ValueError):
        return {"error": f"onebot standalone send: invalid port {port_raw!r}"}
    path = (os.getenv("ONEBOT_PATH") or extra.get("path") or DEFAULT_PATH).strip()
    token = _get_scoped_secret("ONEBOT_ACCESS_TOKEN").strip()

    if not chat_id:
        return {"error": "onebot standalone send: chat_id is required"}

    adapter = OneBotAdapter(pconfig)
    chat_type = "group" if chat_id in adapter._allowed_groups else "dm"
    segments: List[Dict[str, Any]] = []
    if chat_type == "group":
        segments.append({"type": "at", "data": {"qq": "all"}})
        segments.append({"type": "text", "data": {"text": " "}})
    segments.append({"type": "text", "data": {"text": _strip_markdown(message)}})

    action = "send_group_msg" if chat_type == "group" else "send_private_msg"
    params: Dict[str, Any] = {"message": segments}
    if chat_type == "group":
        params["group_id"] = chat_id
    else:
        params["user_id"] = chat_id

    connected: asyncio.Future = asyncio.get_running_loop().create_future()
    client_box: Dict[str, Any] = {}

    async def _handler(websocket: ServerConnection) -> None:
        try:
            if websocket.request and websocket.request.path != path:
                await websocket.close(code=1008, reason="wrong path")
                return
            if token:
                header = ""
                try:
                    header = websocket.request.headers.get("Authorization", "") or ""
                except Exception:
                    header = ""
                supplied = (
                    header[7:].strip()
                    if header.lower().startswith("bearer ")
                    else header.strip()
                )
                if supplied != token:
                    await websocket.close(code=1008, reason="unauthorized")
                    return
            client_box["ws"] = websocket
            if not connected.done():
                connected.set_result(True)
            await asyncio.sleep(60)  # keep the socket open for the send
        except Exception:
            if not connected.done():
                connected.set_exception(RuntimeError("standalone handshake failed"))

    server = None
    try:
        server = await serve(
            _handler, host, port, max_size=16 * 1024 * 1024, close_timeout=2
        )
    except Exception as exc:
        return {"error": f"onebot standalone send: listen failed: {exc}"}

    try:
        await asyncio.wait_for(connected, timeout=20.0)
        websocket = client_box.get("ws")
        if websocket is None:
            return {"error": "onebot standalone send: NapCat did not connect"}
        echo = adapter._next_echo()
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        adapter._pending[echo] = future
        await websocket.send(
            json.dumps({"action": action, "params": params, "echo": echo}, ensure_ascii=False)
        )
        data = await asyncio.wait_for(future, timeout=20.0)
        return {
            "success": True,
            "platform": "onebot",
            "chat_id": chat_id,
            "message_id": _onebot_id((data or {}).get("message_id")) or "",
        }
    except asyncio.TimeoutError:
        return {"error": "onebot standalone send: NapCat did not connect in time"}
    except Exception as exc:
        return {"error": f"onebot standalone send failed: {exc}"}
    finally:
        if server is not None:
            try:
                server.close()
                await server.wait_closed()
            except Exception:
                pass


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system at startup."""
    ctx.register_platform(
        name="onebot",
        label="QQ (NapCat / OneBot 11)",
        adapter_factory=lambda cfg: OneBotAdapter(cfg),
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=is_connected,
        required_env=[],
        install_hint="websockets is already a Hermes dependency",
        env_enablement_fn=_env_enablement,
        cron_deliver_env_var="ONEBOT_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        allowed_users_env="ONEBOT_ALLOWED_PRIVATE_USERS",
        allow_all_env="ONEBOT_ALLOW_ALL_USERS",
        max_message_length=MAX_MESSAGE_CODEPOINTS,
        emoji="🐱",
        # QQ numbers are the user identity — treat as PII for redaction.
        pii_safe=False,
        allow_update_command=True,
        platform_hint=(
            "You are chatting via QQ through NapCat (OneBot 11). "
            "QQ does not render markdown — use plain text only. "
            "Keep replies concise; long replies are split into ~1500-character "
            "messages. In groups you are only addressed when the user @s you. "
            "You can send images by replying with a local image file path."
        ),
    )
