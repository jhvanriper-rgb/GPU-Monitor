#!/usr/bin/env python3
"""GPU Monitor — a small desktop window with live stats for all NVIDIA GPUs.

Reads data from `nvidia-smi` (bundled with the NVIDIA driver), so the only
requirement is Python 3 with Tkinter (both ship with stock installs).

Usage:
    python3 gpu_monitor.py                 # open the monitor
    python3 gpu_monitor.py --interval 2    # refresh every 2 s
    python3 gpu_monitor.py --topmost       # start always-on-top
    python3 gpu_monitor.py --once          # print one JSON snapshot and exit

Defaults (power caps, fan max RPM, refresh rate, colour thresholds ...) live in
settings.json next to this file. It is created on first run and can be edited.
"""
from __future__ import annotations

import argparse
import copy
import csv
import io
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings (settings.json next to this script)
# ---------------------------------------------------------------------------

SETTINGS_PATH = Path(os.path.dirname(os.path.abspath(__file__))) / "settings.json"

DEFAULT_SETTINGS: dict = {
    # RPM that counts as 100 % on the fan row. null = learn it: the highest
    # RPM ever seen is saved here automatically. Set a number to pin it.
    "fan_max_rpm": None,
    # (chip-name substring, fan label) of the fan cooling the passive card,
    # as shown by `sensors -j`.
    "fan_sensor": ["it8792", "fan1"],
    # Default value of each card's power-cap box. Keys are a part of the GPU
    # name (case-insensitive) or its PCI id. A successful Apply updates it.
    "power_caps": {"Tesla": 175},
    "interval": 1.0,            # seconds between refreshes
    "topmost": False,           # start always-on-top
    "temp_warn_c": 60,          # temperature turns amber at / above this
    "temp_hot_c": 78,           # ... and red at / above this
    "ctx_warn": 0.75,           # context fill turns amber at / above this
    "ctx_hot": 0.90,            # ... and red at / above this
}

SETTINGS: dict = copy.deepcopy(DEFAULT_SETTINGS)
_settings_writable = True
_settings_dirty = False
_settings_saved_at = 0.0


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def save_settings() -> None:
    """Write settings.json atomically (never over a file we couldn't parse)."""
    global _settings_dirty, _settings_saved_at
    if not _settings_writable:
        return
    try:
        tmp = SETTINGS_PATH.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(SETTINGS, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, SETTINGS_PATH)
        _settings_dirty = False
        _settings_saved_at = time.time()
    except OSError as exc:
        print(f"warning: could not save {SETTINGS_PATH}: {exc}", file=sys.stderr)


def save_settings_soon() -> None:
    """Save now, or at most every 30 s when it keeps changing."""
    global _settings_dirty
    _settings_dirty = True
    if time.time() - _settings_saved_at >= 30:
        save_settings()


def flush_settings() -> None:
    if _settings_dirty:
        save_settings()


def load_settings() -> None:
    """Load settings.json over the defaults; create it on first run."""
    global _settings_writable
    if not SETTINGS_PATH.exists():
        save_settings()
        return
    try:
        data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("top level must be a JSON object")
    except (OSError, ValueError) as exc:
        _settings_writable = False      # leave the user's file alone
        print(f"warning: ignoring {SETTINGS_PATH}: {exc}. Using defaults and "
              f"not overwriting the file.", file=sys.stderr)
        return
    for key, val in data.items():
        default = DEFAULT_SETTINGS.get(key)
        if key not in DEFAULT_SETTINGS:
            SETTINGS[key] = val                     # keep unknown keys as-is
        elif key == "fan_max_rpm":
            SETTINGS[key] = val if (val is None or (_is_num(val) and val > 0)) else None
        elif key == "fan_sensor":
            if (isinstance(val, list) and len(val) == 2
                    and all(isinstance(x, str) for x in val)):
                SETTINGS[key] = val
        elif key == "power_caps":
            if isinstance(val, dict):
                SETTINGS[key] = {str(k): v for k, v in val.items() if _is_num(v) and v > 0}
        elif key == "topmost":
            if isinstance(val, bool):
                SETTINGS[key] = val
        elif _is_num(val) and _is_num(default):
            SETTINGS[key] = val


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------

FIELDS = (
    "name,index,pci.bus_id,uuid,driver_version,"
    "memory.total,memory.used,memory.free,"
    "temperature.gpu,"
    "power.draw,power.limit,"
    "utilization.gpu,utilization.memory,"
    "clocks.current.graphics,clocks.current.memory,"
    "fan.speed"
)

_NOT_NUM = {"", "n/a", "[n/a]", "[not supported]", "[unknown]"}

# When a card reports no fan speed (passively cooled), show this label
# on its fan row instead of "n/a", plus the live speed of the case fan
# that cools it (read from `sensors --json`).
# For a different label per card, add a name-substring -> text entry to
# FAN_NOTES (e.g. "P40": "Sys Fan 3").
SILENT_FAN_LABEL = "Pump"
FAN_NOTES: dict[str, str] = {}


def fan_ratio(rpm: float) -> float:
    """Fan speed as a 0..1 fraction of the saved max RPM (learned if unset)."""
    ref = SETTINGS.get("fan_max_rpm")
    if not _is_num(ref) or ref <= 0 or rpm > ref:
        if rpm <= 0:
            return 0.0
        ref = SETTINGS["fan_max_rpm"] = round(rpm)
        save_settings_soon()
    return max(0.0, min(1.0, rpm / ref))


def fan_note(name: str | None) -> str:
    for needle, text in FAN_NOTES.items():
        if needle.lower() in (name or "").lower():
            return text
    return SILENT_FAN_LABEL


def read_sys_fan_rpm() -> float | None:
    """Current RPM of the configured case fan, or None if unreadable."""
    try:
        p = subprocess.run(["sensors", "-j"], capture_output=True, text=True,
                           timeout=4)
        if p.returncode != 0:
            return None
        data = json.loads(p.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    chip_key, fan_key = SETTINGS["fan_sensor"]
    for chip, fields in data.items():
        if chip_key.lower() not in chip.lower():
            continue
        entry = fields.get(fan_key)
        if entry and entry.get(f"{fan_key}_input") is not None:
            return float(entry[f"{fan_key}_input"])
    return None


# ---------------------------------------------------------------------------
# Live tokens/s of the running LLM, read from the serving engine (the
# llama.cpp server Ollama spawns per model). Its port is dynamic, so it's
# discovered by probing loopback listeners and cached briefly.
# ---------------------------------------------------------------------------

_engine_ports: list[str] = []
_engine_ports_ts: float = 0.0
_ENGINE_HIT_TTL = 60.0
_ENGINE_MISS_TTL = 15.0


def _loopback_ports() -> list[str]:
    try:
        p = subprocess.run(["ss", "-tln"], capture_output=True, text=True,
                           timeout=3)
    except (OSError, subprocess.SubprocessError):
        return []
    ports: list[str] = []
    for line in p.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 4:
            local = parts[3]
            if local.startswith("127.") or local.startswith("[::1]"):
                port = local.rsplit(":", 1)[-1]
                if port.isdigit() and port not in ports:
                    ports.append(port)
    return ports


def _engine_slots(port: str) -> list:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots",
                                timeout=1) as r:
        data = json.loads(r.read().decode("utf-8", "replace"))
    return data if isinstance(data, list) else []


def _find_engine_ports() -> list[str]:
    """All loopback ports whose /slots endpoint is the llama.cpp slot API."""
    global _engine_ports, _engine_ports_ts
    ttl = _ENGINE_HIT_TTL if _engine_ports else _ENGINE_MISS_TTL
    if time.time() - _engine_ports_ts < ttl:
        return _engine_ports
    found: list[str] = []
    for port in _loopback_ports():
        try:
            probe = _engine_slots(port)
        except Exception:  # noqa: BLE001 — closed or not an engine: try next
            continue
        if probe and isinstance(probe, list) and len(probe) > 0 and isinstance(probe[0], dict) and                 ("is_processing" in probe[0] or "n_ctx" in probe[0]):
            found.append(port)
    _engine_ports, _engine_ports_ts = found, time.time()
    return found


_slot_prev: dict = {}  # (port, slot id) -> (time, id_task, n_decoded)


def _slot_tps(port: str, slot: dict) -> float | None:
    """Generation speed of one busy slot.

    Prefers the per-token time (`t_token`) when the server reports it; newer
    llama.cpp builds dropped that field, so fall back to the change in the
    decoded-token count between two polls."""
    try:
        t = slot.get("t_token")
        if t:
            return 1000.0 / float(t)
    except (TypeError, ValueError):
        pass
    nt = slot.get("next_token")
    if isinstance(nt, list):
        nt = nt[0] if nt else None
    try:
        n = float(nt["n_decoded"]) if isinstance(nt, dict) else None
    except (KeyError, TypeError, ValueError):
        n = None
    if n is None:
        return None
    key = (port, slot.get("id"))
    now, task = time.time(), slot.get("id_task")
    prev = _slot_prev.get(key)
    _slot_prev[key] = (now, task, n)
    if prev is None or prev[1] != task or n < prev[2] or now <= prev[0]:
        return None
    if n == prev[2]:
        return None  # still reading the prompt, nothing generated yet
    return (n - prev[2]) / (now - prev[0])


def read_live_tps() -> dict | None:
    """Busiest active slot across all engines:
    {"active": True, "tps": float, "slot": id}.
    Engine found but idle: {"active": False}. No engine: None."""
    global _engine_ports, _engine_ports_ts
    for _attempt in range(2):
        best = None
        reached = False
        dead = False
        for port in _find_engine_ports():
            try:
                slots = _engine_slots(port)
            except Exception:  # noqa: BLE001 — engine gone (restart?)
                dead = True
                continue
            reached = True
            for slot in slots:
                if not isinstance(slot, dict) or not slot.get("is_processing"):
                    continue
                tps = _slot_tps(port, slot)
                if tps is not None and (best is None or best[1] < tps):
                    best = (slot.get("id"), tps)
        if best is not None:
            return {"active": True, "tps": best[1], "slot": best[0]}
        if reached:
            return {"active": False}
        if dead and _attempt == 0:
            _engine_ports, _engine_ports_ts = [], 0.0  # force a fresh probe
            continue
        return None
    return None


_ctx_last: dict = {}  # (port, slot id) -> last context size seen while busy


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _slot_ctx_used(port: str, slot: dict) -> int | None:
    """Tokens currently held in a slot's context.

    Older llama.cpp builds report it directly (n_past and friends). Newer ones
    give the prompt length plus tokens generated so far, only while the slot is
    busy; when idle, show the size it ended at (the cache is still there)."""
    for key in ("n_past", "n_tokens", "n_ctx_used"):
        if _is_num(slot.get(key)):
            return int(slot[key])
    key = (port, slot.get("id"))
    if slot.get("is_processing") and _is_num(slot.get("n_prompt_tokens")):
        nt = slot.get("next_token")
        if isinstance(nt, list):
            nt = nt[0] if nt else None
        decoded = nt.get("n_decoded") if isinstance(nt, dict) else 0
        used = int(slot["n_prompt_tokens"]) + (int(decoded) if _is_num(decoded) else 0)
        _ctx_last[key] = used
        return used
    return _ctx_last.get(key)


def _engine_kv_tokens(port: str) -> int | None:
    """KV-cache token count from /metrics (server started with --metrics)."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",
                                    timeout=1) as r:
            body = r.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — metrics endpoint off or unreachable
        return None
    for line in body.splitlines():
        if line.startswith("llamacpp:kv_cache_tokens"):
            try:
                return int(float(line.rsplit(None, 1)[-1]))
            except ValueError:
                return None
    return None


def read_ctx_usage() -> dict | None:
    """Context fill of the busiest slot across engines:
    {"used": int|None, "total": int, "slot": id, "active": bool}.
    `used` is None when this server build doesn't expose it. None: no engine."""
    best = None
    for port in _find_engine_ports():
        try:
            slots = _engine_slots(port)
        except Exception:  # noqa: BLE001
            continue
        kv, kv_tried = None, False
        for slot in slots:
            if not isinstance(slot, dict):
                continue
            total = slot.get("n_ctx")
            if not isinstance(total, (int, float)) or not total:
                continue
            used = _slot_ctx_used(port, slot)
            if used is None:
                if not kv_tried:
                    kv, kv_tried = _engine_kv_tokens(port), True
                used = kv  # server-wide count; exact with a single slot
            cand = {"used": used, "total": int(total), "slot": slot.get("id"),
                    "active": bool(slot.get("is_processing"))}
            rank = (cand["active"], cand["used"] or 0)
            if best is None or rank > best[0]:
                best = (rank, cand)
    return best[1] if best else None


def _trim(obj, limit: int = 60):
    """Copy of a JSON value with long strings shortened (keeps prompt text
    out of the debug dump)."""
    if isinstance(obj, str):
        return obj if len(obj) <= limit else obj[:limit] + f"…(+{len(obj) - limit})"
    if isinstance(obj, list):
        return [_trim(v, limit) for v in obj]
    if isinstance(obj, dict):
        return {k: _trim(v, limit) for k, v in obj.items()}
    return obj


def debug_engines() -> None:
    """Print what each discovered llama.cpp server returns, to work out which
    fields carry the context-token count."""
    ports = _find_engine_ports()
    if not ports:
        print("no llama.cpp server found on loopback /slots")
        return
    for port in ports:
        print(f"=== port {port} /slots ===")
        try:
            print(json.dumps(_trim(_engine_slots(port)), indent=2))
        except Exception as exc:  # noqa: BLE001
            print(f"error: {exc}")
        print(f"=== port {port} /metrics (kv / token / ctx lines) ===")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics",
                                        timeout=2) as r:
                body = r.read().decode("utf-8", "replace")
            hits = [ln for ln in body.splitlines()
                    if not ln.startswith("#") and
                    any(w in ln for w in ("kv", "token", "ctx"))]
            print("\n".join(hits) if hits else "(no matching lines)")
        except Exception as exc:  # noqa: BLE001
            print(f"unavailable: {exc}")


def _fmt_tok(n: float) -> str:
    return f"{n / 1000:.1f}k" if n >= 1000 else f"{n:.0f}"


# LiteLLM proxy metrics (cumulative output-token counter, /metrics).
# Only usable once the proxy's prometheus callback is enabled — until then
# the live engine above covers the readout.
# Source preference: "auto" = LiteLLM if available else live engine;
# "engine" = always the live per-token speed; "litellm" = always the proxy.
LLM_SOURCE = "auto"
LLM_METRICS_URL = "http://127.0.0.1:4000/metrics"
LLM_METRICS_KEY_FILE = os.path.expanduser("~/litellm/config.yaml")
LLM_RATE_WINDOW = 10.0  # seconds of history used to average tokens/s


def _llm_metrics_key() -> str | None:
    """Master key from the LiteLLM config (never printed or stored elsewhere)."""
    try:
        with open(LLM_METRICS_KEY_FILE) as fh:
            m = re.search(r"^\s*master_key:\s*(\S+)", fh.read(), flags=re.M)
        return m.group(1) if m else None
    except OSError:
        return None


def read_llm_stats() -> dict | None:
    """Cumulative output-token counter from LiteLLM, or None if unavailable."""
    req = urllib.request.Request(LLM_METRICS_URL)
    key = _llm_metrics_key()
    if key:
        req.add_header("Authorization", f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = resp.read().decode("utf-8", "replace")
    except Exception:  # noqa: BLE001 — endpoint down or not enabled yet
        return None
    total = 0.0
    found = False
    # Only the real counter: Prometheus also emits a `_created` series holding
    # a unix timestamp (~1.7e9). A prefix match summed it in and produced a
    # huge tok/s spike whenever a new model/label series first appeared.
    names = ("litellm_output_tokens_metric_total", "litellm_output_tokens_metric")
    for line in body.splitlines():
        if line.startswith("#"):
            continue
        parts = line.split(None, 1)
        if not parts or parts[0].split("{", 1)[0] not in names:
            continue
        try:
            total += float(line.rsplit(None, 1)[-1])
            found = True
        except ValueError:
            pass
    if not found:
        return None
    return {"output_tokens": total}


class TokenRate:
    """tokens/s over a trailing window from a monotonically rising counter."""

    def __init__(self, window: float = LLM_RATE_WINDOW):
        self.window = window
        self.samples: list[tuple[float, float]] = []

    def update(self, counter: float | None) -> float | None:
        if counter is None:
            self.samples = []
            return None
        if self.samples and counter < self.samples[-1][1]:
            self.samples = []  # counter reset (proxy restarted)
        now = time.time()
        self.samples.append((now, counter))
        cutoff = now - self.window
        self.samples = [s for s in self.samples if s[0] >= cutoff]
        if len(self.samples) < 2:
            return 0.0
        (t0, c0), (t1, c1) = self.samples[0], self.samples[-1]
        if t1 - t0 <= 0:
            return 0.0
        return max(0.0, (c1 - c0) / (t1 - t0))


def read_llm_metrics() -> dict | None:
    """Tokens/s source of record:
    {"source": "litellm", "counter": float}  or
    {"source": "engine", "tok": float}  or  None."""
    stats = None if LLM_SOURCE == "engine" else read_llm_stats()
    if LLM_SOURCE != "litellm" and stats is not None:
        return {"source": "litellm", "counter": stats["output_tokens"]}
    live = None if LLM_SOURCE == "litellm" else read_live_tps()
    if live is not None:
        return {"source": "engine",
                "tok": live["tps"] if live.get("active") else 0.0}
    if stats is not None:
        return {"source": "litellm", "counter": stats["output_tokens"]}
    return None


def _num(tok: str):
    """Parse an nvidia-smi cell; None when not a plain number (e.g. [N/A])."""
    t = tok.strip().lower()
    if t in _NOT_NUM:
        return None
    try:
        return float(t)
    except ValueError:
        return None


def _fmt_mb(mb) -> str:
    if mb is None:
        return "n/a"
    if mb >= 1024:
        return f"{mb / 1024:.1f} GiB"
    return f"{mb:.0f} MiB"


def query_gpus() -> list[dict]:
    """One nvidia-smi call, one row per GPU."""
    cmd = ["nvidia-smi", f"--query-gpu={FIELDS}", "--format=csv,noheader,nounits"]
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.strip() or f"nvidia-smi exited {p.returncode}")
    cols = FIELDS.split(",")
    gpus = []
    for row in csv.reader(io.StringIO(p.stdout)):
        row = [c.strip() for c in row]
        if len(row) != len(cols):
            continue
        d = dict(zip(cols, row))
        idx = _num(d["index"])
        gpus.append({
            "name": d["name"] or f"GPU {d['index']}",
            "index": int(idx) if idx is not None else len(gpus),
            "pci": d["pci.bus_id"],
            "uuid": d["uuid"],
            "driver": d["driver_version"],
            "mem_total": _num(d["memory.total"]),
            "mem_used": _num(d["memory.used"]),
            "mem_free": _num(d["memory.free"]),
            "temp": _num(d["temperature.gpu"]),
            "power": _num(d["power.draw"]),
            "power_limit": _num(d["power.limit"]),
            "util": _num(d["utilization.gpu"]),
            "mem_util": _num(d["utilization.memory"]),
            "gfx_clock": _num(d["clocks.current.graphics"]),
            "mem_clock": _num(d["clocks.current.memory"]),
            "fan": _num(d["fan.speed"]),
        })
    if not gpus:
        raise RuntimeError("nvidia-smi returned no GPU rows")

    # Some vGPU slices report [N/A] for utilization on the query interface;
    # pmon reads the same counters through another path — use it as a fallback.
    missing = [g for g in gpus if g["util"] is None or g["mem_util"] is None]
    if missing:
        for g in missing:
            sm, mem = _pmon_util(g["index"])
            if g["util"] is None and sm is not None:
                g["util"] = sm
            if g["mem_util"] is None and mem is not None:
                g["mem_util"] = mem

    # Attach the cooling case fan's live speed to any passively-cooled card.
    if any(g["fan"] is None for g in gpus):
        rpm = read_sys_fan_rpm()
        if rpm is not None:
            for g in gpus:
                if g["fan"] is None:
                    g["sys_fan_rpm"] = rpm

    gpus.sort(key=lambda g: g["index"])
    return gpus


def _pmon_util(idx: int) -> tuple:
    """Return (sm_util, mem_util) for one GPU via `nvidia-smi pmon`, else Nones."""
    try:
        p = subprocess.run(
            ["nvidia-smi", "-i", str(idx), "pmon", "-c", "1", "-s", "cu"],
            capture_output=True, text=True, timeout=6)
    except (OSError, subprocess.SubprocessError):
        return (None, None)
    if p.returncode != 0:
        return (None, None)
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("==="):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        return (_num(parts[1]), _num(parts[2]))
    return (None, None)


def query_top_process(gpus: list[dict]) -> dict[int, tuple[str, float] | None]:
    """Heaviest VRAM process per card: {gpu_index: (name, mem_mb) or None}."""
    by_uuid: dict[str, tuple[str, float]] | None = None
    cmd = ["nvidia-smi",
           "--query-compute-apps=pid,process_name,used_memory,gpu_uuid",
           "--format=csv,noheader,nounits"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
        if p.returncode == 0:
            by_uuid = {}
            for row in csv.reader(io.StringIO(p.stdout)):
                if len(row) < 4:
                    continue
                try:
                    mem = float(row[2])
                except ValueError:
                    continue
                name = row[1].strip() or f"pid {row[0]}"
                uuid = row[3].strip()
                cur = by_uuid.get(uuid)
                if cur is None or mem > cur[1]:
                    by_uuid[uuid] = (name, mem)
    except (OSError, subprocess.SubprocessError):
        pass

    out: dict[int, tuple[str, float] | None] = {}
    for g in gpus:
        if by_uuid is not None:
            out[g["index"]] = by_uuid.get(g.get("uuid") or "")
        else:
            out[g["index"]] = _top_process_on_card(g["index"])
    return out


def _top_process_on_card(idx: int) -> tuple[str, float] | None:
    """Fallback: heaviest VRAM process on one card (via -i), or None."""
    cmd = ["nvidia-smi", "-i", str(idx),
           "--query-compute-apps=pid,process_name,used_memory",
           "--format=csv,noheader,nounits"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
        if p.returncode != 0:
            return None
    except (OSError, subprocess.SubprocessError):
        return None
    best = None
    for row in csv.reader(io.StringIO(p.stdout)):
        if len(row) < 3:
            continue
        try:
            mem = float(row[2])
        except ValueError:
            continue
        name = row[1].strip() or f"pid {row[0]}"
        if best is None or mem > best[1]:
            best = (name, mem)
    return best


# NVML clock-throttle reason bits -> (label, severity 1=notice 2=warn 3=bad).
# Idle (0x1), application clocks (0x2) and display clocks (0x100) are normal
# and deliberately left out.
_THROTTLE_BITS = (
    (0x80, "Power brake", 3),
    (0x40, "HW thermal", 3),
    (0x08, "HW slowdown", 3),
    (0x20, "Thermal", 2),
    (0x04, "Power cap", 2),
    (0x10, "Sync boost", 1),
)
_THROTTLE_IGNORED = 0x1 | 0x2 | 0x100
_throttle_field: str | None = None  # whichever query name this driver accepts


def query_throttle() -> dict[int, int]:
    """Active throttle-reason bitmask per GPU index. Kept as its own
    nvidia-smi call so an unsupported field can't break the main query.
    Newer drivers call the field clocks_event_reasons; older ones
    clocks_throttle_reasons."""
    global _throttle_field
    fields = ([_throttle_field] if _throttle_field else
              ["clocks_event_reasons.active", "clocks_throttle_reasons.active"])
    for f in fields:
        try:
            p = subprocess.run(
                ["nvidia-smi", f"--query-gpu=index,{f}",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=6)
        except (OSError, subprocess.SubprocessError):
            return {}
        if p.returncode != 0:
            continue
        out: dict[int, int] = {}
        for row in csv.reader(io.StringIO(p.stdout)):
            if len(row) < 2:
                continue
            idx = _num(row[0])
            try:
                mask = int(row[1].strip(), 16)
            except ValueError:
                continue
            if idx is not None:
                out[int(idx)] = mask
        _throttle_field = f
        return out
    return {}


def throttle_info(mask: int | None) -> tuple[list[str], int, bool]:
    """(labels of real throttling, worst severity 0-3, card is idle)."""
    if mask is None:
        return [], 0, False
    labels, worst = [], 0
    for bit, label, sev in _THROTTLE_BITS:
        if mask & bit:
            labels.append(label)
            worst = max(worst, sev)
    return labels, worst, bool(mask & 0x1) and not labels


def snapshot() -> dict:
    gpus = query_gpus()
    tops = query_top_process(gpus)
    throttle = query_throttle()
    for g in gpus:
        top = tops.get(g["index"])
        g["top_process"] = top[0] if top else None
        g["top_mem"] = top[1] if top else None
        g["throttle"] = throttle.get(g["index"])
    return {"t": time.time(), "gpus": gpus}


# ---------------------------------------------------------------------------
# UI layer
# ---------------------------------------------------------------------------

BG = "#14161d"
CARD = "#1d212c"
TRACK = "#2a3040"
TEXT = "#e8ebf2"
SUB = "#8b93a7"
ACCENT = "#4da3ff"
GOOD = "#3ddc84"
WARN = "#ffb545"
BAD = "#ff5c5c"


def temp_color(temp: float | None) -> str:
    if temp is None:
        return SUB
    if temp < SETTINGS["temp_warn_c"]:
        return GOOD
    if temp < SETTINGS["temp_hot_c"]:
        return WARN
    return BAD


class Bar:
    """A small horizontal meter drawn on a canvas."""

    def __init__(self, parent, w=62, h=8, fill=ACCENT):
        import tkinter as tk

        self.w, self.h = w, h
        self.cv = tk.Canvas(parent, width=w, height=h, bg=CARD,
                            highlightthickness=0)
        self.cv.create_rectangle(0, 0, w, h, fill=TRACK, outline="")
        self.fill_id = self.cv.create_rectangle(0, 0, 0, h, fill=fill, outline="")

    def set(self, frac: float | None, color: str | None = None):
        if color is not None:
            self.cv.itemconfig(self.fill_id, fill=color)
        if frac is None:
            self.cv.coords(self.fill_id, 0, 0, 0, self.h)
            return
        frac = max(0.0, min(1.0, frac))
        width = self.w * frac
        self.cv.coords(self.fill_id, 0, 0, max(width, 2 if frac > 0 else 0), self.h)


def _cap_key_matches(key: str, name: str, pci: str) -> bool:
    k = key.lower()
    return k == (pci or "").lower() or (k != "" and k in (name or "").lower())


def power_cap_default(name: str, pci: str, current_limit) -> float:
    """Value for a card's cap box: settings.json, else the card's own limit."""
    for key, val in (SETTINGS.get("power_caps") or {}).items():
        if _cap_key_matches(key, name, pci) and _is_num(val):
            return val
    return current_limit or 0


class GpuCard:
    def __init__(self, parent, gpu: dict, on_set=None):
        import tkinter as tk

        self.f = tk.Frame(parent, bg=CARD, highlightbackground="#262c3a",
                          highlightthickness=1, bd=1)
        p = 10
        pad = dict(padx=p, pady=2)
        self._pci = gpu["pci"]
        self._on_set = on_set

        self.name = tk.Label(self.f, text=gpu["name"], bg=CARD, fg=TEXT, anchor="w",
                             font=("Helvetica", 11, "bold"))
        self.name.pack(fill="x", padx=p, pady=(p, 0))
        self.pci = tk.Label(self.f, text=f"{gpu['pci']}   ·   GPU {gpu['index']}",
                            bg=CARD, fg=SUB, anchor="w", font=("Helvetica", 8))
        self.pci.pack(fill="x", padx=p, pady=(0, 4))

        self.rows: list[tuple[str, tk.Label, Bar | None]] = []
        self.peak: dict[str, float] = {}  # session high-water marks per metric
        for label, key in [
            ("VRAM", "mem"), ("Temp", "temp"), ("Power", "power"),
            ("GPU", "util"), ("Mem", "mem_util"), ("Clock", "clock"),
            ("Fan", "fan"), ("Limit", "throttle"), ("Proc", "proc"),
        ]:
            row_f = tk.Frame(self.f, bg=CARD)
            row_f.pack(fill="x", pady=1)
            lbl = tk.Label(row_f, text=label, bg=CARD, fg=SUB, anchor="e",
                           width=5, font=("Helvetica", 8, "bold"))
            lbl.pack(side="left", padx=(p, 4), pady=1)
            val = tk.Label(row_f, text="—", bg=CARD, fg=TEXT, anchor="w")
            val.config(font=("Helvetica", 9))
            val.pack(side="left", padx=(0, 6), pady=1)
            bar = None
            if key not in ("proc", "throttle"):
                bar = Bar(row_f)
                bar.cv.pack(side="right", padx=(2, p), pady=2)
            self.rows.append((key, val, bar))

        # Per-GPU power cap control ----------------------------------------
        ctl = tk.Frame(self.f, bg=CARD)
        ctl.pack(fill="x", pady=(6, 0))
        tk.Label(ctl, text="Power Cap", bg=CARD, fg=SUB, anchor="w",
                 font=("Helvetica", 8, "bold")).pack(side="left", padx=(p, 6))
        self.gpu_name = gpu["name"]
        default = power_cap_default(gpu["name"], gpu["pci"], gpu.get("power_limit"))
        self.cap_var = tk.StringVar(value=f"{default:.0f}" if default else "")
        self.cap_entry = tk.Entry(ctl, textvariable=self.cap_var, width=5,
                                  justify="right", bg=TRACK, fg=TEXT,
                                  insertbackground=TEXT, relief="flat",
                                  highlightthickness=1,
                                  highlightbackground="#262c3a",
                                  highlightcolor=ACCENT, font=("Helvetica", 9))
        self.cap_entry.pack(side="left")
        tk.Label(ctl, text="W", bg=CARD, fg=SUB,
                 font=("Helvetica", 9)).pack(side="left", padx=(3, 6))
        self.cap_btn = tk.Button(ctl, text="Apply", command=self._apply_cap,
                                 bg=TRACK, fg=TEXT, activebackground=ACCENT,
                                 activeforeground=BG, relief="flat", bd=0,
                                 padx=10, font=("Helvetica", 8, "bold"),
                                 cursor="hand2", highlightthickness=0)
        self.cap_btn.pack(side="left")
        self.cap_msg = tk.Label(self.f, text="", bg=CARD, fg=SUB, anchor="w",
                                justify="left", wraplength=250,
                                font=("Helvetica", 8))
        self.cap_msg.pack(fill="x", padx=p, pady=(3, p))
        self.cap_entry.bind("<Return>", lambda _e: self._apply_cap())

    def _apply_cap(self):
        text = self.cap_var.get().strip()
        try:
            watts = int(float(text))
        except ValueError:
            self.cap_msg.config(text="Enter a whole number of watts.", fg=BAD)
            return
        if watts <= 0:
            self.cap_msg.config(text="Enter a whole number of watts.", fg=BAD)
            return
        if self._on_set:
            self._on_set(self, self._pci, watts)

    def peak_ratio(self, key: str, value: float | None) -> float | None:
        if value is None:
            return None
        peak = self.peak.get(key)
        if peak is None or value > peak:
            peak = self.peak[key] = max(value, 1e-9)
        return value / peak

    def update(self, gpu: dict):
        for key, val, bar in self.rows:
            if key == "mem":
                total, used = gpu["mem_total"], gpu["mem_used"]
                val.config(text=f"{_fmt_mb(used)} / {_fmt_mb(total)}")
                if bar:
                    bar.set(used / total if total else None)
            elif key == "temp":
                t = gpu["temp"]
                val.config(text=f"{t:.0f} °C" if t is not None else "n/a",
                           fg=temp_color(t))
                if bar:
                    frac = None if t is None else max(0.0, min(100.0, t)) / 100.0
                    bar.set(frac, temp_color(t))
            elif key == "power":
                pw, lim = gpu["power"], gpu["power_limit"]
                if pw is None:
                    val.config(text="n/a")
                elif lim:
                    val.config(text=f"{pw:.1f} / {lim:.0f} W")
                else:
                    val.config(text=f"{pw:.1f} W")
                if bar:
                    bar.set(pw / lim if (pw and lim) else None)
            elif key == "util":
                u = gpu["util"]
                val.config(text=f"{u:.0f} %" if u is not None else "n/a")
                if bar:
                    bar.set(u / 100 if u is not None else None)
            elif key == "mem_util":
                u = gpu["mem_util"]
                val.config(text=f"{u:.0f} %" if u is not None else "n/a")
                if bar:
                    bar.set(u / 100 if u is not None else None)
            elif key == "clock":
                g, m = gpu["gfx_clock"], gpu["mem_clock"]
                bits = []
                if g is not None:
                    bits.append(f"{g:.0f} MHz")
                if m is not None:
                    bits.append(f"mem {m / 1000:.1f} GHz" if m >= 1000 else f"mem {m:.0f} MHz")
                val.config(text=" · ".join(bits) if bits else "n/a")
                if bar:
                    bar.set(self.peak_ratio("clock", g))
            elif key == "fan":
                f = gpu["fan"]
                if f is not None:
                    val.config(text=f"{f:.0f} %")
                    if bar:
                        bar.set(f / 100)
                else:
                    label = fan_note(gpu["name"])
                    rpm = gpu.get("sys_fan_rpm")
                    if rpm is not None:
                        ratio = fan_ratio(rpm)
                        val.config(text=f"{label} · {ratio * 100.0:.0f} %", fg=SUB)
                        if bar:
                            bar.set(ratio)
                    else:
                        val.config(text=label, fg=SUB)
                        if bar:
                            bar.set(None)
            elif key == "throttle":
                labels, sev, idle = throttle_info(gpu.get("throttle"))
                if gpu.get("throttle") is None:
                    val.config(text="n/a", fg=SUB)
                elif labels:
                    val.config(text=" + ".join(labels),
                               fg=BAD if sev >= 3 else WARN if sev == 2 else SUB)
                else:
                    val.config(text="idle" if idle else "none",
                               fg=SUB if idle else GOOD)
            elif key == "proc":
                if gpu.get("top_process"):
                    name = gpu["top_process"].rsplit("/", 1)[-1]
                    val.config(text=f"{name}  ({_fmt_mb(gpu['top_mem'])})", fg=ACCENT)
                else:
                    val.config(text="idle", fg=SUB)


class MonitorApp:
    def __init__(self, root, interval: float, topmost: bool):
        import tkinter as tk

        self.root = root
        self.interval = interval
        self.q: queue.Queue = queue.Queue()
        self.last_ok: dict | None = None
        self.last_err: str | None = None
        self._llm_rate = TokenRate()
        self.llm: dict | None = None
        self.ctx: dict | None = None
        self._cards: list[GpuCard] = []
        self._topmost = False

        root.title("GPU Monitor")
        try:
            icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "icon.png")
            full = tk.PhotoImage(file=icon_path)
            self._icons = [full.subsample(4, 4), full.subsample(16, 16), full.subsample(32, 32)]
            root.iconphoto(True, *self._icons)
        except Exception:
            pass
        root.configure(bg=BG)
        root.resizable(False, False)

        # Header -----------------------------------------------------------
        head = tk.Frame(root, bg=BG)
        head.pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(head, text="GPU MONITOR", bg=BG, fg=SUB,
                 font=("Helvetica", 8, "bold")).pack(side="left")
        self.pin = tk.Button(head, text="📌 pin", command=self._toggle_pin,
                             bg=BG, fg=SUB, activebackground=BG,
                             activeforeground=TEXT, relief="flat", bd=0,
                             font=("Helvetica", 8), cursor="hand2",
                             highlightthickness=0)
        self.pin.pack(side="right")

        # Body -------------------------------------------------------------
        self.body = tk.Frame(root, bg=BG)
        self.body.pack(fill="both", expand=True)

        # Status bar -------------------------------------------------------
        self.status = tk.Label(root, bg=BG, fg=SUB, anchor="w",
                               font=("Helvetica", 9))
        self.status.pack(fill="x", padx=14, pady=(4, 0))
        self.ctx_status = tk.Label(root, bg=BG, fg=SUB, anchor="w",
                                   font=("Helvetica", 9))
        self.ctx_status.pack(fill="x", padx=14, pady=(0, 10))

        root.protocol("WM_DELETE_WINDOW", self._on_close)
        if topmost:
            self._set_topmost(True)
        self.root.after(100, self._drain)

    def _set_topmost(self, on: bool):
        self._topmost = on
        try:
            self.root.attributes("-topmost", on)
        except Exception:
            pass
        self.pin.config(text="📌 pinned" if on else "📌 pin",
                        fg=ACCENT if on else SUB)

    def _toggle_pin(self):
        self._set_topmost(not self._topmost)

    # Power cap ------------------------------------------------------------
    def _set_power_limit(self, card, pci: str, watts: int):
        """Ask for authorisation (polkit) and set the cap on this GPU only."""
        smi = shutil.which("nvidia-smi")
        pk = shutil.which("pkexec")
        if smi is None:
            card.cap_msg.config(text="nvidia-smi not found.", fg=BAD)
            return
        if pk is None:
            card.cap_msg.config(
                text="pkexec not found. Install it with: sudo apt install pkexec",
                fg=BAD)
            return
        card.cap_btn.config(state="disabled")
        card.cap_msg.config(text="Waiting for authorisation…", fg=WARN)
        threading.Thread(target=self._power_worker,
                         args=(card, pk, smi, pci, watts), daemon=True).start()

    def _power_worker(self, card, pk: str, smi: str, pci: str, watts: int):
        try:
            # Range check first; reading the limits needs no privileges.
            lo = hi = None
            r = subprocess.run(
                [smi, "-i", pci, "--query-gpu=power.min_limit,power.max_limit",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                try:
                    lo, hi = (float(x) for x in r.stdout.strip().split(","))
                except ValueError:
                    lo = hi = None
            if lo is not None and not (lo <= watts <= hi):
                self.q.put(("cap", card, False,
                            f"{watts} W is outside this card's range "
                            f"({lo:.0f}–{hi:.0f} W).", None))
                return
            r = subprocess.run([pk, smi, "-i", pci, "-pl", str(watts)],
                               capture_output=True, text=True, timeout=180)
            if r.returncode == 0:
                self.q.put(("cap", card, True, f"Power Cap set to {watts} W.", watts))
            elif r.returncode in (126, 127):
                self.q.put(("cap", card, False,
                            "Authorisation cancelled or denied. No change made.",
                            None))
            else:
                err = (r.stderr or r.stdout or "unknown error").strip().splitlines()
                self.q.put(("cap", card, False, err[0] if err else "failed", None))
        except Exception as exc:  # noqa: BLE001
            self.q.put(("cap", card, False, f"Failed: {exc}", None))

    def _remember_cap(self, card, watts: int):
        """Make a successfully applied cap this card's saved default."""
        caps = SETTINGS.setdefault("power_caps", {})
        for key in [k for k in caps if _cap_key_matches(k, card.gpu_name, card._pci)]:
            del caps[key]
        caps[card.gpu_name] = watts
        save_settings()

    def _on_close(self):
        flush_settings()
        self.root.destroy()

    def _worker(self):
        while True:
            t0 = time.time()
            try:
                snap = snapshot()
                m = read_llm_metrics()
                if m is None:
                    self.llm = None
                elif m["source"] == "litellm":
                    self.llm = {"tok": self._llm_rate.update(m["counter"]),
                                "src": "litellm"}
                else:
                    self.llm = {"tok": m["tok"], "src": "engine"}
                self.ctx = read_ctx_usage()
                self.q.put(("ok", snap))
            except Exception as exc:
                self.q.put(("error", str(exc)))
            elapsed = time.time() - t0
            time.sleep(max(0.05, self.interval - elapsed))

    def _drain(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg[0] == "ok":
                    self.last_ok, self.last_err = msg[1], None
                elif msg[0] == "cap":
                    _, card, ok, text, watts = msg
                    try:
                        card.cap_btn.config(state="normal")
                        card.cap_msg.config(text=text, fg=GOOD if ok else BAD)
                    except Exception:
                        pass  # card was rebuilt meanwhile
                    if ok and watts:
                        self._remember_cap(card, watts)
                else:
                    self.last_err = msg[1]
        except queue.Empty:
            pass
        self._render()
        self.root.after(100, self._drain)

    def _render(self):
        snap = self.last_ok
        if snap is None:
            for c in self._cards:
                c.f.destroy()
            self._cards = []
            if self.last_err:
                self.status.config(text="error: " + self.last_err, fg=BAD)
            else:
                self.status.config(text="waiting for first GPU read…", fg=SUB)
            return

        gpus = snap["gpus"]
        if len(self._cards) != len(gpus):
            for c in self._cards:
                c.f.destroy()
            self._cards = []
            for g in gpus:
                card = GpuCard(self.body, g, on_set=self._set_power_limit)
                self._cards.append(card)
                card.f.pack(side="left", fill="both", expand=True, padx=6)
        for card, g in zip(self._cards, gpus):
            card.name.config(text=g["name"])
            card.pci.config(text=f"{g['pci']}   ·   GPU {g['index']}")
            card.update(g)

        driver = gpus[0].get("driver") or "?"
        when = time.strftime("%H:%M:%S", time.localtime(snap["t"]))
        if self.llm is None or self.llm.get("tok") is None:
            llm_txt, active = "LLM n/a", False
        else:
            llm_txt, active = f"LLM {self.llm['tok']:.1f} tok/s",                 self.llm["tok"] > 0
        ctx = self.ctx
        if ctx is None:
            self.ctx_status.config(text="ctx n/a", fg=SUB)
        elif ctx["used"] is None:
            self.ctx_status.config(
                text=f"ctx {'?' if ctx['active'] else 'idle'} / "
                     f"{_fmt_tok(ctx['total'])}  ·  slot {ctx['slot']}",
                fg=SUB)
        else:
            frac = ctx["used"] / ctx["total"]
            self.ctx_status.config(
                text=f"ctx {_fmt_tok(ctx['used'])} / {_fmt_tok(ctx['total'])}"
                     f"  ·  {frac * 100:.0f} %  ·  slot {ctx['slot']}",
                fg=BAD if frac >= SETTINGS["ctx_hot"]
                else WARN if frac >= SETTINGS["ctx_warn"]
                else ACCENT if ctx["active"] else SUB)
        if self.last_err:
            self.status.config(
                text=f"stale · {when} · {llm_txt} · last error: {self.last_err}",
                fg=WARN)
        else:
            self.status.config(
                text=f"driver {driver}  ·  {len(gpus)} GPUs  ·  {llm_txt}  ·  "
                     f"updated {when}  ·  {self.interval:.0f}s",
                fg=ACCENT if active else SUB)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interval", type=float, default=None,
                    help="seconds between refreshes (default: settings.json)")
    ap.add_argument("--topmost", action="store_true", help="start always-on-top")
    ap.add_argument("--once", action="store_true",
                    help="print one JSON snapshot to stdout and exit")
    ap.add_argument("--debug-slots", action="store_true",
                    help="print raw llama.cpp /slots and /metrics data and exit")
    args = ap.parse_args(argv)
    load_settings()
    interval = max(0.5, args.interval if args.interval is not None
                   else float(SETTINGS["interval"]))
    topmost = args.topmost or bool(SETTINGS["topmost"])

    if args.debug_slots:
        debug_engines()
        return 0

    if args.once:
        try:
            s = snapshot()
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        s["llm"] = read_llm_metrics()
        s["ctx"] = read_ctx_usage()
        print(json.dumps(s, indent=2))
        return 0

    if shutil.which("nvidia-smi") is None:
        print("error: nvidia-smi not found on PATH (install the NVIDIA driver)",
              file=sys.stderr)
        return 2

    import tkinter as tk  # imported late so --once works headless

    root = tk.Tk()
    app = MonitorApp(root, interval=interval, topmost=topmost)
    threading.Thread(target=app._worker, daemon=True).start()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
