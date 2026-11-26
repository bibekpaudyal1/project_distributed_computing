"""Detect this machine's hardware: CPU, memory, GPUs, disks and network interfaces.

Works on macOS (Apple Silicon / Intel), Linux (with or without NVIDIA) and Windows.
Everything is best-effort: if a probe fails we fall back to sane defaults.
"""
from __future__ import annotations

import ipaddress
import os
import platform
import re
import shutil
import socket
import subprocess
from pathlib import Path

import psutil

GB = 1024 ** 3


def _run(cmd: list[str], timeout: float = 4.0) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return out.stdout.strip()
    except Exception:
        return ""


# --------------------------------------------------------------------------- CPU / GPU

def cpu_name() -> str:
    system = platform.system()
    if system == "Darwin":
        name = _run(["sysctl", "-n", "machdep.cpu.brand_string"])
        if name:
            return name
    elif system == "Linux":
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine()


# Rough memory bandwidth (GB/s) used only for speed *estimates* in the planner.
APPLE_BW = [
    ("ultra", 800), ("max", 400), ("pro", 200),
]


def apple_bandwidth(chip: str) -> int:
    c = chip.lower()
    for key, bw in APPLE_BW:
        if key in c:
            return bw
    if "m4" in c:
        return 120
    return 100  # base M1/M2/M3


def nvidia_bandwidth(name: str) -> int:
    n = name.lower()
    table = [("h200", 4800), ("h100", 3300), ("a100", 1900), ("5090", 1790), ("4090", 1000),
             ("3090", 936), ("5080", 960), ("4080", 716), ("a6000", 768), ("l40", 864),
             ("4070", 504), ("3080", 760), ("3070", 448), ("3060", 360), ("4060", 272)]
    for key, bw in table:
        if key in n:
            return bw
    return 400


def detect_gpus() -> list[dict]:
    gpus: list[dict] = []
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                    "--format=csv,noheader,nounits"])
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                try:
                    total = float(parts[1]) / 1024
                    used = float(parts[2]) / 1024
                except ValueError:
                    continue
                gpus.append({"vendor": "nvidia", "name": parts[0],
                             "vram_gb": round(total, 1), "vram_used_gb": round(used, 1),
                             "bandwidth_gbs": nvidia_bandwidth(parts[0])})
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        chip = cpu_name()
        gpus.append({"vendor": "apple", "name": f"{chip} GPU (Metal, unified memory)",
                     "vram_gb": None, "bandwidth_gbs": apple_bandwidth(chip)})
    return gpus


def usable_memory_gb(ram_total_gb: float, gpus: list[dict]) -> tuple[float, str]:
    """Memory this device can realistically give to model weights."""
    nv = [g for g in gpus if g["vendor"] == "nvidia"]
    if nv:
        return round(sum(g["vram_gb"] * 0.92 for g in nv), 1), "nvidia-vram"
    if any(g["vendor"] == "apple" for g in gpus):
        # macOS lets Metal use ~75% of unified memory by default (more on 64GB+ machines)
        frac = 0.75 if ram_total_gb < 64 else 0.8
        return round(ram_total_gb * frac - 2, 1), "apple-unified"
    return round(ram_total_gb * 0.7, 1), "cpu-ram"


# --------------------------------------------------------------------------- Network

SKIP_PREFIXES = ("lo", "docker", "veth", "br-", "virbr", "vmnet", "utun", "awdl", "llw",
                 "gif", "stf", "anpi", "ap", "bridge100", "tun", "tap", "zt", "vboxnet")


def _mac_port_map() -> dict[str, str]:
    """Map macOS device names (en0) to hardware port names (Wi-Fi, Thunderbolt Bridge...)."""
    out = _run(["networksetup", "-listallhardwareports"])
    mapping, port = {}, None
    for line in out.splitlines():
        if line.startswith("Hardware Port:"):
            port = line.split(":", 1)[1].strip()
        elif line.startswith("Device:") and port:
            mapping[line.split(":", 1)[1].strip()] = port
            port = None
    return mapping


def classify_interface(name: str, port_name: str = "") -> str:
    n, p = name.lower(), port_name.lower()
    if "wi-fi" in p or "wifi" in p or "airport" in p or "wireless" in p:
        return "wifi"
    if "thunderbolt" in p or n.startswith("thunderbolt") or n.startswith("tb"):
        return "thunderbolt"
    if "ethernet" in p or "lan" in p:
        return "ethernet"
    if "tailscale" in n or n.startswith("wg") or "vpn" in p:
        return "vpn"
    system = platform.system()
    if system == "Linux":
        if os.path.exists(f"/sys/class/net/{name}/wireless") or n.startswith("wl"):
            return "wifi"
        if n.startswith(("eth", "en", "em")):
            return "ethernet"
    if system == "Windows":
        if "wi-fi" in n or "wireless" in n or "wlan" in n:
            return "wifi"
        if "ethernet" in n:
            return "ethernet"
    if system == "Darwin" and n == "en0" and not p:
        return "wifi"
    return "other"


def detect_interfaces() -> list[dict]:
    addrs = psutil.net_if_addrs()
    stats = psutil.net_if_stats()
    port_map = _mac_port_map() if platform.system() == "Darwin" else {}
    result = []
    for name, entries in addrs.items():
        if name.lower().startswith(SKIP_PREFIXES) and not name.lower().startswith("bridge0"):
            continue
        st = stats.get(name)
        if st is not None and not st.isup:
            continue
        for e in entries:
            if e.family != socket.AF_INET:
                continue
            try:
                ip = ipaddress.ip_address(e.address)
            except ValueError:
                continue
            if ip.is_loopback:
                continue
            kind = classify_interface(name, port_map.get(name, ""))
            result.append({
                "name": name,
                "label": port_map.get(name) or name,
                "type": kind,
                "ip": e.address,
                "netmask": e.netmask,
                "broadcast": e.broadcast or _guess_broadcast(e.address, e.netmask),
                "link_local": ip.is_link_local,
                "speed_mbps": (st.speed if st and st.speed else None),
            })
    order = {"ethernet": 0, "thunderbolt": 1, "wifi": 2, "vpn": 3, "other": 4}
    result.sort(key=lambda i: (order.get(i["type"], 9), i["link_local"]))
    return result


def _guess_broadcast(ip: str, mask: str | None) -> str | None:
    if not mask:
        return None
    try:
        return str(ipaddress.ip_network(f"{ip}/{mask}", strict=False).broadcast_address)
    except ValueError:
        return None


# --------------------------------------------------------------------------- llama.cpp

def find_llama_binaries(project_root: Path) -> dict:
    """Locate rpc-server and llama-server (PATH, $LLAMA_CPP_DIR, or ./vendor build)."""
    candidates = []
    if os.environ.get("LLAMA_CPP_DIR"):
        candidates.append(Path(os.environ["LLAMA_CPP_DIR"]))
    candidates += [project_root / "vendor" / "llama.cpp" / "build" / "bin",
                   project_root / "vendor" / "llama.cpp" / "build" / "bin" / "Release"]
    found = {}
    for binary in ("rpc-server", "llama-server"):
        path = shutil.which(binary) or shutil.which(f"llama-{binary}")
        if not path:
            for c in candidates:
                for suffix in ("", ".exe"):
                    p = c / f"{binary}{suffix}"
                    if p.exists():
                        path = str(p)
                        break
                if path:
                    break
        found[binary] = path
    return found


# --------------------------------------------------------------------------- Summary

def snapshot(models_dir: Path, project_root: Path) -> dict:
    vm = psutil.virtual_memory()
    ram_total = vm.total / GB
    gpus = detect_gpus()
    usable, mem_kind = usable_memory_gb(ram_total, gpus)
    try:
        disk = psutil.disk_usage(str(models_dir))
        disk_free = round(disk.free / GB, 1)
    except Exception:
        disk_free = None
    system = platform.system()
    os_label = {"Darwin": "macOS", "Linux": "Linux", "Windows": "Windows"}.get(system, system)
    if system == "Darwin":
        os_label += " " + platform.mac_ver()[0]
    if gpus:
        bw = max(g["bandwidth_gbs"] for g in gpus)
    else:
        bw = 50
    return {
        "hostname": socket.gethostname(),
        "os": os_label,
        "arch": platform.machine(),
        "cpu": cpu_name(),
        "cores": psutil.cpu_count(logical=True),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "ram_total_gb": round(ram_total, 1),
        "ram_available_gb": round(vm.available / GB, 1),
        "gpus": gpus,
        "usable_memory_gb": max(usable, 0),
        "memory_kind": mem_kind,
        "mem_bandwidth_gbs": bw,
        "disk_free_gb": disk_free,
        "interfaces": detect_interfaces(),
        "llama_cpp": find_llama_binaries(project_root),
    }
