"""Zero-config peer discovery over UDP broadcast.

Each node broadcasts a small JSON beacon every few seconds on every IPv4 interface.
Nodes in the same *cluster* (and with the same token fingerprint) record each other.
Office networks sometimes block broadcast (Wi-Fi "client isolation"); for those the
UI also lets you add a device manually by IP.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import time

from . import hardware

DISCOVERY_PORT = 7861
BEACON_EVERY = 2.0


def token_fingerprint(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()[:12] if token else ""


class _Protocol(asyncio.DatagramProtocol):
    def __init__(self, on_beacon):
        self.on_beacon = on_beacon

    def datagram_received(self, data, addr):
        try:
            msg = json.loads(data.decode())
        except Exception:
            return
        self.on_beacon(msg, addr[0])


class Discovery:
    def __init__(self, node_id: str, name: str, port: int, cluster: str, token: str,
                 on_peer):
        self.node_id = node_id
        self.name = name
        self.port = port
        self.cluster = cluster
        self.fp = token_fingerprint(token)
        self.on_peer = on_peer  # callback(node_id, name, ip, port, iface_hint)
        self.transport = None
        self.errors: list[str] = []

    async def start(self):
        loop = asyncio.get_running_loop()
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            try:
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:
                pass
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        try:
            sock.bind(("", DISCOVERY_PORT))
        except OSError as e:
            self.errors.append(f"discovery port {DISCOVERY_PORT} busy: {e}")
            return
        self.transport, _ = await loop.create_datagram_endpoint(
            lambda: _Protocol(self._handle), sock=sock)
        asyncio.create_task(self._beacon_loop())

    def _handle(self, msg: dict, ip: str):
        if msg.get("app") != "localmesh" or msg.get("cluster") != self.cluster:
            return
        if msg.get("fp", "") != self.fp or msg.get("id") == self.node_id:
            return
        self.on_peer(msg["id"], msg.get("name", ip), ip, int(msg.get("port", 7860)))

    async def _beacon_loop(self):
        while True:
            payload = json.dumps({"app": "localmesh", "id": self.node_id, "name": self.name,
                                  "port": self.port, "cluster": self.cluster, "fp": self.fp,
                                  "t": time.time()}).encode()
            targets = {"255.255.255.255"}
            for iface in hardware.detect_interfaces():
                if iface.get("broadcast"):
                    targets.add(iface["broadcast"])
            for t in targets:
                try:
                    self.transport.sendto(payload, (t, DISCOVERY_PORT))
                except Exception:
                    pass
            await asyncio.sleep(BEACON_EVERY)
