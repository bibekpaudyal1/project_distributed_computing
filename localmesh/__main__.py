"""Run a LocalMesh AI node:  python -m localmesh [--demo] [--token SECRET] [--name NAME]"""
import argparse
import os
import socket

import uvicorn

from .server import Config, create_app


def lan_ips():
    try:
        from .hardware import detect_interfaces
        return [f"{i['ip']} ({i['type']}{', direct cable/switch, no router' if i['link_local'] else ''})"
                for i in detect_interfaces()]
    except Exception:  # noqa: BLE001
        return []


def main():
    p = argparse.ArgumentParser(prog="localmesh", description="LocalMesh AI node")
    p.add_argument("--port", type=int, default=int(os.environ.get("LOCALMESH_PORT", 7860)))
    p.add_argument("--name", default=os.environ.get("LOCALMESH_NAME"))
    p.add_argument("--cluster", default=os.environ.get("LOCALMESH_CLUSTER", "default"),
                   help="Only devices with the same cluster name see each other")
    p.add_argument("--token", default=os.environ.get("LOCALMESH_TOKEN", ""),
                   help="Shared secret: devices need the same token to join")
    p.add_argument("--models-dir", default=os.environ.get("LOCALMESH_MODELS"))
    p.add_argument("--rpc-port", type=int, default=int(os.environ.get("LOCALMESH_RPC_PORT", 50052)))
    p.add_argument("--llama-port", type=int, default=int(os.environ.get("LOCALMESH_LLAMA_PORT", 8080)))
    p.add_argument("--demo", action="store_true", help="Add simulated devices for a demo")
    p.add_argument("--no-discovery", action="store_true")
    args = p.parse_args()

    cfg = Config(port=args.port, name=args.name, cluster=args.cluster, token=args.token,
                 models_dir=args.models_dir, demo_mode=args.demo,
                 discovery=not args.no_discovery, rpc_port=args.rpc_port,
                 llama_port=args.llama_port)
    print("\n  LocalMesh AI node", f"'{cfg.name}'", f"(id {cfg.node_id}, cluster '{cfg.cluster}')")
    print(f"  Dashboard:  http://localhost:{cfg.port}")
    for ip in lan_ips():
        addr, label = ip.split(" ", 1)
        print(f"  On the LAN: http://{addr}:{cfg.port}   {label}")
    if cfg.token:
        print("  Token protection: ON")
    print(f"  Models folder: {cfg.models_dir}\n")
    uvicorn.run(create_app(cfg), host="0.0.0.0", port=cfg.port, log_level="warning")


if __name__ == "__main__":
    main()
