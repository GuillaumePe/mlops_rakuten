"""
TCP tunnel : forward localhost → remote via SOCKS5.

Utilisé sur les pods RunPod pour atteindre le MongoDB local via Tailscale.
Lancé en background par l'entrypoint avant le runner.

Usage (entrypoint) :
    python /workspace/scripts/mongo_tunnel.py \
        --remote-host 100.126.74.73 --remote-port 27017 \
        --local-port 27018 \
        --proxy-host localhost --proxy-port 1055 &
    export MONGO_URI="mongodb://localhost:27018"
"""
import argparse
import select
import socket
import threading

import socks


def handle_client(client_sock, remote_host, remote_port, proxy_host, proxy_port):
    remote = None
    try:
        remote = socks.socksocket(socket.AF_INET, socket.SOCK_STREAM)
        remote.set_proxy(socks.SOCKS5, proxy_host, proxy_port)
        remote.connect((remote_host, remote_port))

        while True:
            readable, _, _ = select.select([client_sock, remote], [], [], 120)
            if not readable:
                break
            for s in readable:
                data = s.recv(8192)
                if not data:
                    return
                if s is client_sock:
                    remote.sendall(data)
                else:
                    client_sock.sendall(data)
    except Exception as e:
        print(f"[mongo-tunnel] Connection error: {e}")
    finally:
        client_sock.close()
        if remote:
            try:
                remote.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="TCP tunnel via SOCKS5")
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-port", type=int, default=27017)
    parser.add_argument("--local-port", type=int, default=27018)
    parser.add_argument("--proxy-host", default="localhost")
    parser.add_argument("--proxy-port", type=int, default=1055)
    args = parser.parse_args()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", args.local_port))
    server.listen(10)
    print(
        f"[mongo-tunnel] Listening on 127.0.0.1:{args.local_port} "
        f"→ {args.remote_host}:{args.remote_port} via SOCKS5 "
        f"{args.proxy_host}:{args.proxy_port}",
        flush=True,
    )

    while True:
        client, addr = server.accept()
        t = threading.Thread(
            target=handle_client,
            args=(client, args.remote_host, args.remote_port, args.proxy_host, args.proxy_port),
            daemon=True,
        )
        t.start()


if __name__ == "__main__":
    main()
