"""TCP bridge: PyBullet backend -> Unity frontend.

Protocol: newline-delimited JSON over TCP (one JSON object per line).
Backend -> Unity: {"type": "hello"|"state"|"path"|"prob"|"detection", ...}
Unity -> backend: {"type": "cmd", "cmd": "goto"|"set_lkp"|"pause"|"resume", ...}

All positions are in PyBullet world coordinates (meters, right-handed, Z-up).
Unity converts on its side (see CoordConv.cs); the backend never knows about Unity.
"""
import json
import queue
import socket
import threading


class Bridge:
    # Latest message of these types is replayed to clients that connect late
    # (Unity usually starts after the backend).
    STICKY = ("path", "prob", "detection")

    def __init__(self, host="127.0.0.1", port=5555):
        self.host, self.port = host, port
        self._sticky = {}
        self._clients = []
        self._lock = threading.Lock()
        self._cmds = queue.Queue()
        self._hello = None
        self._srv = None
        self._running = False

    # ---- lifecycle -------------------------------------------------------
    def start(self):
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((self.host, self.port))
        self._srv.listen(4)
        self._running = True
        threading.Thread(target=self._accept_loop, daemon=True).start()

    def stop(self):
        self._running = False
        with self._lock:
            for c in self._clients:
                try:
                    c.close()
                except OSError:
                    pass
            self._clients.clear()
        if self._srv:
            self._srv.close()

    def set_hello(self, msg):
        """Message replayed to every client on connect (terrain metadata etc.)."""
        self._hello = dict(msg, type="hello")

    # ---- backend API -----------------------------------------------------
    def send(self, msg):
        data = (json.dumps(msg, separators=(",", ":")) + "\n").encode()
        with self._lock:
            if msg.get("type") in self.STICKY:
                self._sticky[msg["type"]] = data
            for c in list(self._clients):
                self._send_raw(c, data)

    def poll(self):
        """Return all commands received since the last call (non-blocking)."""
        out = []
        while True:
            try:
                out.append(self._cmds.get_nowait())
            except queue.Empty:
                return out

    @property
    def n_clients(self):
        with self._lock:
            return len(self._clients)

    # ---- internals -------------------------------------------------------
    def _send_raw(self, conn, data):
        try:
            conn.sendall(data)
        except OSError:
            if conn in self._clients:
                self._clients.remove(conn)
            try:
                conn.close()
            except OSError:
                pass

    def _accept_loop(self):
        while self._running:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self._lock:
                self._clients.append(conn)
                if self._hello is not None:
                    self._send_raw(conn, (json.dumps(self._hello) + "\n").encode())
                for data in self._sticky.values():
                    self._send_raw(conn, data)
            threading.Thread(target=self._reader, args=(conn,), daemon=True).start()

    def _reader(self, conn):
        buf = b""
        try:
            while self._running:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line)
                    except ValueError:
                        continue
                    if msg.get("type") == "cmd":
                        self._cmds.put(msg)
        except OSError:
            pass
        finally:
            with self._lock:
                if conn in self._clients:
                    self._clients.remove(conn)
            try:
                conn.close()
            except OSError:
                pass
