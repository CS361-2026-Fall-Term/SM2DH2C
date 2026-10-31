"""Protocol smoke test: stands in for Unity. Run while sar_demo.py is running."""
import json
import socket
import sys
import time

port = int(sys.argv[1]) if len(sys.argv) > 1 else 5555
s = socket.create_connection(("127.0.0.1", port), timeout=5)
f = s.makefile("rwb")

counts, first = {}, {}
sent_goto = False
t0 = time.time()
while time.time() - t0 < 6:
    line = f.readline()
    if not line:
        break
    m = json.loads(line)
    counts[m["type"]] = counts.get(m["type"], 0) + 1
    first.setdefault(m["type"], m)
    if m["type"] == "state" and counts["state"] == 20 and not sent_goto:
        f.write((json.dumps({"type": "cmd", "cmd": "goto", "pos": [300.0, 100.0]}) + "\n").encode())
        f.flush()
        sent_goto = True
        goto_at = counts["state"]
    if sent_goto and m["type"] == "path" and counts["path"] >= 2:
        first["path_after_goto"] = m

print("message counts:", counts)
print("hello keys:", sorted(first["hello"]))
print("terrain n/size:", first["hello"]["terrain"]["n"], first["hello"]["terrain"]["size_m"])
print("first state:", first["state"])
print("prob cells:", len(first["prob"]["data"]), "sum=%.3f" % sum(first["prob"]["data"]))
pa = first.get("path_after_goto")
print("path after goto (pts flat len):", len(pa["pts"]) if pa else None,
      "->", pa["pts"][:3] if pa else None)
assert counts.get("hello") == 1 and counts["state"] > 50
assert len(first["prob"]["data"]) == first["prob"]["n"] ** 2
assert pa and len(pa["pts"]) == 3, "goto command did not replace the path"
print("OK")
