#!/usr/bin/env python3
import json, pathlib, datetime as dt, os
ROOT = pathlib.Path(os.environ.get("ROOT", str(pathlib.Path.home()/ "PycharmProjects/Bot_BTC")))
day  = dt.datetime.utcnow().date().isoformat()
out  = ROOT / "evidence" / f"dayN_{day}"
out.mkdir(parents=True, exist_ok=True)
payload = {
  "lab": "LAB4_BullGuard",
  "rule": "Daily < SMA200_d AND 2w < SMA200_w",
  "double_confirm": None,   # sombra/stub (no toca lógica)
  "ts_utc": dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
  "note": "shadow-only stub; no trading logic touched"
}
(out / "LAB4_BG2w.shadow.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
(out / "LAB4_BG2w.shadow.md").write_text("# LAB4 Bull-guard (shadow)\n\nInforme sombra generado (stub).\n", encoding="utf-8")
print(f"[OK] LAB4 shadow -> {out}")
