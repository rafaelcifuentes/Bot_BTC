#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, datetime, pathlib

DAY3_VARS = ".day3_vars"  # creado por wf_day3.sh
OUT_DIR   = pathlib.Path("configs")
OUT_FILE  = OUT_DIR / "diamante_config.yaml"

def read_day3_vars(path):
    d = {}
    if not pathlib.Path(path).exists():
        return d
    for line in pathlib.Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"): continue
        if "=" in line:
            k,v = line.split("=",1)
            d[k.strip()] = v.strip().strip('"').strip("'")
    return d

def env_dict(keys):
    out = {}
    for k in keys:
        v = os.getenv(k)
        if v is not None:
            # coerce num if posible
            try:
                if "." in v: out[k] = float(v)
                else: out[k] = int(v)
            except Exception:
                out[k] = v
    return out

def dump_yaml(d):
    # minimal YAML dumper sin dependencias
    def _dump(obj, indent=0):
        sp = "  " * indent
        if isinstance(obj, dict):
            out = []
            for k,v in obj.items():
                if isinstance(v,(dict,list)):
                    out.append(f"{sp}{k}:")
                    out.append(_dump(v, indent+1))
                else:
                    val = "null" if v is None else v
                    out.append(f"{sp}{k}: {val}")
            return "\n".join(out)
        elif isinstance(obj, list):
            out = []
            for it in obj:
                if isinstance(it,(dict,list)):
                    out.append(f"{sp}-")
                    out.append(_dump(it, indent+1))
                else:
                    out.append(f"{sp}- {it}")
            return "\n".join(out)
        else:
            return f"{sp}{obj}"
    return _dump(d) + "\n"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    vars_d = read_day3_vars(DAY3_VARS)
    H = vars_d.get("H")
    TH_NEW = vars_d.get("TH_NEW")

    env_keys = [
        "TREND_FILTER","TREND_MODE","TREND_FREQ","TREND_SPAN",
        "TREND_BAND_PCT","REARM_MIN","HYSTERESIS_PP"
    ]
    env_cfg = env_dict(env_keys)

    cfg = {
        "name": "diamante_day3",
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "gates": {"pf_min": 1.6, "wr_min": 0.60, "trades_min": 30},
        "env": env_cfg,
        "candidate": {
            "asset": "BTC-USD",
            "horizon": int(H) if H else None,
            "threshold": float(TH_NEW) if TH_NEW else None,
            "note": "near-miss bumped +0.02 desde Q4 si aplica"
        }
    }
    OUT_FILE.write_text(dump_yaml(cfg), encoding="utf-8")
    print(f"[ok] snapshot -> {OUT_FILE}")

if __name__ == "__main__":
    main()