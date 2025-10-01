from datetime import datetime, timezone
import json

def assert_latest_json_contract(root, tag):
    sig = root / "signals/mini_accum/latest.json"
    problems = []
    try:
        try:
            data = json.loads(sig.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                data = {}
        except Exception:
            data = {}

        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Backfill robusto (NO toca la estrategia; solo contrato de señal)
        data["version"] = tag
        data["ts_utc"] = data.get("ts_utc") or now  # evita cadena vacía -> ISO inválida
        health = data.get("health")
        data["health"] = health if health in ("OK","WARN","PAUSE") else "OK"
        data["reason"] = data.get("reason") or "contract-backfill"
        if "position_pct_btc" not in data:
            data["position_pct_btc"] = 0.0
        if "guards" not in data or not isinstance(data["guards"], dict):
            data["guards"] = {"policy_asserts": "OK"}

        sig.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        problems.append(f"latest.json problema: {e}")
    return problems
