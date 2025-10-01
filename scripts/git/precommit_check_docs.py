#!/usr/bin/env python3
import sys, os, json, re

ROOT = os.getcwd()
manifest_path = os.path.join(ROOT, "docs", "mini_accum", "manifest.json")
roadmap_path  = os.path.join(ROOT, "docs", "mini_accum", "roadmap.md")

def fail(msg):
    print(f"[pre-commit] ERROR: {msg}")
    sys.exit(1)
def warn(msg):
    print(f"[pre-commit] WARNING: {msg}")

# Manifest
if not os.path.isfile(manifest_path):
    fail(f"manifest.json no encontrado en {manifest_path}")
with open(manifest_path, "r", encoding="utf-8") as f:
    data = json.load(f)
mods = data.get("modules", [])
if not mods:
    fail("manifest.json no contiene módulos.")
names = {m["name"] for m in mods}
placeholders = {m["name"] for m in mods if m.get("placeholder")}

# Runbooks/overlays (no-placeholder)
missing = False
for m in mods:
    if m["name"] in placeholders:
        continue
    rb = os.path.join(ROOT, m["runbook"])
    ov = os.path.join(ROOT, m["overlay"])
    if not os.path.isfile(rb):
        warn(f"Runbook faltante: {m['name']} → {rb}"); missing = True
    if not os.path.isfile(ov):
        warn(f"Overlay faltante: {m['name']} → {ov}"); missing = True
if missing:
    fail("Faltan runbooks/overlays requeridos según manifest.json (no-placeholder).")

# Roadmap vs manifest
if not os.path.isfile(roadmap_path):
    fail(f"roadmap.md no encontrado en {roadmap_path}")
with open(roadmap_path, "r", encoding="utf-8") as f:
    text = f.read().lower()
tokens = set(re.findall(r"\b[a-z][a-z0-9_]{2,}\b", text))
candidates = {t for t in tokens if '_' in t}
unknown = sorted(candidates - names)
if unknown:
    fail("roadmap.md hace referencia a módulos NO listados en manifest.json: " + ", ".join(unknown))

# Whitelist de .md no-runbook para no spamear warnings
docs_dir = os.path.join(ROOT, "docs", "mini_accum")
known = { os.path.basename(m["runbook"]).lower() for m in mods }
allowed = {
    "brochure.md", "roadmap.md", "readme.md", "progreso.md", "plan.md",
    "manual_operativo_semanal.md", "deployment_stages.md",
    "mini_accum_plan_bkp.md", "progreso_bkp1.md"
}
if os.path.isdir(docs_dir):
    for fname in os.listdir(docs_dir):
        if not fname.lower().endswith(".md"): 
            continue
        if fname.lower() in allowed: 
            continue
        if fname.lower() not in known:
            warn(f"Runbook encontrado sin entrada en manifest.json: {fname}")

print("[pre-commit] OK: manifest/runbooks/overlays y roadmap alineados.")
sys.exit(0)
