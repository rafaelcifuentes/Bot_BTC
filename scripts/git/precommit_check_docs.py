#!/usr/bin/env python3
import sys, os, json

ROOT = os.getcwd()
manifest_path = os.path.join(ROOT, "docs", "mini_accum", "manifest.json")

def fail(msg):
    print(f"[pre-commit] ERROR: {msg}")
    sys.exit(1)

def warn(msg):
    print(f"[pre-commit] WARNING: {msg}")

if not os.path.isfile(manifest_path):
    fail(f"manifest.json no encontrado en {manifest_path}")

with open(manifest_path, "r", encoding="utf-8") as f:
    data = json.load(f)

mods = data.get("modules", [])
if not mods:
    fail("manifest.json no contiene módulos.")

# 1) Check runbook & overlay existence
missing = False
for m in mods:
    rb = os.path.join(ROOT, m["runbook"])
    ov = os.path.join(ROOT, m["overlay"])
    if not os.path.isfile(rb):
        warn(f"Runbook faltante: {m['name']} → {rb}")
        missing = True
    if not os.path.isfile(ov):
        warn(f"Overlay faltante: {m['name']} → {ov}")
        missing = True

if missing:
    fail("Faltan runbooks/overlays requeridos según manifest.json.")

# 2) Check there are no runbooks without manifest entry
docs_dir = os.path.join(ROOT, "docs", "mini_accum")
known = { os.path.basename(m["runbook"]).lower() for m in mods }
for fname in os.listdir(docs_dir):
    if not fname.lower().endswith(".md"):
        continue
    if fname.lower() in {"brochure.md", "roadmap.md", "readme.md", "progreso.md"}:
        continue
    if fname.lower() not in known:
        warn(f"Runbook encontrado sin entrada en manifest.json: {fname}")

print("[pre-commit] OK: manifest.json, runbooks y overlays alineados.")
sys.exit(0)
