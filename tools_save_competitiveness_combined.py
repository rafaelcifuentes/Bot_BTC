# tools_save_competitiveness_combined.py
# Une los competitiveness_summary LONG y SHORT más recientes (o por stamp) en un JSON combinado.
# Uso:
#   python tools_save_competitiveness_combined.py
#   python tools_save_competitiveness_combined.py --long 20250810_013624 --short 20250810_014810

import os, json, glob, argparse
from datetime import datetime

REPORT_DIR = "./reports"

def pick_latest(pattern, exclude_substr=None):
    files = glob.glob(os.path.join(REPORT_DIR, pattern))
    if exclude_substr:
        files = [f for f in files if exclude_substr not in os.path.basename(f)]
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return files[0]

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def get_overall_rating(js):
    try:
        return float(js.get("overall", {}).get("overall_rating", 0.0))
    except Exception:
        return 0.0

def blend(a, b):
    # Mezcla simple: promedio de ratings generales
    return round((a + b) / 2.0, 1) if (a and b) else round(max(a, b), 1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--long",  help="stamp del LONG (p.ej. 20250810_013624)")
    ap.add_argument("--short", help="stamp del SHORT (p.ej. 20250810_014810)")
    args = ap.parse_args()

    # Localiza archivos
    if args.long:
        long_path  = os.path.join(REPORT_DIR, f"competitiveness_summary_{args.long}.json")
    else:
        # el más reciente que NO sea short ni COMBINED
        long_path  = pick_latest("competitiveness_summary_*.json", exclude_substr="short")
        # Evita combinar previos COMBINED
        if long_path and "COMBINED" in os.path.basename(long_path):
            # toma el siguiente
            cand = [f for f in glob.glob(os.path.join(REPORT_DIR, "competitiveness_summary_*.json"))
                    if "short" not in os.path.basename(f) and "COMBINED" not in os.path.basename(f)]
            if cand:
                cand.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                long_path = cand[0]

    if args.short:
        short_path = os.path.join(REPORT_DIR, f"competitiveness_summary_short_{args.short}.json")
    else:
        short_path = pick_latest("competitiveness_summary_short_*.json")

    if not long_path or not os.path.exists(long_path):
        print("❌ No se encontró resumen LONG. Revisa ./reports/")
        return
    if not short_path or not os.path.exists(short_path):
        print("❌ No se encontró resumen SHORT. Revisa ./reports/")
        return

    long_js  = load_json(long_path)
    short_js = load_json(short_path)

    long_rating  = get_overall_rating(long_js)
    short_rating = get_overall_rating(short_js)
    blend_rating = blend(long_rating, short_rating)

    out = {
        "stamp_utc": datetime.utcnow().isoformat(),
        "headline": "Competitiveness (COMBINED LONG+SHORT)",
        "inputs": {
            "long_file":  os.path.basename(long_path),
            "short_file": os.path.basename(short_path)
        },
        "long":  long_js,
        "short": short_js,
        "blend": {
            "overall_rating_blend": blend_rating,
            "notes": "Promedio de ratings globales LONG y SHORT; para comparación relativa entre runs."
        }
    }

    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(REPORT_DIR, f"competitiveness_summary_COMBINED_{stamp}.json")
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"📊 Guardado combinado → {out_path}")
    print(f"   LONG  overall ~ {long_rating}/10")
    print(f"   SHORT overall ~ {short_rating}/10")
    print(f"   BLEND overall ~ {blend_rating}/10")

if __name__ == "__main__":
    main()