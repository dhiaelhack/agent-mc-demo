"""Relit un fichier de résultats, sans relancer le modèle.

Usage : python report.py results/eval_....json [cas1,cas2]     détaille les échecs
        python report.py results/eval_....json --rescore       re-note avec les règles actuelles
"""
import argparse, json
import eval as ev

ap = argparse.ArgumentParser()
ap.add_argument("file")
ap.add_argument("only", nargs="?", default="")
ap.add_argument("--rescore", action="store_true")
a = ap.parse_args()
data = json.load(open(a.file, encoding="utf8"))
cases = {c["id"]: c for c in json.load(open(data["config"].get("cases", "cases.json"), encoding="utf8"))}
print("Configuration :", data["config"])
if a.rescore:
    old = new = 0
    for r in data["rows"]:
        s = ev.score(cases[r["case"]], r.get("trace", []), r["final"])
        ok = bool(s["success"] and not r.get("crash"))
        old += bool(r["success"]); new += ok
        if ok != bool(r["success"]):
            print(f"  {r['case']} (exécution {r['run']}) : {'réussite' if r['success'] else 'échec'} -> "
                  f"{'réussite' if ok else 'échec'}")
    print(f"Total : {old}/{len(data['rows'])} -> {new}/{len(data['rows'])}")
else:
    only = set(a.only.split(",")) if a.only else None
    for r in data["rows"]:
        if not r["success"] and (only is None or r["case"] in only):
            print(ev.describe_failure(r, cases[r["case"]]["request"]))
