"""Banc d'évaluation : lance chaque cas de cases.json plusieurs fois et mesure la réussite.

Une exécution est réussie si (1) les bons outils ont été appelés avec les bons arguments et
(2) la réponse finale contient les bonnes valeurs numériques. Pour les cas sans appel attendu
(matériau inconnu, transmission impossible), elle est réussie si aucun outil n'a abouti et si la
réponse ne donne aucune épaisseur en cm (critère heuristique).

Usage : python eval.py --planner rule
        python eval.py --planner ollama --model llama3.2:3b --runs 3 --temperature 0.7
"""
import argparse, datetime, json, math, os, re, sys, tempfile, time
import mc_slab, agent


def expected_values(case):
    out = []
    for c in case["expected_calls"]:
        mu = mc_slab.MU[c["material"]]
        if c["tool"] == "find_thickness":
            out.append(("cm", -math.log(c["target_transmission"]) / mu))
        else:
            out.append(("T", math.exp(-mu * c["thickness_cm"])))
    return out


def numbers(text):
    return [float(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", text or "")]


def close(v, exp, unit):
    if unit == "cm":
        return abs(v - exp) <= max(0.05, 0.01 * exp)
    return any(abs(c - exp) <= 0.02 * exp + 0.001 for c in (v, v / 100))   # fraction ou pourcentage


def call_matches(entry, exp):
    if entry.get("tool") != exp["tool"] or "error" in entry.get("result", {}):
        return False
    a = entry.get("args", {})
    try:
        if a.get("material") != exp["material"]:
            return False
        if exp["tool"] == "find_thickness":
            return math.isclose(float(a["target_transmission"]), exp["target_transmission"], rel_tol=1e-6)
        return math.isclose(float(a["thickness_cm"]), exp["thickness_cm"], rel_tol=1e-6)
    except (KeyError, TypeError, ValueError):
        return False


def score(case, trace, final):
    calls = [e for e in trace if "tool" in e]
    ok_calls = [e for e in calls if "error" not in e.get("result", {})]
    invalid = sum(1 for e in trace if "invalid_output" in e)
    stopped = any("stopped" in x for x in trace)               # arrêt faute de réponse finale
    base = {"invalid_outputs": invalid, "tool_calls": len(calls),
            "tool_errors": len(calls) - len(ok_calls), "stopped": stopped}
    if not case["expected_calls"]:                                 # refus attendu
        tool_ok = not ok_calls
        final_ok = isinstance(final, str) and not stopped and not re.search(r"\d\s*cm", final)
        return {**base, "calls_ok": tool_ok, "final_ok": final_ok, "success": tool_ok and final_ok}
    calls_ok = all(any(call_matches(e, exp) for e in ok_calls) for exp in case["expected_calls"])
    nums = numbers(final) if isinstance(final, str) else []
    final_ok = not stopped and all(any(close(v, ev, u) for v in nums) for u, ev in expected_values(case))
    return {**base, "calls_ok": calls_ok, "final_ok": final_ok, "success": calls_ok and final_ok}


def describe_failure(row, request):
    lines = [f"- {row['case']} (exécution {row['run']}) : {request}"]
    for e in row.get("trace", []):
        if "invalid_output" in e:
            lines.append(f"    sortie invalide : {e['invalid_output'][:150]}")
        elif "tool" in e:
            res = e["result"]
            res_txt = f"ERREUR {res['error']}" if "error" in res else f"ok {res.get('thickness_cm', res.get('transmission'))}"
            lines.append(f"    {e['tool']}({json.dumps(e['args'], ensure_ascii=False)}) -> {res_txt}")
    lines.append(f"    réponse finale : {row['final']!r}")
    if row["crash"]:
        lines.append(f"    plantage : {row['crash']}")
    return "\n".join(lines)


def run_case(case, make_planner, run_index, system=None, guardrails=False):
    path = os.path.join(tempfile.mkdtemp(), "trace.jsonl")
    t0 = time.time()
    try:
        final = agent.run(case["request"], make_planner(run_index), trace_path=path, system=system,
                          guardrails=guardrails)
        crash = None
    except Exception as e:                                         # panne du planificateur (réseau, etc.)
        final, crash = None, f"{type(e).__name__}: {e}"
    dt = time.time() - t0
    trace = [json.loads(l) for l in open(path, encoding="utf8")] if os.path.exists(path) else []
    s = score(case, trace, final)
    if crash:
        s["success"] = False
    return {"case": case["id"], "run": run_index, "seconds": round(dt, 2), "final": final,
            "crash": crash, **s, "trace": trace}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", choices=["rule", "ollama"], default="rule")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--cases", default="cases.json")
    ap.add_argument("--prompt", choices=list(agent.PROMPTS), default="v1", help="version du prompt système")
    ap.add_argument("--guardrails", action="store_true",
                    help="active les garde-fous du code (matériau toléré aux accents ; arrêt si matériau inconnu)")
    ap.add_argument("--timeout", type=int, default=120, help="secondes max par requête au modèle")
    ap.add_argument("--max-tokens", type=int, default=400, help="longueur maximale d'une réponse du modèle")
    ap.add_argument("--only", default="", help="identifiants de cas séparés par des virgules")
    ap.add_argument("--show-failures", action="store_true", help="détaille chaque exécution échouée")
    a = ap.parse_args()
    cases = json.load(open(a.cases, encoding="utf8"))
    if a.only:
        wanted = set(a.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]
    if a.planner == "rule":
        make = lambda i: agent.RuleBasedPlanner()
    else:
        make = lambda i: agent.OllamaPlanner(a.model, temperature=a.temperature, seed=i,
                                             timeout=a.timeout, num_predict=a.max_tokens)
    rows, total = [], len(cases) * a.runs
    for c in cases:
        for i in range(a.runs):
            r = run_case(c, make, i, agent.PROMPTS[a.prompt], a.guardrails)
            rows.append(r)
            print(f"[{len(rows)}/{total}] {c['id']} #{i} -> {'réussite' if r['success'] else 'échec'} "
                  f"({r['seconds']} s)" + (f" PLANTAGE : {r['crash']}" if r["crash"] else ""),
                  file=sys.stderr, flush=True)

    print(f"\nPlanificateur : {a.planner}" + (f" ({a.model}, T={a.temperature})" if a.planner == "ollama" else "")
          + f" | prompt {a.prompt}{' + garde-fous' if a.guardrails else ''} | {len(cases)} cas x {a.runs} exécution(s)\n")
    print("| cas | réussites | sorties invalides | erreurs d'outil | arrêts à 8 étapes | durée moy. (s) |")
    print("|---|---|---|---|---|---|")
    for c in cases:
        r = [x for x in rows if x["case"] == c["id"]]
        print(f"| {c['id']} | {sum(x['success'] for x in r)}/{len(r)} | {sum(x['invalid_outputs'] for x in r)} "
              f"| {sum(x['tool_errors'] for x in r)} | {sum(bool(x.get('stopped')) for x in r)} "
              f"| {sum(x['seconds'] for x in r) / len(r):.1f} |")
    n = len(rows)
    print(f"\nTotal : {sum(x['success'] for x in rows)}/{n} réussites "
          f"({100 * sum(x['success'] for x in rows) / n:.0f} %) ; "
          f"{sum(x['invalid_outputs'] for x in rows)} sorties invalides ; "
          f"{sum(x['tool_errors'] for x in rows)} erreurs d'outil ; "
          f"{sum(1 for x in rows if x['crash'])} plantages du planificateur.")
    if a.show_failures:
        req = {c["id"]: c["request"] for c in cases}
        print("\nExécutions échouées :")
        for x in rows:
            if not x["success"]:
                print(describe_failure(x, req[x["case"]]))
    os.makedirs("results", exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = a.planner if a.planner == "rule" else f"{a.planner}_{a.model.replace(':', '-')}_{a.prompt}{'_garde' if a.guardrails else ''}"
    out = f"results/eval_{name}_{stamp}.json"
    json.dump({"config": vars(a), "rows": rows}, open(out, "w", encoding="utf8"),
              ensure_ascii=False, indent=1)
    print(f"Détails : {out}")


if __name__ == "__main__":
    main()
