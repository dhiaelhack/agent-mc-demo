"""Agent à outils qui pilote le calcul Monte Carlo à partir d'une demande en langage naturel.

Deux « planificateurs » sont fournis :
  - RuleBasedPlanner : simulateur de LLM à base de règles, pour tester la boucle sans modèle ;
  - OllamaPlanner    : appelle un LLM open source LOCAL via Ollama (http://localhost:11434),
                       donc sans accès internet une fois le modèle téléchargé. NON TESTÉ ici.
Chaque appel d'outil est tracé dans traces/run.jsonl.
"""
import argparse, json, os, re, time, urllib.request
import mc_slab

TOOLS = {
    "run_simulation": lambda a: mc_slab.simulate(a["material"], float(a["thickness_cm"]),
                                                 int(a.get("n_particles", 200_000))),
    "find_thickness": lambda a: mc_slab.find_thickness(a["material"], float(a["target_transmission"])),
}
SYSTEM = (
    "Tu pilotes un calcul Monte Carlo de transmission de photons de 662 keV. Matériaux : "
    + ", ".join(mc_slab.MU) + ".\nOutils : run_simulation(material, thickness_cm, n_particles) ; "
    "find_thickness(material, target_transmission) avec une transmission entre 0 et 1.\n"
    'Réponds UNIQUEMENT par un JSON : {"tool": "...", "args": {...}} pour appeler un outil, '
    'ou {"final": "..."} quand tu as la réponse. Ne devine jamais un résultat numérique.'
)


class RuleBasedPlanner:
    """Simule un LLM : repère matériaux et pourcentage dans la demande, puis conclut."""
    def __init__(self):
        self.plan = None

    def next(self, messages):
        if self.plan is None:
            text = messages[1]["content"].lower().replace("beton", "béton")
            mats = [m for m in mc_slab.MU if re.search(rf"\b{m}\b", text)] or list(mc_slab.MU)
            pct = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
            target = float(pct.group(1).replace(",", ".")) / 100 if pct else 0.01
            self.plan = [{"tool": "find_thickness",
                          "args": {"material": m, "target_transmission": target}} for m in mats]
        done = sum(1 for m in messages if m["role"] == "tool")
        if done < len(self.plan):
            return self.plan[done]
        res = [json.loads(m["content"]) for m in messages if m["role"] == "tool"]
        lines = [f"{r['material']} : {r['thickness_cm']} cm" for r in res]
        return {"final": f"Épaisseurs pour une transmission de {res[0]['target_transmission']:.1%} : "
                         + " ; ".join(lines) + "."}


class OllamaPlanner:
    def __init__(self, model="llama3.1:8b", url="http://localhost:11434/api/chat"):
        self.model, self.url = model, url

    def next(self, messages):
        msgs = [{"role": "assistant" if m["role"] == "tool_call" else "user" if m["role"] == "tool"
                 else m["role"], "content": ("Résultat de l'outil : " + m["content"])
                 if m["role"] == "tool" else m["content"]} for m in messages]
        req = urllib.request.Request(self.url, json.dumps(
            {"model": self.model, "messages": msgs, "stream": False, "format": "json"}).encode(),
            {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(json.loads(r.read())["message"]["content"])


def run(request, planner, max_steps=8, trace_path="traces/run.jsonl"):
    os.makedirs(os.path.dirname(trace_path), exist_ok=True)
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": request}]
    with open(trace_path, "a", encoding="utf8") as tr:
        for step in range(max_steps):
            action = planner.next(messages)
            if "final" in action:
                tr.write(json.dumps({"step": step, "final": action["final"]}, ensure_ascii=False) + "\n")
                return action["final"]
            tool, args = action.get("tool"), action.get("args", {})
            try:
                if tool not in TOOLS:
                    raise ValueError(f"outil inconnu : {tool}")
                if args.get("material") not in mc_slab.MU:
                    raise ValueError(f"matériau inconnu : {args.get('material')}")
                result = TOOLS[tool](args)
            except Exception as e:                       # l'erreur est renvoyée à l'agent
                result = {"error": str(e)}
            tr.write(json.dumps({"step": step, "tool": tool, "args": args, "result": result,
                                 "t": round(time.time())}, ensure_ascii=False) + "\n")
            messages.append({"role": "tool_call", "content": json.dumps(action, ensure_ascii=False)})
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
    return "Arrêt : nombre maximal d'étapes atteint sans réponse finale."


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("--planner", choices=["rule", "ollama"], default="rule")
    ap.add_argument("--model", default="llama3.1:8b")
    a = ap.parse_args()
    p = RuleBasedPlanner() if a.planner == "rule" else OllamaPlanner(a.model)
    print(run(a.request, p))
