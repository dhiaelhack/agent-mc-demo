"""Agent à outils qui pilote le calcul Monte Carlo à partir d'une demande en langage naturel.

Deux « planificateurs » sont fournis :
  - RuleBasedPlanner : simulateur de LLM à base de règles, pour tester la boucle sans modèle ;
  - OllamaPlanner    : appelle un LLM open source LOCAL via Ollama (http://localhost:11434).
Chaque étape (appel d'outil, sortie invalide, réponse finale) est tracée dans un fichier JSONL.
"""
import argparse, json, os, re, time, unicodedata, urllib.request
import mc_slab


class InvalidOutput(ValueError):
    """Sortie du planificateur qui n'est pas une action JSON valide."""
    def __init__(self, message, raw="", truncated=False):
        super().__init__(message)
        self.raw, self.truncated = raw, truncated


def _material(a):
    m = a.get("material")
    if m not in mc_slab.MU:
        raise ValueError(f"matériau inconnu : {m!r}. Matériaux disponibles : {', '.join(mc_slab.MU)}")
    return m


def _run_simulation(a):
    m, x, n = _material(a), float(a["thickness_cm"]), int(a.get("n_particles", 200_000))
    if not x > 0:
        raise ValueError("thickness_cm doit être > 0")
    if not 1_000 <= n <= 5_000_000:
        raise ValueError("n_particles doit être entre 1000 et 5000000")
    return mc_slab.simulate(m, x, n)


def _find_thickness(a):
    m, t = _material(a), float(a["target_transmission"])
    if not 0 < t < 1:
        raise ValueError("target_transmission doit être strictement entre 0 et 1 (ex. 0.01 pour 1 %)")
    return mc_slab.find_thickness(m, t)


TOOLS = {"run_simulation": _run_simulation, "find_thickness": _find_thickness}
SYSTEM = (
    "Tu pilotes un calcul Monte Carlo de transmission de photons de 662 keV. Matériaux : "
    + ", ".join(mc_slab.MU) + ".\nOutils :\n"
    "- run_simulation(material, thickness_cm, n_particles) : transmission à travers une épaisseur donnée.\n"
    "- find_thickness(material, target_transmission) : épaisseur pour atteindre une transmission cible, "
    "fraction entre 0 et 1 (0.01 = 1 %).\n"
    'Réponds UNIQUEMENT par un JSON : {"tool": "...", "args": {...}} pour appeler un outil, '
    'ou {"final": "..."} quand tu as la réponse ; "final" doit être une phrase en français avec les '
    "valeurs numériques et leurs unités. Ne devine jamais un résultat numérique."
)


SYSTEM_V2 = SYSTEM + (
    "\nExemples d'appels corrects (à adapter à la demande) :\n"
    "Demande : « Épaisseur d'eau pour une transmission de 20 % ? » -> "
    '{"tool": "find_thickness", "args": {"material": "eau", "target_transmission": 0.2}}\n'
    "Demande : « Quelle fraction passe à travers 3 cm de béton ? » -> "
    '{"tool": "run_simulation", "args": {"material": "béton", "thickness_cm": 3}}\n'
    "Les nombres sont des nombres JSON (0.2), jamais des chaînes ni 0 ; 20 % s'écrit 0.2. "
    "Appelle toujours un outil avant de donner un nombre. Si un outil renvoie une erreur, corrige ton appel "
    "et rappelle l'outil : ne réponds jamais avec le message d'erreur lui-même."
)
SYSTEM_V3 = SYSTEM_V2 + (
    "\nSi la demande est impossible ou hors limites (matériau absent de la liste, transmission hors de 0 à 1 "
    "exclus), n'appelle aucun outil de substitution : réponds avec final en expliquant ce qui ne va pas. "
    "Ne remplace jamais une valeur de la demande par une autre. Dès que tu as les résultats nécessaires, "
    "réponds avec final en une ou deux phrases, sans rien ajouter."
)
PROMPTS = {"v1": SYSTEM, "v2": SYSTEM_V2, "v3": SYSTEM_V3}


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
        res = [r for r in (json.loads(m["content"]) for m in messages if m["role"] == "tool")
               if "error" not in r]
        if not res:
            return {"final": "Je n'ai pas pu obtenir de résultat."}
        lines = [f"{r['material']} : {r['thickness_cm']} cm" for r in res]
        return {"final": f"Épaisseurs pour une transmission de {res[0]['target_transmission']:.1%} : "
                         + " ; ".join(lines) + "."}


class OllamaPlanner:
    def __init__(self, model="llama3.2:3b", url="http://localhost:11434/api/chat",
                 temperature=0.0, seed=0, timeout=120, num_predict=400):
        self.model, self.url, self.temperature, self.seed = model, url, temperature, seed
        self.timeout, self.num_predict = timeout, num_predict      # garde-fous : pas d'attente infinie

    def next(self, messages):
        msgs = [{"role": "assistant" if m["role"] == "tool_call" else "user" if m["role"] == "tool"
                 else m["role"], "content": ("Résultat de l'outil : " + m["content"])
                 if m["role"] == "tool" else m["content"]} for m in messages]
        body = {"model": self.model, "messages": msgs, "stream": False, "format": "json",
                "options": {"temperature": self.temperature, "seed": self.seed,
                            "num_predict": self.num_predict}}
        req = urllib.request.Request(self.url, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            resp = json.loads(r.read())
        raw = resp["message"]["content"]
        if resp.get("done_reason") == "length":
            raise InvalidOutput("réponse tronquée : max-tokens atteint", raw, truncated=True)
        try:
            action = json.loads(raw)
        except json.JSONDecodeError as e:
            raise InvalidOutput(f"JSON invalide : {raw[:200]!r}", raw) from e
        if not isinstance(action, dict) or not ("final" in action or "tool" in action):
            raise InvalidOutput(f"action inattendue : {raw[:200]!r}", raw)
        return action


def _canon(name):
    """Tolère majuscules et accents : « Beton » -> « béton ». Inchangé si le matériau est inconnu."""
    strip = lambda s: "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    return next((m for m in mc_slab.MU if strip(m) == strip(name.strip())), name)


def run(request, planner, max_steps=8, trace_path="traces/run.jsonl", system=None, guardrails=False):
    d = os.path.dirname(trace_path)
    if d:
        os.makedirs(d, exist_ok=True)
    messages = [{"role": "system", "content": system or SYSTEM}, {"role": "user", "content": request}]
    with open(trace_path, "a", encoding="utf8") as tr:
        def log(entry):
            tr.write(json.dumps(entry, ensure_ascii=False) + "\n")
        for step in range(max_steps):
            try:
                action = planner.next(messages)
            except InvalidOutput as e:                   # on le signale à l'agent et on réessaie
                log({"step": step, "invalid_output": str(e)})
                hint = (" Ta réponse était trop longue : réponds en une seule phrase courte."
                        if e.truncated else "")
                messages += [{"role": "tool_call", "content": e.raw[:500]},
                             {"role": "tool", "content": json.dumps({"error": (
                                 'Sortie invalide : réponds uniquement par un JSON '
                                 '{"tool": ..., "args": ...} ou {"final": "..."}.' + hint)})}]
                continue
            if "final" in action:
                final = action["final"]
                if not isinstance(final, str):
                    final = json.dumps(final, ensure_ascii=False)
                log({"step": step, "final": final})
                return final
            tool, args = action.get("tool"), action.get("args", {})
            if guardrails and isinstance(args, dict) and isinstance(args.get("material"), str):
                args = {**args, "material": _canon(args["material"])}
            try:
                if tool not in TOOLS:
                    raise ValueError(f"outil inconnu : {tool}. Outils : {', '.join(TOOLS)}. "
                                     'Pour donner ta réponse, réponds {"final": "ta phrase"} : ce n\'est pas un outil.')
                result = TOOLS[tool](args if isinstance(args, dict) else {})
            except KeyError as e:
                result = {"error": f"argument manquant : {e}"}
            except Exception as e:                       # l'erreur est renvoyée à l'agent
                result = {"error": str(e)}
            log({"step": step, "tool": tool, "args": args, "result": result, "t": round(time.time())})
            messages.append({"role": "tool_call", "content": json.dumps(action, ensure_ascii=False)})
            messages.append({"role": "tool", "content": json.dumps(result, ensure_ascii=False)})
            if guardrails and str(result.get("error", "")).startswith("matériau inconnu"):
                final = f"Je ne peux pas traiter cette demande : {result['error']}"   # on n'improvise pas
                log({"step": step + 1, "final": final, "guardrail": "matériau inconnu"})
                return final
        log({"step": max_steps, "stopped": "nombre maximal d'étapes atteint"})
    return "Arrêt : nombre maximal d'étapes atteint sans réponse finale."


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("request")
    ap.add_argument("--planner", choices=["rule", "ollama"], default="rule")
    ap.add_argument("--model", default="llama3.2:3b")
    ap.add_argument("--prompt", choices=list(PROMPTS), default="v1")
    ap.add_argument("--guardrails", action="store_true")
    a = ap.parse_args()
    p = RuleBasedPlanner() if a.planner == "rule" else OllamaPlanner(a.model)
    print(run(a.request, p, system=PROMPTS[a.prompt], guardrails=a.guardrails))
