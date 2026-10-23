"""Tests simples : python tests.py"""
import os, tempfile, json
import mc_slab, agent


def test_mc_matches_analytic():
    for m in mc_slab.MU:
        r = mc_slab.simulate(m, 1.5)
        assert abs(r["transmission"] - r["analytic"]) < 4 * r["stat_error"] + 1e-4, r


def test_find_thickness_close_to_analytic():
    r = mc_slab.find_thickness("plomb", 0.01)
    assert abs(r["thickness_cm"] - r["analytic_thickness_cm"]) < 0.05, r


def test_agent_single_material_not_confused_by_faisceau():
    tr = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    out = agent.run("Quelle épaisseur de plomb pour réduire le faisceau à 1 % ?",
                    agent.RuleBasedPlanner(), trace_path=tr)
    calls = [json.loads(l) for l in open(tr, encoding="utf8") if '"tool"' in l]
    assert [c["args"]["material"] for c in calls] == ["plomb"], calls
    assert "plomb" in out


def test_agent_reports_tool_errors():
    class Bad:
        n = 0
        def next(self, msgs):
            self.n += 1
            return {"tool": "run_simulation", "args": {"material": "or", "thickness_cm": 1}} \
                if self.n == 1 else {"final": "erreur gérée"}
    tr = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    assert agent.run("test", Bad(), trace_path=tr) == "erreur gérée"
    assert "matériau inconnu" in open(tr, encoding="utf8").read()




# ---- tests du banc d'évaluation et des nouvelles protections ----
import eval as ev


class Oracle:
    """Planificateur parfait : fait exactement les appels attendus, puis cite les résultats."""
    def __init__(self, case):
        self.calls, self.k = case["expected_calls"], 0

    def next(self, msgs):
        if not self.calls:
            return {"final": "Cette demande n'est pas réalisable avec les matériaux disponibles."}
        if self.k < len(self.calls):
            c = self.calls[self.k]; self.k += 1
            args = {"material": c["material"]}
            args.update({"target_transmission": c["target_transmission"]} if c["tool"] == "find_thickness"
                        else {"thickness_cm": c["thickness_cm"]})
            return {"tool": c["tool"], "args": args}
        res = [json.loads(m["content"]) for m in msgs if m["role"] == "tool"]
        return {"final": " ; ".join(
            f"{r['material']} : {r['thickness_cm']} cm" if "thickness_cm" in r and "target_transmission" in r
            else f"transmission {r['transmission']:.4f}" for r in res)}


def test_eval_oracle_scores_everything_ok():
    for case in json.load(open("cases.json", encoding="utf8")):
        row = ev.run_case(case, lambda i, c=case: Oracle(c), 0)
        assert row["success"], row


def test_eval_detects_guessing_without_tool():
    case = json.load(open("cases.json", encoding="utf8"))[0]
    class Guess:
        def next(self, msgs): return {"final": "Environ 3,65 cm."}
    row = ev.run_case(case, lambda i: Guess(), 0)
    assert row["final_ok"] and not row["calls_ok"] and not row["success"], row


def test_target_in_percent_is_rejected():
    class P:
        n = 0
        def next(self, msgs):
            self.n += 1
            return {"tool": "find_thickness", "args": {"material": "plomb", "target_transmission": 1}} \
                if self.n == 1 else {"final": "ok"}
    tr = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    agent.run("x", P(), trace_path=tr)
    assert "strictement entre 0 et 1" in open(tr, encoding="utf8").read()


def test_invalid_json_is_retried_and_final_is_string():
    class P:
        n = 0
        def next(self, msgs):
            self.n += 1
            if self.n == 1:
                raise agent.InvalidOutput("JSON invalide", "pas du json")
            return {"final": {"thickness_cm": 3.6}}
    tr = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    out = agent.run("x", P(), trace_path=tr)
    assert isinstance(out, str) and "3.6" in out
    assert '"invalid_output"' in open(tr, encoding="utf8").read()


def test_eval_keeps_trace_and_describes_failures():
    case = json.load(open("cases.json", encoding="utf8"))[0]
    class Guess:
        def next(self, msgs): return {"final": "Environ 3,65 cm."}
    row = ev.run_case(case, lambda i: Guess(), 0)
    assert isinstance(row["trace"], list) and row["trace"][-1]["final"] == "Environ 3,65 cm."
    assert "réponse finale" in ev.describe_failure(row, case["request"])


def test_prompt_versions_and_no_leakage_of_test_cases():
    assert set(agent.PROMPTS) == {"v1", "v2", "v3"} and len(set(agent.PROMPTS.values())) == 3
    # les exemples des prompts ne doivent pas reprendre une demande de cases.json
    for case in json.load(open("cases.json", encoding="utf8")):
        for p in agent.PROMPTS.values():
            assert case["request"] not in p
    class P:
        seen = None
        def next(self, msgs):
            P.seen = msgs[0]["content"]; return {"final": "ok"}
    tr = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    agent.run("x", P(), trace_path=tr, system=agent.PROMPTS["v2"])
    assert P.seen == agent.PROMPTS["v2"]


def test_unknown_tool_error_explains_how_to_answer():
    class P:
        n = 0
        def next(self, msgs):
            self.n += 1
            return {"tool": "final", "args": {"text": "x"}} if self.n == 1 else {"final": "ok"}
    tr = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    agent.run("x", P(), trace_path=tr)
    assert '{\\"final\\"' in open(tr, encoding="utf8").read() or '"final"' in open(tr, encoding="utf8").read()
    entry = [json.loads(l) for l in open(tr, encoding="utf8")][0]
    assert "outil inconnu : final" in entry["result"]["error"] and '{"final"' in entry["result"]["error"]


def test_guardrail_stops_on_unknown_material_and_accepts_accents():
    class P:
        n = 0
        def next(self, msgs):
            self.n += 1
            if self.n == 1:
                return {"tool": "find_thickness", "args": {"material": "or", "target_transmission": 0.01}}
            return {"tool": "find_thickness", "args": {"material": "plomb", "target_transmission": 0.01}}
    tr = os.path.join(tempfile.mkdtemp(), "t.jsonl")
    out = agent.run("x", P(), trace_path=tr, guardrails=True)
    assert "matériau inconnu" in out and "cm" not in out
    lines = [json.loads(l) for l in open(tr, encoding="utf8")]
    assert len([l for l in lines if "tool" in l]) == 1          # pas d'appel de substitution
    assert agent._canon("Beton") == "béton" and agent._canon("PLOMB") == "plomb" and agent._canon("or") == "or"


def test_truncated_generation_is_reported_as_such():
    import io
    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    orig = agent.urllib.request.urlopen
    agent.urllib.request.urlopen = lambda req, timeout=None: R(json.dumps(
        {"message": {"content": '{"final": "tron'}, "done_reason": "length"}).encode())
    try:
        agent.OllamaPlanner("m").next([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
        raise AssertionError("aurait dû lever InvalidOutput")
    except agent.InvalidOutput as e:
        assert e.truncated
    finally:
        agent.urllib.request.urlopen = orig


def test_holdout_is_well_formed_and_disjoint():
    main = json.load(open("cases.json", encoding="utf8"))
    hold = json.load(open("cases_holdout.json", encoding="utf8"))
    assert {c["request"] for c in main}.isdisjoint({c["request"] for c in hold})
    for c in hold:
        assert all(x["material"] in mc_slab.MU for x in c["expected_calls"])
        for p in agent.PROMPTS.values():
            assert c["request"] not in p
        row = ev.run_case(c, lambda i, c=c: Oracle(c), 0)
        assert row["success"], row


def test_refusal_is_not_credited_when_agent_hits_step_limit():
    case = {"id": "r", "request": "x", "expected_calls": []}
    err = {"step": 0, "tool": "find_thickness", "args": {}, "result": {"error": "x"}}
    stopped = [err, {"step": 8, "stopped": "nombre maximal d'étapes atteint"}]
    assert not ev.score(case, stopped, "Arrêt : nombre maximal d'étapes atteint sans réponse finale.")["success"]
    concluded = [err, {"step": 1, "final": "Demande impossible."}]
    assert ev.score(case, concluded, "Demande impossible.")["success"]


def test_ollama_planner_sends_limits_and_parses():
    import io
    captured = {}
    class R(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False
    def fake(req, timeout=None):
        captured["body"], captured["timeout"] = json.loads(req.data), timeout
        return R(json.dumps({"message": {"content": '{"final": "ok"}'}}).encode())
    orig = agent.urllib.request.urlopen
    agent.urllib.request.urlopen = fake
    try:
        p = agent.OllamaPlanner("m", temperature=0, seed=3, timeout=45, num_predict=123)
        out = p.next([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    finally:
        agent.urllib.request.urlopen = orig
    assert out == {"final": "ok"}
    assert captured["body"]["options"]["num_predict"] == 123 and captured["timeout"] == 45


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("OK ", n)
