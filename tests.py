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


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f(); print("OK ", n)
