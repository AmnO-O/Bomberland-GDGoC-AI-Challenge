import json
import tempfile
from pathlib import Path

from competition.evaluation.match_runner import MatchRunner
from competition.evaluation.runtime_guard import runtime_precheck


def _write_agent(path: Path, code: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(code, encoding="utf-8")


def test_runtime_precheck_rejects_timeout_agent():
    with tempfile.TemporaryDirectory() as td:
        agent_path = Path(td) / "timeout_agent.py"
        _write_agent(
            agent_path,
            """
class Agent:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id

    def act(self, obs):
        while True:
            pass
""",
        )

        ok, reason = runtime_precheck(str(agent_path), timeout_s=0.05)
        assert ok is False
        assert "timeout" in reason


def test_runtime_precheck_rejects_invalid_action_agent():
    with tempfile.TemporaryDirectory() as td:
        agent_path = Path(td) / "invalid_action_agent.py"
        _write_agent(
            agent_path,
            """
class Agent:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id

    def act(self, obs):
        return 99
""",
        )

        ok, reason = runtime_precheck(str(agent_path), timeout_s=0.05)
        assert ok is False
        assert reason == "runtime_precheck_invalid_action"


def test_match_runner_survives_fraud_agents_with_fallbacks():
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)

        timeout_agent = td_path / "timeout_agent.py"
        invalid_agent = td_path / "invalid_agent.py"
        safe_agent_1 = td_path / "safe_agent_1.py"
        safe_agent_2 = td_path / "safe_agent_2.py"

        _write_agent(
            timeout_agent,
            """
class Agent:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id

    def act(self, obs):
        while True:
            pass
""",
        )
        _write_agent(
            invalid_agent,
            """
class Agent:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id

    def act(self, obs):
        return 42
""",
        )
        _write_agent(
            safe_agent_1,
            """
class Agent:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id

    def act(self, obs):
        return 0
""",
        )
        _write_agent(
            safe_agent_2,
            """
class Agent:
    def __init__(self, agent_id=0):
        self.agent_id = agent_id

    def act(self, obs):
        return 1
""",
        )

        runner = MatchRunner(log_dir=str(td_path / "logs"))
        ranks, survival_steps, _gif_path, json_path = runner.run_match(
            agent_paths=[
                str(timeout_agent),
                str(invalid_agent),
                str(safe_agent_1),
                str(safe_agent_2),
            ],
            team_ids=["timeout", "invalid", "safe1", "safe2"],
            seed=7,
            max_steps=5,
            inference_timeout_s=0.05,
        )

        assert len(ranks) == 4
        assert len(survival_steps) == 4

        with open(json_path, "r", encoding="utf-8") as f:
            payload = json.load(f)

        stats = payload.get("runtime_stats", {})
        assert stats["0"]["timeouts"] > 0
        assert stats["1"]["invalid_actions"] > 0
