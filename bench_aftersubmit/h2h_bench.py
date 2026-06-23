import os
import sys
import random
import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

# ------------------------------------------------------------------
# Xác định thư mục gốc của repository
script_dir = Path(__file__).resolve().parent
REPO_ROOT = script_dir.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ------------------------------------------------------------------
# Import các module cần thiết
from engine.game import BomberEnv
from agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, GeniusRuleAgent, BoxFarmerAgent, TacticalRuleAgent
from competition.evaluation.runtime_guard import load_agent_instance

# ------------------------------------------------------------------
# Định nghĩa các agent (đường dẫn đã được căn chỉnh)
AGENT_SPECS = {
    "sub_v9_up": {"kind": "file", "path": str(REPO_ROOT / "agent" / "submission_v9" / "submission_v9_up.py")},
    "sub_v8_up": {"kind": "file", "path": str(REPO_ROOT / "agent" / "submission_v8" / "submission_v8_up.py")},
    "v6":        {"kind": "file", "path": str(REPO_ROOT / "agent" / "submission_v6" / "submission_v6.py")},
    "core_base": {"kind": "file", "path": str(REPO_ROOT / "agent" / "samnu_agent.py")},
    "model_v8" : {"kind": "file", "path": str(REPO_ROOT / "agent" / "hisu_v8" / "agent.py")},
    "top_pool_1" : {"kind" : "file", "path": str(REPO_ROOT / "agent" / "submission_v9" / "agent1.py") },
    "top_pool_2" : {"kind" : "file", "path": str(REPO_ROOT / "agent" / "submission_v9" / "agent2.py") },
    "last_dance" : {"kind" : "file", "path" : str(REPO_ROOT / "agent" / "submission_v10" / "tmp.py") },
    "genius":    {"kind": "baseline", "baseline": "GeniusRuleAgent"},
    "smarter":   {"kind": "baseline", "baseline": "SmarterRuleAgent"},
    "tactical":  {"kind": "baseline", "baseline": "TacticalRuleAgent"},
}
OTHERS = ["genius", "smarter", "v6", "tactical", "core_base", "model_v8", "sub_v8_up", "top_pool_1", "top_pool_2"]

# ------------------------------------------------------------------
def load_agent_by_spec(agent_name, player_id, agent_specs, repo_root):
    spec = agent_specs[agent_name]
    if spec["kind"] == "baseline":
        baseline_name = spec["baseline"]
        if baseline_name == "RandomAgent":
            return RandomAgent(player_id)
        elif baseline_name == "SimpleRuleAgent":
            return SimpleRuleAgent(player_id)
        elif baseline_name == "SmarterRuleAgent":
            return SmarterRuleAgent(player_id)
        elif baseline_name == "GeniusRuleAgent":
            return GeniusRuleAgent(player_id)
        elif baseline_name == "BoxFarmerAgent":
            return BoxFarmerAgent(player_id)
        elif baseline_name == "TacticalRuleAgent":
            return TacticalRuleAgent(player_id)
        else:
            raise ValueError(f"Unknown baseline agent: {baseline_name}")
    elif spec["kind"] == "file":
        agent_path = Path(spec["path"])
        if not agent_path.exists():
            raise FileNotFoundError(f"Agent file not found: {agent_path}")
        return load_agent_instance(str(agent_path), player_id)
    else:
        raise ValueError(f"Unknown agent kind: {spec['kind']}")

# ------------------------------------------------------------------
def run_one_match(task):
    lineup = task["lineup"]
    agent_specs = task["agent_specs"]
    repo_root = Path(task["repo_dir"])
    max_steps = task.get("max_steps", 500)
    seed = task.get("seed")
    timeout_ms = task.get("timeout_ms", 100.0)

    n_players = len(lineup)
    agents = []
    names = []
    for i, name in enumerate(lineup):
        try:
            agent = load_agent_by_spec(name, i, agent_specs, repo_root)
            agents.append(agent)
            if hasattr(agent, "team_id"):
                names.append(agent.team_id)
            else:
                names.append(name)
        except Exception as e:
            return {
                "failed": True,
                "error": f"Failed to load agent {name}: {e}",
                "ranks": [], "stats": [], "alive_final": [], "runtime": {}
            }

    env = BomberEnv(max_steps=max_steps, seed=seed)
    obs = env.reset()
    done = False
    step = 0
    runtime_data = {str(i): {"times_ms": [], "timeout_over_100ms": 0} for i in range(n_players)}

    while not done and step < max_steps:
        actions = []
        for i in range(n_players):
            start = time.perf_counter()
            try:
                action = agents[i].act(obs)
            except Exception as e:
                print(f"Agent {names[i]} error: {e}")
                action = 0
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            runtime_data[str(i)]["times_ms"].append(elapsed_ms)
            if elapsed_ms > timeout_ms:
                runtime_data[str(i)]["timeout_over_100ms"] += 1
            actions.append(action)

        obs, terminated, truncated = env.step(actions)
        done = terminated or truncated
        step += 1

    # Lấy thống kê cuối cùng từ các đối tượng Player bên trong env
    final_stats = []
    alive_final = []
    for i in range(n_players):
        player = env.players[i]
        alive_final.append(player.alive)
        stats_dict = getattr(player, "stats", {})
        final_stats.append({
            "kills": stats_dict.get("kills", 0),
            "boxes": stats_dict.get("boxes", 0),
            "items": stats_dict.get("items", 0),
            "bombs": stats_dict.get("bombs", 0),
        })

    # Xếp hạng: còn sống trước, sau đó kill, boxes, items, bombs
    ranking = []
    for i in range(n_players):
        ranking.append((
            i,
            alive_final[i],
            final_stats[i]["kills"],
            final_stats[i]["boxes"],
            final_stats[i]["items"],
            final_stats[i]["bombs"]
        ))
    ranking.sort(key=lambda x: (x[1], x[2], x[3], x[4], x[5]), reverse=True)
    ranks = [0] * n_players
    for pos, (idx, _, _, _, _, _) in enumerate(ranking):
        ranks[idx] = pos

    return {
        "failed": False,
        "ranks": ranks,
        "stats": final_stats,
        "alive_final": alive_final,
        "runtime": runtime_data
    }

# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="last_dance", help="Các agent cần so sánh, cách nhau bằng dấu phẩy")
    parser.add_argument("--matches", type=int, default=800, help="Số trận đấu")
    parser.add_argument("--seed", type=int, default=999, help="Hạt giống random")
    parser.add_argument("--out", default="res.json", help="File output kết quả")
    parser.add_argument("--log_deaths", action="store_true", default=True, help="Lưu các trận mà target agent bị chết vào file death_logs.json")
    parser.add_argument("--death_log_file", default="death_logs.json", help="File để lưu log các trận bị chết")
    args = parser.parse_args()

    targets = args.targets.split(",")
    n_fill = 4 - len(targets)

    agg = defaultdict(lambda: {
        "rank_sum": 0, "games": 0, "wins": 0, "deaths": 0,
        "kills": 0, "boxes": 0, "items": 0, "bombs": 0,
        "maxms": 0.0, "timeouts": 0
    })

    rng = random.Random(args.seed)
    done = 0
    t0 = time.time()
    out = {}
    death_logs = []

    for m in range(args.matches):
        fillers = [rng.choice(OTHERS) for _ in range(n_fill)]
        lineup = targets + fillers
        rng.shuffle(lineup)

        task = {
            "match_idx": m,
            "seed": args.seed + m * 7 + 1,
            "max_steps": 500,
            "timeout_ms": 100.0,
            "fault_limit": 100000,
            "lineup": lineup,
            "agent_specs": AGENT_SPECS,
            "repo_dir": str(REPO_ROOT)
        }

        result = run_one_match(task)

        if result.get("failed"):
            print(f"Trận {m} thất bại: {result.get('error', '')[:120]}", flush=True)
            continue

        ranks = result["ranks"]
        stats = result["stats"]
        alive = result["alive_final"]
        runtime = result["runtime"]

        # Cập nhật tổng hợp
        for seat, name in enumerate(lineup):
            a = agg[name]
            a["rank_sum"] += ranks[seat]
            a["games"] += 1
            if ranks[seat] == 0:
                a["wins"] += 1
            if not alive[seat]:
                a["deaths"] += 1
            st = stats[seat]
            a["kills"] += st["kills"]
            a["boxes"] += st["boxes"]
            a["items"] += st["items"]
            a["bombs"] += st["bombs"]
            times = runtime.get(str(seat), {}).get("times_ms", [])
            if times:
                a["maxms"] = max(a["maxms"], max(times))
            a["timeouts"] += runtime.get(str(seat), {}).get("timeout_over_100ms", 0)

        # Lưu log death nếu được yêu cầu
        if args.log_deaths:
            for seat, name in enumerate(lineup):
                if name in targets and not alive[seat]:
                    death_logs.append({
                        "match": m,
                        "seed": task["seed"],
                        "lineup": lineup,
                        "ranks": ranks,
                        "stats": stats,
                        "alive": alive,
                        "runtime": runtime,
                        "death_seat": seat,
                        "death_name": name
                    })

        done += 1

        # Cập nhật kết quả tổng hợp
        out = {
            "done": done,
            "matches": args.matches,
            "targets": targets,
            "elapsed_s": round(time.time() - t0, 1),
            "agents": {}
        }
        for name, a in agg.items():
            g = a["games"] or 1
            out["agents"][name] = {
                "games": a["games"],
                "avgRank": round(a["rank_sum"] / g, 3),
                "win_pct": round(100 * a["wins"] / g, 1),
                "death_pct": round(100 * a["deaths"] / g, 1),
                "kills": round(a["kills"] / g, 2),
                "boxes": round(a["boxes"] / g, 2),
                "items": round(a["items"] / g, 2),
                "bombs": round(a["bombs"] / g, 1),
                "maxms": round(a["maxms"], 1),
                "timeouts": a["timeouts"]
            }

        # Ghi res.json sau mỗi 5 trận
        if done % 5 == 0:
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
            # Ghi death_logs.json cùng lúc
            if args.log_deaths:
                with open(args.death_log_file, "w") as f:
                    json.dump(death_logs, f, indent=2)
            print(f"  {done}/{args.matches} ({out['elapsed_s']}s)", flush=True)

    # Ghi lần cuối cùng (đảm bảo dữ liệu được lưu đầy đủ)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    if args.log_deaths:
        with open(args.death_log_file, "w") as f:
            json.dump(death_logs, f, indent=2)
        print(f"Đã lưu {len(death_logs)} trận bị chết vào {args.death_log_file}")

    print("Hoàn thành!", flush=True)

if __name__ == "__main__":
    main()