import os
import random
import argparse
import sys
from pathlib import Path
parent_dir = Path(__file__).resolve().parent.parent
# Add parent directory to sys.path if not already present
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


import torch

from engine.game import BomberEnv
from agent import RandomAgent, SimpleRuleAgent, SmarterRuleAgent, GeniusRuleAgent, BoxFarmerAgent, TacticalRuleAgent
from training.DQN import DQNAgent, encode_obs


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "t"}:
        return True
    if value in {"false", "0", "no", "n", "f"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")

def run_match(model_paths, num_episodes=10, max_steps=500, seed=None):
    env = BomberEnv(max_steps=max_steps, seed=seed)
    n_players = len(model_paths)
    agents = [None] * n_players
    info = [{"name": None, "wins": 0} for _ in range(n_players)]
    
    if seed is not None:
        random.seed(seed)
    
    for i, path in enumerate(model_paths):
        if path != "None":
            # suppose submission file is /agent/team_name/agent.py -> extract team_name as agent name
            checkpoint = torch.load(path, map_location="cpu")
            input_dim = checkpoint["input_dim"]
            num_actions = checkpoint["num_actions"]
            agents[i] = DQNAgent(i, input_dim, num_actions, lr=1e-3, device="cuda" if torch.cuda.is_available() else "cpu", pretrained_model=path)
            agents[i].load_agent(pretrained_model=path)
            info[i]["name"] = os.path.basename(path)
        else:
            # Random rule-based agent for headless testing
            x = random.randint(0, 6)
            if x == 0:
                info[i]["name"] = "RandomAgent"
                agents[i] = RandomAgent(i)
            elif x == 1:
                info[i]["name"] = "SimpleRuleAgent"
                agents[i] = SimpleRuleAgent(i)
            elif x == 2:
                info[i]["name"] = "SmarterRuleAgent"
                agents[i] = SmarterRuleAgent(i)
            elif x == 3:
                info[i]["name"] = "GeniusRuleAgent"
                agents[i] = GeniusRuleAgent(i)
            elif x == 4:
                info[i]["name"] = "BoxFarmerAgent"
                agents[i] = BoxFarmerAgent(i)
            else:
                info[i]["name"] = "TacticalRuleAgent"
                agents[i] = TacticalRuleAgent(i)

    for episode in range(num_episodes):
        episode_seed = None if seed is None else seed + episode
        obs = env.reset(seed=episode_seed)
        encoded_obs = encode_obs(obs, agent_ids=[i for i in range(n_players)])
        done = False
        step = 0
        death_order = []
        prev_alive = [bool(p[2]) for p in obs["players"]]

        while not done and step < max_steps:
            encoded_obs = encode_obs(obs, agent_ids=[i for i in range(n_players)])
            # actions = [agent.act(obs) for agent in agents]
            actions = []
            for i in range(n_players):
                if isinstance(agents[i], DQNAgent):
                    actions.append(agents[i].act(encoded_obs, epsilon=0.05))
                else:
                    actions.append(agents[i].act(obs))
            obs, terminated, truncated = env.step(actions)
            done = terminated or truncated
            step += 1

            alive_now = [bool(p[2]) for p in obs["players"]]
            for i in range(n_players):
                if prev_alive[i] and not alive_now[i]:
                    death_order.append(info[i]["name"])
            prev_alive = alive_now
        
        alive_final = [bool(p[2]) for p in obs["players"]]
        survivors = [i for i in range(n_players) if alive_final[i]]
        
        if len(survivors) == 1:
            winner = survivors[0]
            info[winner]["wins"] += 1
            print(f"Episode {episode + 1}: {info[winner]['name']} wins | Died: {death_order}")
        else:
            print(f"Episode {episode + 1}: Draw | Died: {death_order}")

    print("\n=== Summary ===")
    for i in range(n_players):
        print(f"{info[i]['name']}: {info[i]['wins']} wins")
    return info

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_paths", nargs="+", default=["None", "None", "None", "None"])
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--visualize", type=str2bool, default=False)
    parser.add_argument("--autoplay", type=str2bool, default=True)
    args = parser.parse_args()
    
    run_match(
        model_paths=args.model_paths,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        seed=args.seed
    )

    if args.visualize:
        # Lazy import so non-visual usage does not require pygame.
        from visualizer.run_local_evaluation import run_simple_viewer

        run_simple_viewer(
            model_paths=args.model_paths,
            num_episodes=args.num_episodes,
            max_steps=args.max_steps,
            seed=args.seed,
            autoplay=args.autoplay,
        )