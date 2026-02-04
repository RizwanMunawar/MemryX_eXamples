import argparse
from pathlib import Path

from agent import Mario
from env import make_env
from utils import parse_cfg, find_newest_cfg, GREEN, RED, RESET

import nes_py.nes_env as nes_env

WINDOW_SCALE = 1.5
nes_env.SCREEN_HEIGHT = int(nes_env.SCREEN_HEIGHT * WINDOW_SCALE)
nes_env.SCREEN_WIDTH = int(nes_env.SCREEN_WIDTH * WINDOW_SCALE)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Play Super Mario Bros using a trained Mario agent."
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="path to the checkpoint file to load the trained agent from",
    )
    parser.add_argument(
        "--render_mode",
        type=str,
        choices=["human", "rgb_array"],
        default="human",
        help="render mode for the environment, default is human",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default=None,
        help="name of environment, default is to use the model's training environment",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="disable exploration for the agent to make actions deterministic",
    )
    parser.add_argument(
        "--mxa",
        action="store_true",
        help="enable memryx accelerator for the agent",
    )
    return parser.parse_args()


def log_game_start(game_idx):
    print(f"Playing game {game_idx:2}...", end=" ", flush=True)


def log_game_info(info):
    color = GREEN if info["flag_get"] else RED
    print(
        f"Score: {info['score']:4}, Reached Flag: {color}{str(info['flag_get']):5}{RESET},",
        f"X Position: {info['x_pos']:4}, Time Remaining: {info['time']}",
    )


if __name__ == "__main__":
    """
    Use this block as a template to configure your own example.
    See `Mario.__init__` in `agent.py` for options to customize Mario.
    See `make_env` in `env.py` for options to cutomize the game environment.
    """
    args = parse_args()
    cfg = parse_cfg(find_newest_cfg(args.ckpt.parent))  # Only for env config
    cfg.env["render_mode"] = args.render_mode
    if args.env_name:
        cfg.env["name"] = args.env_name

    # Make agent and environment
    env = make_env(**cfg.env)
    agent = Mario.from_ckpt(args.ckpt)
    if args.deterministic:
        agent.val()

    # Compile DFP to run on accelerator
    if args.mxa:
        agent.net.compile_dfp()

    # Game Loop
    game_idx = 1
    print(f"{GREEN}Game Log:{RESET}")
    log_game_start(game_idx)
    try:
        state = env.reset()
        while True:
            action = agent.act(state, mxa=args.mxa)
            next_state, reward, done, _, info = env.step(action)
            state = next_state
            if done or info["flag_get"]:
                log_game_info(info)
                game_idx += 1
                state = env.reset()
                log_game_start(game_idx)
    except KeyboardInterrupt:
        print(f"{RED}\nGame stopped by user.{RESET}\n")
    finally:
        env.close()
