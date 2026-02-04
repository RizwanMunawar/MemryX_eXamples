from tqdm import tqdm
from pathlib import Path
import datetime
from argparse import ArgumentParser
import toml
from shutil import copyfile

from env import make_env
from utils import parse_cfg, copy_cfg_file, GREEN, RED, RESET
from agent import Mario


def train(
    agent,
    env,
    episodes=40_000,
    print_every=None,
    save_every=None,
    cfg_path=None,
    resume=False,
):
    """
    Defines the training loop of the agent interacting with and learning from the environment.

    Args:
        agent (agent.Mario): The agent to train.
        env (gym.Env): The environment to train the agent on.
        episodes (int): The number of episodes to train the agent for.
        print_every (int): Number of episodes between printing metrics.
        save_every (int): Number of episodes between saving checkpoints.
        cfg_path (str): To copy the cfg file to the training directory.
        resume (bool): Whether to resume training from a checkpoint.

    Returns:
        agent (agent.Mario): The trained agent (containing the trained Q-networks).
    """
    # Logging
    print(
        f"{GREEN}Training Mario for {episodes:,} episodes in the {agent.env_name} environment.",
        f"KeyboardInterrupt anytime to stop training and save final checkpoint.{RESET}",
    )
    logger = agent.logger
    save_dir = agent.save_dir
    if not resume:
        save_dir.mkdir(parents=True)
        logger._init_logfile()
    print_every = print_every if print_every else max(episodes // 50, 10)
    save_every = save_every if save_every else episodes // 10

    print(f"Saving logs and checkpoints to: {str(save_dir)}")

    # Save run config
    if cfg_path:
        copy_cfg_file(cfg_path, save_dir)

    # Progress bar for episodes
    start = logger.episode + 1
    end = start + episodes
    episodes_pbar = tqdm(range(start, end), unit="episodes", colour="green", ncols=150)

    # Training loop
    for e in episodes_pbar:
        state = env.reset()
        # Play the game!
        while True:
            # Agent decides what action to take based on observed state
            action = agent.act(state)
            # Agent performs action and receives next state and reward
            next_state, reward, done, _, info = env.step(action)
            # Agent saves the experience to memory
            agent.cache(state, next_state, action, reward, done)
            # Agent learns from a random sample of memory
            q, loss = agent.learn()

            logger.log_step(reward, loss, q)
            state = next_state
            # Check if end of game
            if done or info["flag_get"]:
                break
        logger.log_episode()

        # Print Log
        if (e % print_every == 0) or (e == episodes):
            logger.record(
                epsilon=agent.exploration_rate,
                print_fn=episodes_pbar.write,
            )

        # Save checkpoint
        if (e % save_every == 0) or (e == episodes):
            agent.save(save_dir / f"mario_net_{e}.ckpt", print_fn=episodes_pbar.write)

    return agent


def parse_args():
    parser = ArgumentParser(
        description=(
            "Train Mario to play Super Mario Bros using Double Deep Q-Learning. "
        )
    )
    parser.add_argument(
        "--cfg",
        type=Path,
        required=True,
        help=(
            "path to the configuration file, defaults to cfg/default.toml, "
            "use this to customize agent, environment, and training parameters"
        ),
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        metavar="CKPT",
        help="path to checkpoint to resume training from, the original save_dir will be used",
    )
    parser.add_argument(
        "--save_dir",
        type=Path,
        default=Path("runs") / datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S"),
        help=(
            "directory to save logs and checkpoints, defaults to runs/<timestamp>, "
            "unused if resuming training"
        ),
    )

    return parser.parse_args()


if __name__ == "__main__":
    """
    Use this block as a template to configure your own example.
    See `train` in `train.py` (above) for options to customize training.
    See `Mario.__init__` in `agent.py` for options to customize Mario.
    See `make_env` in `env.py` for options to cutomize the game environment.
    """
    # Parse args
    args = parse_args()
    cfg = parse_cfg(args.cfg)

    # Make agent and environment
    env = make_env(**cfg.env)

    if args.resume:
        mario = Mario.from_ckpt(args.resume, cfg.agent)
    else:
        mario = Mario(
            state_dim=env.observation_space.shape,
            action_dim=env.action_space.n,
            env_name=env.name,
            save_dir=args.save_dir,
            **cfg.agent,
        )
    try:
        # Train agent
        mario = train(
            mario,
            env,
            resume=bool(args.resume),
            cfg_path=args.cfg,
            **cfg.train,
        )
    except KeyboardInterrupt:
        print(f"{RED}Training stopped by user.{RESET}")
    finally:
        # Checkpoint final model
        mario.save(mario.save_dir / f"mario_net_final.ckpt")
        env.close()
        print()
