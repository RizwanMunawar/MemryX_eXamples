import toml
from argparse import Namespace
from pathlib import Path
import shutil

import models
from gym_super_mario_bros import actions

# Constants
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"


# Helpers
def parse_cfg(cfg_path):
    """See cfg/default.toml for an example of the configuration file."""
    print(f"Loading configuration from: {str(cfg_path)}\n")

    cfg = toml.load(cfg_path)
    cfg = Namespace(**cfg)

    # Convert model_name to model_constructor, they should be the same name
    if hasattr(cfg, "agent"):
        model_constructor = getattr(models, cfg.agent["model_name"], None)
        if model_constructor is None:
            raise AttributeError(
                f"Model {cfg.agent['model_name']} not found in models.py."
            )
        cfg.agent["model_constructor"] = model_constructor

    # Convert moveset to gym_super_mario_bros.actions
    if hasattr(cfg, "env"):
        if isinstance(cfg.env["moveset"], str):
            moveset = getattr(actions, cfg.env["moveset"], None)
            if moveset is None:
                raise AttributeError(
                    f"Moveset {cfg.env['moveset']} not found in gym_super_mario_bros.actions."
                )
            cfg.env["moveset"] = moveset

    return cfg


def copy_cfg_file(cfg_path, save_dir):
    """Copy the configuration file to the save_dir, increment path if necessary."""
    cfg_files = list(save_dir.glob("cfg*.toml"))

    # Just copy file
    if len(cfg_files) == 0:
        dst = save_dir / "cfg-0.toml"
    else:
        # Increment the path if the file already exists
        i = 1
        while (dst := save_dir / f"cfg-{i}.toml").exists():
            i += 1
    shutil.copyfile(cfg_path, dst)

    # Prepend the message to the copied file
    warning_msg = (
        f"# Don't edit this file to preserve your training log.\n"
        f"# Make changes to and use the original config file in cfg/.\n\n"
    )
    with open(dst, "r+") as file:
        content = file.read()
        file.seek(0, 0)
        file.write(warning_msg + content)


def find_newest_cfg(save_dir):
    """Find the newest configuration file in the save_dir."""
    cfg_files = list(save_dir.glob("cfg*.toml"))
    if len(cfg_files) == 1:
        return cfg_files[0]

    # Sort files by their index, they all have the form cfg-i.toml
    cfg_files.sort(key=lambda f: int(f.stem.split("-")[-1]))
    return cfg_files[0]


def state_to_array(state):
    """Converts the given state (LazyFrame or tuple[LazyFrame, ...]) to a numpy array."""
    return state[0].__array__() if isinstance(state, tuple) else state.__array__()
