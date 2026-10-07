from __future__ import annotations

from pathlib import Path
from typing import Any

import orbax.checkpoint as ocp

from ..training.trainer import TrainState


def create_checkpoint_manager(
    drive_dir: str | Path, max_to_keep: int = 3
) -> ocp.CheckpointManager:
    """
    Creates a manager that keep the `max_to_keep` most recent steps.

    Args:
        drive_dir: Path under mounted Drive
        max_to_keep: How many recent checkponts to retain (older ones are deleted automatically as new ones are saved)
    """
    directory = Path(drive_dir).resolve()

    options = ocp.CheckpointManagerOptions(
        max_to_keep=max_to_keep,
        create=True,
    )

    return ocp.CheckpointManager(
        directory,
        options=options,
    )


def save_checkpoint(
    manager: ocp.CheckpointManager, state: TrainState, metrics: dict | None = None
) -> None:
    """
    Saves `state` at its own step number. Call this periodically inside the training loop e.g. every N steps or every eval.
    """
    payload = {
        "params": state.params,
        "opt_state": state.opt_state,
        "step": state.step,
    }
    manager.save(state.step, args=ocp.args.StandardSave(payload), metrics=metrics)
    manager.wait_until_finished()


def return_latest(
    manager: ocp.CheckpointManager, target_state: TrainState
) -> TrainState:
    """
    Returns the most recent checkpoint, if one exits

    """
    latest_step = manager.latest_step()
    if latest_step is None:
        return target_state

    target_payload = {
        "params": target_state.params,
        "opt_state": target_state.opt_state,
        "step": target_state.step,
    }

    restored = manager.restore(
        latest_step, args=ocp.args.StandardRestore(target_payload)
    )
    return TrainState(
        params=restored["params"],
        opt_state=restored["opt_state"],
        step=restored["step"],
    )


def load_checkpoint(
    path: str | Path, target_params: dict, target_opt_state=None
) -> tuple[TrainState, dict[str, Any]]:
    """
    Loads model parameters from the most recent orbax checkpoint.

    Args:
        path: Checkpoint manager directory
        target_params: Initialized model parameter used to provide orbax with the expected PyTree structure

    Returns:
        Restored model parameters and checkpoint metadata
    """
    directory = Path(path).resolve()

    manager = ocp.CheckpointManager(
        directory, options=ocp.CheckpointManagerOptions(create=False)
    )

    latest_step = manager.latest_step()

    if latest_step is None:
        raise FileNotFoundError(f"No checkpoint found in {directory}")

    target = {
        "params": target_params,
        "step": 0,
    }

    if target_opt_state is not None:
        target["opt_state"] = target_opt_state

    restored = manager.restore(
        latest_step,
        args=ocp.args.PyTreeRestore(target, partial_restore=True),
    )

    params = restored["params"]

    extra = {
        "step": restored["step"],
    }

    return params, extra
