import asyncio

import numpy as np
import pytest

from pyrl_trainer.agent import Transition
from pyrl_trainer.config import Config
from pyrl_trainer.learner import SharedState, learner_loop

pytestmark = pytest.mark.system


@pytest.mark.asyncio
async def test_learner_loop_updates_once():
    cfg = Config(
        minibatch=2,
        epochs=1,
        net_hidden=8,
        net_layers=1,
        train_device="cpu",
        infer_device="cpu",
    )
    shared_state = SharedState(obs_dim=3, cfg=cfg)
    experience_q: asyncio.Queue = asyncio.Queue()

    obs = np.zeros(3, dtype=np.float32)
    tr = Transition(
        obs=obs,
        action_turn=0.0,
        action_boost=0.0,
        logp=0.0,
        value=0.0,
        reward=1.0,
        done=0.0,
    )

    await experience_q.put((0, [tr, tr], 0.0))

    task = asyncio.create_task(learner_loop(cfg, shared_state, experience_q))

    try:
        for _ in range(100):
            if shared_state.update_steps >= 1:
                break
            await asyncio.sleep(0.01)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert shared_state.update_steps >= 1
