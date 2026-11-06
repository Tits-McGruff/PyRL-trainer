"""Task supervision tests."""

# pylint: disable=import-error,protected-access

import asyncio

import pytest

from pyrl_trainer.__main__ import _supervise_tasks

pytestmark = pytest.mark.unit


@pytest.mark.asyncio
async def test_fatal_task_cancels_and_settles_sibling():
    """A fatal runtime task failure cancels and settles unfinished siblings."""
    sibling_cancelled = asyncio.Event()

    async def failing():
        await asyncio.sleep(0)
        raise RuntimeError("fatal learner failure")

    async def sibling():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise

    failed = asyncio.create_task(failing())
    other = asyncio.create_task(sibling())

    with pytest.raises(RuntimeError, match="fatal learner failure"):
        await _supervise_tasks([failed, other])

    assert sibling_cancelled.is_set()
    assert failed.done()
    assert other.done()
    assert other.cancelled()


@pytest.mark.asyncio
async def test_supervisor_cancellation_settles_children():
    """External cancellation settles every task owned by the supervisor."""
    child_cancelled = asyncio.Event()

    async def child():
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            child_cancelled.set()
            raise

    task = asyncio.create_task(child())
    supervisor = asyncio.create_task(_supervise_tasks([task]))
    await asyncio.sleep(0)
    supervisor.cancel()

    with pytest.raises(asyncio.CancelledError):
        await supervisor

    assert child_cancelled.is_set()
    assert task.done()
    assert task.cancelled()


@pytest.mark.asyncio
async def test_completed_budget_settles_live_actor():
    """The learner can stop a bounded run while its actor is still connected."""
    actor_started = asyncio.Event()

    async def learner():
        await actor_started.wait()

    async def actor():
        actor_started.set()
        await asyncio.Event().wait()

    completion = asyncio.create_task(learner())
    ongoing = asyncio.create_task(actor())
    await _supervise_tasks([completion, ongoing], completion_task=completion)
    assert completion.done()
    assert ongoing.cancelled()
