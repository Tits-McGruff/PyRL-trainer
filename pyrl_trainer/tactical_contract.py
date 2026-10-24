"""Stable identifiers for native tactical corpus and graph-input studies."""

SCENARIOS = (
    "open-wander", "approaching", "receding", "crossing", "head-on", "corridor",
    "crowd-opening", "wall-escape", "food-body-trap", "smaller-intercept",
    "larger-threat", "boost-escape", "predicted-crossing", "saturation",
)
POLICIES = ("motion-v2-deterministic", "motion-v2-stochastic")
# Split port zero is the scalar prefix; later ports are canonical Motion channels.
VARIANTS = {
    "full": (), "no-bearing": (8,), "no-heading": (10, 11),
    "no-velocity": (12, 13), "no-urgency": (14,), "no-density": (15,),
    "core-only": tuple(range(7, 16)),
}



def policies_for_profile(profile):
    """Name the actual deterministic/stochastic teachers of one declared v4 corpus."""
    if profile not in ("basic", "motion"):
        raise ValueError("tactical corpus requires a v4 Basic/Motion profile")
    return tuple(profile + "-v2-" + mode for mode in ("deterministic", "stochastic"))


def variants_for_profile(profile):
    """Basic trains its complete declared input; Motion retains the seven paired ablations."""
    policies_for_profile(profile)
    return tuple(VARIANTS) if profile == "motion" else ("full",)
