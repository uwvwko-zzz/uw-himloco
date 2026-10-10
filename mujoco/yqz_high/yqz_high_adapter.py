"""MuJoCo adapter matching the dedicated YQZ high-platform policy."""

from pathlib import Path
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
YQZ000_DIR = SCRIPT_DIR.parent / "yqz_000"
sys.path.insert(0, str(YQZ000_DIR))

from yqz_adapter import YQZ000Adapter  # noqa: E402


class YQZHighAdapter(YQZ000Adapter):
    """Use the action and observation scaling from ``YQZHighCfg``."""

    def __init__(self):
        # YQZHighCfg inherits the current YQZ-000 action scale of 0.35.
        super().__init__(action_scale=0.35, wheel_action_scale=30.0)

    def build_observation(self, model, data, command, **context):
        # YQZHighRobot inherits WheeledLeggedRobot's range normalization.
        # Its fixed [0.47, 0.47] range makes the height observation exactly
        # zero.  YQZ000Adapter produces zero when passed 0.51 here.
        return super().build_observation(
            model, data, command, height=0.51
        )
