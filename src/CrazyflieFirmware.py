"""Firmware-side controller wrapper to run SB3 policies in sim or on hardware.

This module provides a `CrazyflieFirmware` class that exposes a simple
`predict_action(obs)` API where `obs` is the stacked observation produced
by the environment. It handles normalization, inference, action scaling,
and basic safety checks. It purposely keeps hardware driver code behind a
thin `DriverInterface` which can be implemented by a `SimDriver` for tests
and by a `CrazyflieDriver` (using `cflib`) for real hardware.
"""
from typing import Optional, Sequence, Tuple
import logging
import numpy as np

from FirmwareDeploymentUtils import load_model_and_normalizer

logger = logging.getLogger(__name__)


class DriverInterface:
    """Minimal driver interface used by firmware.

    Implementations should provide a `send_action(action)` method where
    `action` is a 4-element array [thrust, mx, my, mz].
    """

    def send_action(self, action: Sequence[float]):
        raise NotImplementedError()


class SimDriver(DriverInterface):
    """Driver used in simulation that simply stores the last action.

    This allows the firmware to call `send_action` without requiring radio
    hardware. The SimAdapter or test harness can read `last_action` if
    desired.
    """

    def __init__(self):
        self.last_action = np.zeros(4, dtype=np.float32)

    def send_action(self, action: Sequence[float]):
        a = np.asarray(action, dtype=np.float32).reshape(4,)
        self.last_action = a


class CrazyflieFirmware:
    def __init__(self, model_path: Optional[str] = None, norm_path: Optional[str] = None,
                 vec_env=None, driver: Optional[DriverInterface] = None, hover_thrust: float = 0.27):
        self.driver = driver if driver is not None else SimDriver()
        self.hover_thrust = float(hover_thrust)

        model, vecnorm = load_model_and_normalizer(model_path, norm_path, vec_env)
        self.model = model
        self.vecnorm = vecnorm

        # Basic action limits (tuned to the environment used in training)
        self.thrust_min = 0.0
        self.thrust_max = 0.35
        self.moment_limit = 0.25

    def process_observation(self, obs_stacked: Sequence[float]) -> np.ndarray:
        """Extract the newest single-frame observation from a stacked vector.

        The environment uses 13-element single frames and stacks them.
        This function returns the latest frame (last 13 values) as a 1-D
        numpy array of dtype float32.
        """
        arr = np.asarray(obs_stacked, dtype=np.float32).ravel()
        if arr.size >= 13:
            single = arr[-13:]
        else:
            # Fallback: pad zeros if the shape is unexpected
            single = np.zeros(13, dtype=np.float32)
            single[-arr.size:] = arr
        return single

    def _normalize_for_policy(self, obs_stacked: Sequence[float]) -> np.ndarray:
        import numpy as _np
        obs = _np.asarray(obs_stacked, dtype=_np.float32)[None, :]
        if self.vecnorm is not None:
            # prefer normalize_obs if available
            if hasattr(self.vecnorm, 'normalize_obs'):
                return self.vecnorm.normalize_obs(obs)
            else:
                return self.vecnorm.normalize(obs)
        return obs

    def _apply_safety_and_scale(self, raw_action: Sequence[float], obs_single: np.ndarray) -> np.ndarray:
        a = np.asarray(raw_action, dtype=np.float32).reshape(4,)
        # Clamp thrust
        a[0] = float(np.clip(a[0], self.thrust_min, self.thrust_max))
        # Clamp moments
        a[1:] = np.clip(a[1:], -self.moment_limit, self.moment_limit)

        # Safety checks: if altitude below threshold or nan occurs, return hover
        z = float(obs_single[2])
        if not np.isfinite(a).all() or z < -0.5:
            logger.warning("Safety trigger: invalid action or low altitude; switching to hover")
            return np.array([self.hover_thrust, 0.0, 0.0, 0.0], dtype=np.float32)

        # Return scaled/clamped action
        return a

    def predict_action(self, obs_stacked: Sequence[float]) -> np.ndarray:
        """Return a 4-element action given a stacked observation.

        This method normalizes the observation (if VecNormalize present),
        runs policy.predict(..., deterministic=True) and applies clamping
        and safety fallbacks.
        """
        obs_single = self.process_observation(obs_stacked)
        obs_for_policy = self._normalize_for_policy(obs_stacked)

        # If no model, return hover
        if self.model is None:
            return np.array([self.hover_thrust, 0.0, 0.0, 0.0], dtype=np.float32)

        try:
            action, _ = self.model.predict(obs_for_policy, deterministic=True)
            action = np.asarray(action, dtype=np.float32).squeeze()
        except Exception as e:
            logger.warning(f"Model predict failed: {e}; returning hover")
            return np.array([self.hover_thrust, 0.0, 0.0, 0.0], dtype=np.float32)

        action_safe = self._apply_safety_and_scale(action, obs_single)
        return action_safe

    def step_and_send(self, obs_stacked: Sequence[float]):
        a = self.predict_action(obs_stacked)
        self.driver.send_action(a)
        return a
