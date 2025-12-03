"""Utilities for loading policies and validators for firmware deployment.

This module provides small helpers to load Stable-Baselines3 models and
VecNormalize statistics in a way that is safe for simulation-only testing.
"""
from typing import Optional, Tuple
import os
import logging

logger = logging.getLogger(__name__)


def load_model_and_normalizer(model_path: Optional[str], norm_path: Optional[str], vec_env=None):
    """Try to load a SB3 policy and VecNormalize instance.

    Returns (model, vecnormalize) where any missing item will be None and
    a warning will be logged.
    """
    model = None
    vecnorm = None

    if model_path and os.path.exists(model_path):
        try:
            from stable_baselines3 import PPO
            logger.info(f"Loading model from {model_path}")
            model = PPO.load(model_path)
        except Exception as e:
            logger.warning(f"Failed to load model '{model_path}': {e}")
    else:
        logger.info("No model_path provided or file does not exist; model will be None")

    if norm_path and os.path.exists(norm_path) and vec_env is not None:
        try:
            from stable_baselines3.common.vec_env import VecNormalize
            logger.info(f"Loading VecNormalize from {norm_path}")
            vecnorm = VecNormalize.load(norm_path, vec_env)
            vecnorm.training = False
            vecnorm.norm_reward = False
        except Exception as e:
            logger.warning(f"Failed to load VecNormalize '{norm_path}': {e}")
    else:
        if norm_path:
            logger.info("VecNormalize path provided but vec_env is None or file missing; skipping")

    return model, vecnorm


class PolicyValidator:
    """Simple validator to sanity-check a loaded model runs an inference.

    It does not step simulation; it only checks that model.predict(obs)
    executes without raising for a representative observation.
    """

    @staticmethod
    def quick_inference_check(model, vecnorm, sample_obs):
        """Return True if inference runs, False otherwise.

        - `sample_obs` is expected to be a 1-D numpy array representing
          the stacked observation (no batch dim). This function will add
          a batch dim before calling normalization & inference.
        """
        try:
            import numpy as _np
            obs = _np.asarray(sample_obs, dtype=_np.float32)[None, :]
            if vecnorm is not None:
                # Some VecNormalize wrappers expose `normalize_obs` or `normalize`.
                if hasattr(vecnorm, 'normalize_obs'):
                    obs = vecnorm.normalize_obs(obs)
                else:
                    obs = vecnorm.normalize(obs)

            if model is None:
                # no model is considered a failure for a validator
                logger.warning("No model provided to validator")
                return False

            _ = model.predict(obs, deterministic=True)
            return True
        except Exception as e:
            logger.warning(f"Policy quick inference check failed: {e}")
            return False
