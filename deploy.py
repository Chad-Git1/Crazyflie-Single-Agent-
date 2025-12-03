"""Minimal CLI for validating a firmware model in simulation.

Usage examples:
  python deploy.py --validate --model models/PPO/best_model.zip --norm models/PPO/vecnorm.pkl --xml Assets/bitcraze_crazyflie_2/scene.xml
"""
import argparse
import os
import logging

from src.SimAdapter import SimAdapter

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate', action='store_true', help='Run a smoke validation in MuJoCo')
    parser.add_argument('--model', type=str, default='models/PPO/model.zip', help='Path to PPO model zip')
    parser.add_argument('--norm', type=str, default='models/PPO/vecnormalize.pkl', help='Path to VecNormalize file')
    parser.add_argument('--xml', type=str, default='Assets/bitcraze_crazyflie_2/scene.xml', help='Path to MuJoCo XML')
    parser.add_argument('--steps', type=int, default=200, help='Number of sim steps for validation')
    parser.add_argument('--logid', type=str, default='deploy_validate', help='Log id for flight logger')
    args = parser.parse_args()

    if args.validate:
        logger.info('Starting validation run')
        adapter = SimAdapter(model_path=args.model, norm_path=args.norm, xml_path=args.xml, max_steps=args.steps)
        result = adapter.run_episode(log_id=args.logid)
        logger.info(f"Validation result: {result}")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
