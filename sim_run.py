"""
Simple script to run the trained policy in MuJoCo simulation using SimAdapter.
Usage:
    python sim_run.py --model models/Complex2_DR/complex_dr.zip --norm models/Complex2_DR/vecnormalize_dr.pkl --steps 500
"""
import argparse
import os
from src.SimAdapter import SimAdapter

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=os.path.join('models','Complex2_DR','complex_dr.zip'))
parser.add_argument('--norm', type=str, default=os.path.join('models','Complex2_DR','vecnormalize_dr.pkl'))
parser.add_argument('--xml', type=str, default=os.path.join('Assets','bitcraze_crazyflie_2','scene.xml'))
parser.add_argument('--steps', type=int, default=500)
parser.add_argument('--logid', type=str, default='sim_test')
args = parser.parse_args()

adapter = SimAdapter(model_path=args.model, norm_path=args.norm, xml_path=args.xml, target_z=1.0, max_steps=args.steps)
result = adapter.run_episode(render=False, log_id=args.logid)
print('Result:', result)
