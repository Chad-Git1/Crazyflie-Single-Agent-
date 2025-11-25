import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from FirmwareDeploymentUtils import PolicyValidator

here = os.path.dirname(__file__)
model_path = os.path.abspath(os.path.join(here, 'models', 'Complex2_DR', 'complex_dr.zip'))
norm_path = os.path.abspath(os.path.join(here, 'models', 'Complex2_DR', 'vecnormalize_dr.pkl'))
xml_path = os.path.abspath(os.path.join(here, 'Assets', 'bitcraze_crazyflie_2', 'scene.xml'))

print('Model:', model_path)
print('Norm :', norm_path)
print('XML  :', xml_path)

ok = PolicyValidator.test_policy_inference(model_path, norm_path, xml_path, num_steps=20)
print('Validation result:', ok)
