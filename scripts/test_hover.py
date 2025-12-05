import logging
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.positioning.motion_commander import MotionCommander

# URI to the Crazyflie to connect to
URI = 'radio://0/80/2M/E7E7E7E7E7'

def simple_hover(scf):
    print("Taking off...")
    # We use MotionCommander to handle the takeoff/landing automatically
    # default_height=0.3 meters (30cm)
    with MotionCommander(scf, default_height=0.3) as mc:
        print("Hovering at 0.3m for 3 seconds...")
        time.sleep(3)
        print("Landing...")
        # Exiting the 'with' block automatically triggers landing

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    print(f"Connecting to {URI}...")
    
    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            print("Connected!")
            simple_hover(scf)
            print("Done!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the drone is ON and the Flow Deck is attached (if you have one).")
