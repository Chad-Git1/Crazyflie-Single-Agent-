import logging
import time
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

# Default URI. Change this if you changed your drone's address!
# Format: radio://<interface>/<channel>/<datarate>/<address>
URI = 'radio://0/80/2M/E7E7E7E7E7'

def simple_connect():
    print(f"Attempting to connect to {URI}...")
    print("Connected! Radio is working.")
    time.sleep(3)
    print("Disconnecting...")

if __name__ == '__main__':
    # Initialize the low-level drivers
    cflib.crtp.init_drivers()

    try:
        with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
            simple_connect()
    except Exception as e:
        print(f"Could not connect: {e}")
        print("Check if:")
        print("1. The drone is ON.")
        print("2. The Crazyradio dongle is plugged in.")
        print("3. The URI matches your drone.")
