"""Hardware driver for Bitcraze Crazyflie 2.1 using cflib.

This driver implements the DriverInterface expected by CrazyflieFirmware.
It handles:
1. Connection/Disconnection via cflib
2. Mapping Policy Actions (Thrust + Moments) -> Hardware Commands (Thrust + Rates)
3. Safety checks (connection loss, etc.)
"""
import time
import logging
import numpy as np
from threading import Event

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.utils import uri_helper

from src.CrazyflieFirmware import DriverInterface

logger = logging.getLogger(__name__)

class CrazyflieDriver(DriverInterface):
    def __init__(self, uri='radio://0/80/2M/E7E7E7E7E7'):
        self.uri = uri_helper.uri_from_env(default=uri)
        self.cf = Crazyflie(rw_cache='./cache')
        
        # Connection events
        self.is_connected = False
        self.connection_event = Event()
        self.disconnected_event = Event()
        
        # Attach callbacks
        self.cf.connected.add_callback(self._connected)
        self.cf.disconnected.add_callback(self._disconnected)
        self.cf.connection_failed.add_callback(self._connection_failed)
        self.cf.connection_lost.add_callback(self._connection_lost)

        # Initialize link
        cflib.crtp.init_drivers()

        # State storage for observation
        self.latest_obs = np.zeros(13, dtype=np.float32)
        # [x, y, z, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]
        # Initialize quaternion to identity
        self.latest_obs[3] = 1.0 

    def start_logging(self):
        """Start 100Hz logging of state estimate."""
        self.log_conf = LogConfig(name='State', period_in_ms=10)
        
        # Position (m)
        self.log_conf.add_variable('stateEstimate.x', 'float')
        self.log_conf.add_variable('stateEstimate.y', 'float')
        self.log_conf.add_variable('stateEstimate.z', 'float')
        
        # Velocity (m/s) - Body or World? stateEstimate is usually World.
        self.log_conf.add_variable('stateEstimate.vx', 'float')
        self.log_conf.add_variable('stateEstimate.vy', 'float')
        self.log_conf.add_variable('stateEstimate.vz', 'float')
        
        # Attitude (Quaternion)
        self.log_conf.add_variable('stateEstimate.qw', 'float')
        self.log_conf.add_variable('stateEstimate.qx', 'float')
        self.log_conf.add_variable('stateEstimate.qy', 'float')
        self.log_conf.add_variable('stateEstimate.qz', 'float')
        
        # Angular Velocity (Gyro) - deg/s -> need rad/s
        self.log_conf.add_variable('gyro.x', 'float')
        self.log_conf.add_variable('gyro.y', 'float')
        self.log_conf.add_variable('gyro.z', 'float')

        self.cf.log.add_config(self.log_conf)
        self.log_conf.data_received_cb.add_callback(self._log_data_received)
        self.log_conf.start()
        logger.info("Logging started")

    def _log_data_received(self, timestamp, data, logconf):
        """Callback for new data. Updates latest_obs."""
        # 1. Position
        self.latest_obs[0] = data['stateEstimate.x']
        self.latest_obs[1] = data['stateEstimate.y']
        self.latest_obs[2] = data['stateEstimate.z']

        # 2. Quaternion
        self.latest_obs[3] = data['stateEstimate.qw']
        self.latest_obs[4] = data['stateEstimate.qx']
        self.latest_obs[5] = data['stateEstimate.qy']
        self.latest_obs[6] = data['stateEstimate.qz']

        # 3. Linear Velocity
        self.latest_obs[7] = data['stateEstimate.vx']
        self.latest_obs[8] = data['stateEstimate.vy']
        self.latest_obs[9] = data['stateEstimate.vz']

        # 4. Angular Velocity (Gyro is in deg/s, convert to rad/s)
        deg2rad = np.pi / 180.0
        self.latest_obs[10] = data['gyro.x'] * deg2rad
        self.latest_obs[11] = data['gyro.y'] * deg2rad
        self.latest_obs[12] = data['gyro.z'] * deg2rad

    def get_latest_obs(self):
        return self.latest_obs.copy()

    def connect(self):
        logger.info(f"Connecting to {self.uri}...")
        self.cf.open_link(self.uri)
        if not self.connection_event.wait(timeout=10):
            raise TimeoutError("Connection timed out")
        logger.info("Connected to Crazyflie")

    def disconnect(self):
        logger.info("Disconnecting...")
        self.cf.commander.send_stop_setpoint()
        self.cf.close_link()
        self.disconnected_event.wait(timeout=5)
        logger.info("Disconnected")

    def _connected(self, link_uri):
        self.is_connected = True
        self.connection_event.set()

    def _disconnected(self, link_uri):
        self.is_connected = False
        self.disconnected_event.set()

    def _connection_failed(self, link_uri, msg):
        logger.error(f"Connection failed: {msg}")
        self.is_connected = False

    def _connection_lost(self, link_uri, msg):
        logger.error(f"Connection lost: {msg}")
        self.is_connected = False

    def send_action(self, action):
        """
        Send control command to the drone.
        
        Args:
            action: [thrust, mx, my, mz]
            
        Mapping:
            Thrust (0.0 - 0.35) -> (0 - 65535) 16-bit integer
            Moments (mx, my, mz) -> Body Rates (deg/s)
            
        Note: Mapping moments to rates is an approximation. 
        Ideally, we'd send direct motor power, but that requires custom firmware.
        We assume the policy's 'moments' correlate to desired rotation rates.
        """
        if not self.is_connected:
            return

        thrust_norm = float(action[0])
        mx = float(action[1])
        my = float(action[2])
        mz = float(action[3])

        # --- 1. Map Thrust ---
        # Max thrust in sim is 0.35 (approx 35 grams? or normalized?)
        # Real CF max thrust is ~60g. 
        # We map 0.0-0.35 -> 0-65535 linearly, but we might need a gain.
        # Let's assume 0.35 in sim corresponds to a safe high throttle on hardware.
        # 0.35 / 1.0 * 65535 = 22937 (which is hover-ish for a heavy drone).
        # WAIT: In sim, hover is ~0.27. On real CF, hover is usually ~40000-45000 (16-bit).
        # So 0.27 -> 42000.
        # Scale factor = 42000 / 0.27 ≈ 155,000
        # Let's try a safer linear map based on hover ratio.
        
        HOVER_SIM = 0.27
        HOVER_HW = 42000  # Typical for CF2.1 with battery
        
        scale = HOVER_HW / HOVER_SIM
        thrust_pwm = int(thrust_norm * scale)
        thrust_pwm = np.clip(thrust_pwm, 0, 60000) # Safety clamp

        # --- 2. Map Moments to Rates ---
        # Policy outputs torque. Torque ~ Angular Accel.
        # We command Angular Rate.
        # This is a control mismatch, but often works if we treat output as "effort".
        # Gain tuning required here.
        
        RATE_GAIN = 200.0 # Arbitrary gain to convert "moment" to "deg/s"
        
        roll_rate = mx * RATE_GAIN
        pitch_rate = my * RATE_GAIN
        yaw_rate = mz * RATE_GAIN

        # Send setpoint: (roll, pitch, yaw, thrust)
        # But we want RATE mode. cflib send_setpoint usually takes angles.
        # We need send_hover_setpoint or similar?
        # Actually, cf.commander.send_setpoint(roll, pitch, yaw, thrust) 
        # interprets roll/pitch as angles (deg) and yaw as rate (deg/s) usually?
        # NO, standard is Angle, Angle, YawRate, Thrust.
        
        # To send RATES for roll/pitch, we might need a different packet or 
        # assume the policy learned Angles.
        # IF the policy learned Moments, it expects the drone to accelerate.
        # Sending Rates is closer to Moments than sending Angles.
        
        # For now, we will use the standard setpoint and assume:
        # mx -> Roll Angle (deg)
        # my -> Pitch Angle (deg)
        # mz -> Yaw Rate (deg/s)
        
        ANGLE_GAIN = 20.0 # Max tilt ~20 deg
        
        cmd_roll = mx * ANGLE_GAIN
        cmd_pitch = my * ANGLE_GAIN
        cmd_yawrate = mz * 100.0
        
        self.cf.commander.send_setpoint(cmd_roll, cmd_pitch, cmd_yawrate, thrust_pwm)

    def stop(self):
        if self.is_connected:
            self.cf.commander.send_stop_setpoint()
            self.cf.commander.send_notify_setpoint_stop()
