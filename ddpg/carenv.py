import random
import time
import numpy as np
import math
import cv2
import gymnasium
from gymnasium import spaces
import carla

SECONDS_PER_EPISODE = 25

N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320

SHOW_PREVIEW = True


class CarEnv(gymnasium.Env):
    SHOW_CAM = SHOW_PREVIEW
    im_width = WIDTH
    im_height = HEIGHT
    front_camera = None
    CAMERA_POS_Z = 1.3
    CAMERA_POS_X = 1.4

    def __init__(self):
        super(CarEnv, self).__init__()
        # Continuous action space (throttle and steer)
        self.action_space = spaces.Box(low=np.array([-1, 0]), high=np.array([1, 1]), dtype=np.float32)
        # Observation space (camera image)
        self.observation_space = spaces.Box(low=0, high=255, shape=(N_CHANNELS, self.im_height, self.im_width), dtype=np.uint8)

        # CARLA client and world
        self.client = carla.Client("localhost", 4000)  # Ensure CARLA server is running on this port
        self.client.set_timeout(5.0)
        print("Loading Town06 map...")
        self.world = self.client.load_world("Town06")  # Load map ONCE during initialization
        print("Town06 map loaded successfully.")

        self.blueprint_library = self.world.get_blueprint_library()

        # Initialize the vehicle blueprint
        self.vehicle_bp = self.blueprint_library.filter("model3")[0]

        self.vehicle = None
        self.sensors = []  # List to store sensors
        self.camera_feeds = [None] * 5  # For 5-camera setup

    def reset(self, seed=None):
        print("Resetting environment...")
        self.cleanup()
        print("Cleanup complete.")
        spawn_points = self.world.get_map().get_spawn_points()
        print(f"Spawn points available: {len(spawn_points)}")
        spawn_point = self.find_valid_spawn_point(spawn_points)
        self.vehicle = self.world.spawn_actor(self.vehicle_bp, spawn_point)
        print("Vehicle spawned.")
        self.attach_sensors()
        print("Sensors attached.")
        self.front_camera = None
        self.collision_detected = False
        self.episode_start = time.time()

        # Wait for the front camera to initialize
        start_time = time.time()
        while self.front_camera is None:
            time.sleep(0.01)
            if time.time() - start_time > 5:
                raise RuntimeError("Camera initialization timeout.")

        print("Environment reset complete.")
        observation = np.transpose(self.front_camera, (2, 0, 1))  # Convert (H, W, C) to (C, H, W)
        return observation, {}

    def find_valid_spawn_point(self, spawn_points):
        for _ in range(10):  # Retry up to 10 times
            spawn_point = random.choice(spawn_points)
            if self.is_valid_spawn_point(spawn_point):
                return spawn_point
        raise RuntimeError("Unable to find a valid spawn point.")

    def is_valid_spawn_point(self, spawn_point):
        for actor in self.world.get_actors():
            if actor.get_location().distance(spawn_point.location) < 5.0:  # Ensure at least 5m clearance
                return False
        return True

    def attach_sensors(self):
        # Attach Collision Sensor
        collision_sensor_bp = self.blueprint_library.find("sensor.other.collision")
        self.collision_sensor = self.world.spawn_actor(collision_sensor_bp, carla.Transform(), attach_to=self.vehicle)
        self.collision_sensor.listen(self.process_collision_event)
        self.sensors.append(self.collision_sensor)

        # Attach 5 Cameras
        camera_positions = [
            carla.Transform(carla.Location(x=1.5, z=1.4)),  # Front
            carla.Transform(carla.Location(x=1.5, y=-1.5, z=1.4), carla.Rotation(yaw=90)),  # Left
            carla.Transform(carla.Location(x=1.5, y=1.5, z=1.4), carla.Rotation(yaw=-90)),  # Right
            carla.Transform(carla.Location(x=-2.5, z=1.4), carla.Rotation(yaw=180)),  # Rear
            carla.Transform(carla.Location(z=3.0), carla.Rotation(pitch=-90)),  # Top-down
        ]

        for i, cam_transform in enumerate(camera_positions):
            camera_bp = self.blueprint_library.find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", f"{self.im_width}")
            camera_bp.set_attribute("image_size_y", f"{self.im_height}")
            camera_bp.set_attribute("fov", "110")
            camera = self.world.spawn_actor(camera_bp, cam_transform, attach_to=self.vehicle)
            camera.listen(lambda data, idx=i: self.process_camera_feed(data, idx))
            self.sensors.append(camera)

    def process_collision_event(self, event):
        self.collision_detected = True

    def process_camera_feed(self, image, idx):
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self.camera_feeds[idx] = array[:, :, :3]  # Store RGB

        if idx == 0:  # Front camera is used for observations
            self.front_camera = self.camera_feeds[0]

    def step(self, action):
        print(f"Step called with action: {action}")
        throttle = (action[0] + 1) / 2  # Normalize throttle
        steer = max(-1.0, min(1.0, float(action[1])))  # Clamp steering

        # Apply control
        self.vehicle.apply_control(
            carla.VehicleControl(
                throttle=throttle,
                steer=steer,
                brake=0.0,
                hand_brake=False,
                reverse=False
            )
        )

        # Vehicle speed and location
        velocity = self.vehicle.get_velocity()
        speed = 3.6 * math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)  # Convert m/s to km/h
        waypoint = self.world.get_map().get_waypoint(self.vehicle.get_location(), project_to_road=True)

        # Reward function
        reward = 0
        target_speed = 80  # Target speed in km/h
        reward += max(0, 1 - abs(speed - target_speed) / target_speed) * 2  # Speed reward
        reward += max(0, 1 - self.vehicle.get_location().distance(waypoint.transform.location) / 3.5) * 3  # Lane reward
        reward += -100 if self.collision_detected else 0  # Collision penalty

        # Observation
        observation = np.transpose(self.front_camera, (2, 0, 1))
        terminated = False
        truncated = False
        info = {}

        print(f"Step completed. Speed: {speed:.2f} km/h, Reward: {reward}")
        return observation, reward, terminated, truncated, info

    def cleanup(self):
        # Destroy the vehicle
        if hasattr(self, 'vehicle') and self.vehicle is not None:
            if self.vehicle.is_alive:
                self.vehicle.destroy()
            self.vehicle = None

        # Destroy all sensors
        if hasattr(self, 'sensors') and self.sensors:
            for sensor in self.sensors:
                if sensor.is_alive:
                    sensor.destroy()
            self.sensors = []

    def __del__(self):
        self.cleanup()
