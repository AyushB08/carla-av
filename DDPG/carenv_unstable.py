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
        # Continuous action space (throttle and steering)
        self.action_space = spaces.Box(np.array([-1, 0]), np.array([1, 1]), dtype=np.float32)

        # Observation space (image from front camera)
        self.observation_space = spaces.Box(0, 255, (HEIGHT, WIDTH, N_CHANNELS), dtype=np.uint8)

        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        self.vehicle = None
        self.camera = None

    def reset(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()

        # Spawn the vehicle
        blueprint = self.blueprint_library.filter('vehicle.*')[0]
        spawn_point = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = self.world.spawn_actor(blueprint, spawn_point)

        # Attach a camera to the vehicle
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f'{self.im_width}')
        camera_bp.set_attribute('image_size_y', f'{self.im_height}')
        camera_bp.set_attribute('fov', '110')
        camera_transform = carla.Transform(carla.Location(x=self.CAMERA_POS_X, z=self.CAMERA_POS_Z))
        self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.vehicle)

        self.camera.listen(lambda image: self.process_image(image))

        self.front_camera = None
        while self.front_camera is None:
            time.sleep(0.01)

        return self.front_camera

    def process_image(self, image):
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((self.im_height, self.im_width, N_CHANNELS))
        self.front_camera = array

    def step(self, action):
        throttle, steer = action
        control = carla.VehicleControl(throttle=throttle, steer=steer)
        self.vehicle.apply_control(control)

        # Simulation step
        self.world.tick()

        # Get current camera image as observation
        observation = self.front_camera

        # Define a robust reward function
        reward = 0.0

        # Penalize deviation from lane center
        lane_invasion_sensors = self.vehicle.get_world().get_actors().filter('sensor.other.lane_invasion')
        for sensor in lane_invasion_sensors:
            if sensor.get_event_log():
                reward -= 10.0  # Penalty for lane invasion

        # Penalize collisions
        collision_sensors = self.vehicle.get_world().get_actors().filter('sensor.other.collision')
        for sensor in collision_sensors:
            if sensor.get_event_log():
                reward -= 20.0  # Penalty for collision

        # Reward for forward progress
        velocity = self.vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        reward += speed * 0.1  # Reward proportional to speed

        # Reward for staying in the lane
        if not any(sensor.get_event_log() for sensor in lane_invasion_sensors):
            reward += 5.0  # Bonus for staying in the lane

        # Reward for a successful lane change
        if self._is_lane_change_required() and self._performed_lane_change():
            reward += 20.0  # Significant bonus for lane change when needed

        # Penalize staying stationary
        if speed < 0.1:
            reward -= 5.0

        # Define termination condition
        done = False
        if speed < 0.1 and reward < -50:  # Example condition for stopping
            done = True

        return observation, reward, done, {}

    def _is_lane_change_required(self):
        """Determine if a lane change is required based on environment conditions."""
        # Placeholder logic: return True if a lane change is needed
        return random.choice([True, False])

    def _performed_lane_change(self):
        """Check if the vehicle has successfully changed lanes."""
        # Placeholder logic: return True if a lane change was performed
        return random.choice([True, False])

    def render(self, mode='human'):
        if mode == 'human' and self.SHOW_CAM and self.front_camera is not None:
            cv2.imshow("Front Camera", self.front_camera)
            cv2.waitKey(1)

    def close(self):
        if self.vehicle:
            self.vehicle.destroy()
        if self.camera:
            self.camera.destroy()
        cv2.destroyAllWindows()

    def spawn_vehicles(self, location_type="highway_exit", num_vehicles=5):
        """
        Spawn vehicles at specified locations, such as highway exits or adjacent lanes.

        Args:
            location_type (str): Type of location to spawn vehicles ("highway_exit" or "adjacent_lane").
            num_vehicles (int): Number of vehicles to spawn.
        """
        spawn_points = []

        if location_type == "highway_exit":
            # Define highway exit spawn points (example coordinates)
            spawn_points = [carla.Transform(carla.Location(x=200, y=150, z=0.5)),
                            carla.Transform(carla.Location(x=210, y=155, z=0.5))]
        elif location_type == "adjacent_lane":
            # Define adjacent lane spawn points (example coordinates)
            spawn_points = [carla.Transform(carla.Location(x=100, y=50, z=0.5)),
                            carla.Transform(carla.Location(x=105, y=55, z=0.5))]

        vehicle_blueprints = self.blueprint_library.filter('vehicle.*')

        for _ in range(num_vehicles):
            if spawn_points:
                spawn_point = random.choice(spawn_points)
                vehicle_blueprint = random.choice(vehicle_blueprints)
                self.world.try_spawn_actor(vehicle_blueprint, spawn_point)

