import random
import time
import numpy as np
import math
import cv2
import gymnasium
from gymnasium import spaces
import carla

SECONDS_PER_EPISODE = 30
N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320
FIXED_DELTA_SECONDS = 0.05
MAX_SUBSTEPS = 10
MAX_SUBSTEP_DELTA_TIME = 0.01
SHOW_PREVIEW = True
TARGET_EXIT_DISTANCE = 100

class CarEnv(gymnasium.Env):
    SHOW_CAM = SHOW_PREVIEW
    STEER_AMT = 1.0
    im_width = WIDTH
    im_height = HEIGHT
    front_camera = None
    CAMERA_POS_Z = 1.3
    CAMERA_POS_X = 1.4

    def __init__(self):
        print("Initializing CarEnv...")
        super(CarEnv, self).__init__()
        self.action_space = spaces.MultiDiscrete([9,4])
        self.observation_space = spaces.Box(low=0.0, high=1.0,
                                         shape=(HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32)

        print("Connecting to CARLA server...")
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)

        try:
            print("Attempting to load Town04...")
            self.world = self.client.load_world('Town04')
            print("Successfully loaded Town04")
        except Exception as e:
            print(f"Failed to load Town04: {str(e)}")
            print("Using current map instead")
            self.world = self.client.get_world()

        print("Configuring world settings...")
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.world.apply_settings(self.settings)

        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

      
        self.spawn_point = carla.Transform(
            carla.Location(x=120.901894, y=17.373587, z=12),
            carla.Rotation(yaw=180) 
        )

       
        self.exit_points = [
            carla.Location(x=-100.877724, y=-4.664539, z=12)
        ]

    def cleanup(self):
        print("Cleaning up environment...")
        try:
            sensors = self.world.get_actors().filter('*sensor*')
            for sensor in sensors:
                sensor.destroy()

            vehicles = self.world.get_actors().filter('*vehicle*')
            for vehicle in vehicles:
                vehicle.destroy()

            self.world.tick()
            time.sleep(0.5)

            cv2.destroyAllWindows()
            print("Cleanup completed successfully")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")

    def _setup_sensors(self):
        if not self.vehicle:
            raise RuntimeError("Cannot setup sensors: Vehicle not initialized")

        print("Setting up semantic camera...")
        self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", f"90")

        camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z, x=self.CAMERA_POS_X))

        self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
        if not self.sensor:
            raise RuntimeError("Failed to spawn camera sensor")
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))

        print("Setting up collision sensor...")
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
        if not self.colsensor:
            raise RuntimeError("Failed to spawn collision sensor")
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))

        print("Sensors setup complete")

    def process_img(self, image):
        try:
            image.convert(carla.ColorConverter.CityScapesPalette)
            i = np.array(image.raw_data)
            i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3]
            self.front_camera = i
        except Exception as e:
            print(f"Error processing image: {str(e)}")

    def collision_data(self, event):
        print(f"Collision detected with {event.other_actor}")
        self.collision_hist.append(event)

    def _get_distance_to_exit(self):
        """Calculate distance to nearest exit point"""
        vehicle_location = self.vehicle.get_location()
        exit_distances = [exit_point.distance(vehicle_location)
                         for exit_point in self.exit_points]
        return min(exit_distances)

    def is_within_lane(self):
        """Check if the vehicle is within the lane."""
        vehicle_location = self.vehicle.get_location()
        waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True)

        if waypoint:
          
            if waypoint.lane_type == carla.LaneType.Driving:
             
                distance_to_lane_center = vehicle_location.distance(waypoint.transform.location)
           
                if distance_to_lane_center < 1.5:  
                    return True
        return False

    def _get_direction_to_exit(self):
        """Calculate the relative direction to the exit"""
        vehicle_transform = self.vehicle.get_transform()
        vehicle_location = vehicle_transform.location
        vehicle_forward = vehicle_transform.get_forward_vector()
        
     
        exit_location = self.exit_points[0] 
        to_exit = carla.Vector3D(
            x=exit_location.x - vehicle_location.x,
            y=exit_location.y - vehicle_location.y,
            z=0
        )
        
     
        to_exit_norm = math.sqrt(to_exit.x**2 + to_exit.y**2)
        if to_exit_norm > 0:
            to_exit.x /= to_exit_norm
            to_exit.y /= to_exit_norm
        
      
        dot = vehicle_forward.x * to_exit.x + vehicle_forward.y * to_exit.y
        cross = vehicle_forward.x * to_exit.y - vehicle_forward.y * to_exit.x
        
       
        angle = math.atan2(cross, dot)
        return math.degrees(angle)

    def _calculate_reward(self, kmh, distance_to_exit, lane_change_progress, steer):
        """Calculate reward with high priority on exit progress, lane keeping, and safety"""
        reward = 0
        vehicle_location = self.vehicle.get_location()
        waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True)
        
       
        angle_to_exit = self._get_direction_to_exit()

      
        if len(self.collision_hist) > 0:
            return -500

        if waypoint is None or vehicle_location.distance(waypoint.transform.location) > 3.0:
            return -400 

      
        direction_reward = -abs(angle_to_exit) * 0.2 
        if abs(angle_to_exit) < 30:  
            direction_reward += 15
        reward += direction_reward

      
        if (angle_to_exit > 10 and steer < 0) or (angle_to_exit < -10 and steer > 0):
            reward -= 10  

       
        progress = (self.initial_exit_distance - distance_to_exit) / self.initial_exit_distance
        progress_reward = progress * 50
        
        
        if distance_to_exit < 50:
            progress_reward += (50 - distance_to_exit)
        if distance_to_exit < 25:
            progress_reward += (25 - distance_to_exit) * 2
        if distance_to_exit < 10:
            progress_reward += (10 - distance_to_exit) * 4
        
        reward += progress_reward

      
        if self.is_within_lane():
            reward += 20
        else:
            lane_deviation = vehicle_location.distance(waypoint.transform.location)
            lane_penalty = -lane_deviation * 15
            reward += lane_penalty

      
        target_speed = self._get_target_speed(distance_to_exit)
        speed_diff = abs(kmh - target_speed)
        
        if speed_diff > 30:
            reward -= (speed_diff - 30) * 0.5

      
        steering_smoothness = -abs(steer) * 0.5
        reward += steering_smoothness

       
        reward = max(min(reward, 100), -500)

        return reward

    def _get_target_speed(self, distance_to_exit):
        """Calculate target speed based on distance to exit"""
        if distance_to_exit > 50:
            return 90  
        elif distance_to_exit > 30:
            return 70  
        elif distance_to_exit > 15:
            return 50  
        else:
            return 30  

    def _map_steering_action(self, action):
        """Maps discrete steering actions (0-8) to continuous steering values with reduced sensitivity"""
    
        return (action - 4) / 4.0 * 0.4 

    def reset(self, seed=None):
        print("\nResetting environment...")
        self.cleanup()
        self.collision_hist = []
        self.actor_list = []

        try:
            self.vehicle = self.world.spawn_actor(self.model_3, self.spawn_point)
            self.actor_list.append(self.vehicle)

            self.world.tick()
            time.sleep(0.1)

            self._setup_sensors()

            camera_init_timeout = 2.0
            start_time = time.time()
            while self.front_camera is None:
                self.world.tick()
                if time.time() - start_time > camera_init_timeout:
                    raise TimeoutError("Camera initialization timeout")
                time.sleep(0.1)

            self.episode_start = time.time()
            self.step_counter = 0
            self.initial_exit_distance = self._get_distance_to_exit()

            print("Reset complete!")
            return self.front_camera/255.0, {}

        except Exception as e:
            print(f"Reset failed: {str(e)}")
            self.cleanup()
            raise RuntimeError("Failed to reset environment")

    def step(self, action):
        self.step_counter += 1

       
        angle_to_exit = self._get_direction_to_exit()
        
       
        target_steer = self._map_steering_action(action[0])
        current_steer = self.vehicle.get_control().steer
        alpha = 0.15 
        steer = current_steer + alpha * (target_steer - current_steer)

  
        if abs(angle_to_exit) > 10:  
           
            correction = math.copysign(0.05, angle_to_exit)
            steer += correction
        
     
        steer = max(min(steer, 1.0), -1.0)

       
        throttle_map = {
            0: (0.0, 1.0),   
            1: (0.3, 0.0),   
            2: (0.6, 0.0),   
            3: (1.0, 0.0)     
        }
        throttle_val, brake_val = throttle_map[action[1]]

     
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
        distance_to_exit = self._get_distance_to_exit()
        target_speed = self._get_target_speed(distance_to_exit)

      
        if action[1] > 0: 
            speed_diff = target_speed - kmh
            throttle_val = min(1.0, max(0.2, throttle_val + speed_diff * 0.01))

        self.vehicle.apply_control(carla.VehicleControl(
            throttle=throttle_val,
            steer=steer,
            brake=brake_val
        ))

        self.world.tick()

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        distance_to_exit = self._get_distance_to_exit()

        if self.front_camera is None:
            print("Warning: No camera feed available")
            observation = np.zeros((HEIGHT, WIDTH, N_CHANNELS), dtype=np.float32)
        else:
            observation = self.front_camera/255.0

        if self.SHOW_CAM:
            cv2.imshow("Camera Feed", self.front_camera)
            cv2.waitKey(1)

        reward = self._calculate_reward(kmh, distance_to_exit, 0, steer)

        done = self._check_episode_end(distance_to_exit)

        if self.step_counter % 50 == 0:
            print(f"Step {self.step_counter}: Speed={kmh}km/h, Distance to exit={distance_to_exit:.1f}m, Reward={reward:.2f}")

        return observation, reward, done, done, {}

    def _check_episode_end(self, distance_to_exit):
        """Checks if the episode should end"""
        vehicle_location = self.vehicle.get_location()
        waypoint = self.world.get_map().get_waypoint(vehicle_location, project_to_road=True)

        if self.step_counter % 50 == 0:
            print(f"Vehicle position - X: {vehicle_location.x:.2f}, Y: {vehicle_location.y:.2f}, Z: {vehicle_location.z:.2f}")
            if waypoint:
                print(f"Nearest waypoint - X: {waypoint.transform.location.x:.2f}, Y: {waypoint.transform.location.y:.2f}, Z: {waypoint.transform.location.z:.2f}")
                print(f"Distance to waypoint: {vehicle_location.distance(waypoint.transform.location):.2f}")

        if len(self.collision_hist) > 0:
            print("Episode ended due to collision")
            return True

        if distance_to_exit < 5.0:
            print("Successfully reached exit!")
            return True

        if time.time() - self.episode_start > SECONDS_PER_EPISODE:
            print("Episode ended due to timeout")
            return True

        if waypoint is None or vehicle_location.distance(waypoint.transform.location) > 3.0:
            print(f"Episode ended due to vehicle leaving road (Distance to nearest waypoint: {vehicle_location.distance(waypoint.transform.location) if waypoint else 'No waypoint found'})")
            return True

        return False
