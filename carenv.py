 
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

        print("Setting up traffic manager...")
        self.traffic_manager = self.client.get_trafficmanager(8000)
        self.traffic_manager.set_synchronous_mode(True)
        self.traffic_manager.set_random_device_seed(0)

        print("Configuring world settings...")
        self.settings = self.world.get_settings()
        self.settings.synchronous_mode = True
        self.settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        self.settings.max_substeps = MAX_SUBSTEPS
        self.settings.max_substep_delta_time = MAX_SUBSTEP_DELTA_TIME
        self.world.apply_settings(self.settings)
        print(f"World settings applied: Delta={FIXED_DELTA_SECONDS}, "
              f"Substeps={MAX_SUBSTEPS}, Substep Delta={MAX_SUBSTEP_DELTA_TIME}")

        print("Loading blueprints...")
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("model3")[0]

        print("Finding spawn points...")
        self.spawn_points = self.world.get_map().get_spawn_points()
        self.highway_spawns = [p for p in self.spawn_points if self._is_highway_point(p)]
        self.exit_points = self._get_exit_points()

        print(f"Found {len(self.highway_spawns)} highway spawn points and "
              f"{len(self.exit_points)} exit points")

    def _is_highway_point(self, transform):
        waypoint = self.world.get_map().get_waypoint(transform.location)
        is_highway = waypoint.road_id in [36, 37, 38]
        return is_highway and waypoint.lane_type == carla.LaneType.Driving

    def _get_exit_points(self):
        print("Identifying exit points...")
        exit_points = []
        for point in self.spawn_points:
            waypoint = self.world.get_map().get_waypoint(point.location)
            if waypoint.road_id in [41, 42, 43]:
                exit_points.append(point)
        return exit_points

    def _spawn_traffic(self):
        print("Spawning traffic vehicles...")
        traffic_vehicles = []
        vehicle_bps = self.blueprint_library.filter('vehicle.*')

        spawn_attempts = 0
        max_attempts = 10
        desired_vehicles = random.randint(3, 4)

        while len(traffic_vehicles) < desired_vehicles and spawn_attempts < max_attempts:
            spawn_attempts += 1
            try:
                spawn_point = random.choice(self.highway_spawns)
                spawn_point.location.x += random.uniform(-10, 10)
                spawn_point.location.y += random.uniform(-2, 2)

                vehicle_bp = random.choice(vehicle_bps)
                vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)

                if vehicle:
                    print(f"Spawned traffic vehicle: {vehicle_bp.id}")
                    vehicle.set_autopilot(True, self.traffic_manager.get_port())
                    self.traffic_manager.distance_to_leading_vehicle(vehicle, 20.0)
                    self.traffic_manager.vehicle_percentage_speed_difference(vehicle, -10.0)
                    traffic_vehicles.append(vehicle)
            except Exception as e:
                print(f"Failed to spawn traffic vehicle: {str(e)}")

        print(f"Successfully spawned {len(traffic_vehicles)} traffic vehicles")
        return traffic_vehicles

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
    def _find_safe_spawn_point(self):
        """Find a spawn point that's not blocked by other vehicles."""
        print("Finding safe spawn point...")

    
        available_spawns = self.highway_spawns.copy()
        random.shuffle(available_spawns)

        for spawn_point in available_spawns:
 
            spawn_point_modified = carla.Transform(
                spawn_point.location + carla.Location(z=0.5), 
                spawn_point.rotation
            )

          
            if self._is_spawn_point_clear(spawn_point_modified):
                return spawn_point_modified

        return None
    def _is_spawn_point_clear(self, spawn_point, check_radius=5.0):
        """Check if there are any vehicles near the spawn point."""
        vehicles = self.world.get_actors().filter('*vehicle*')

        spawn_loc = spawn_point.location
        for vehicle in vehicles:
            if vehicle.get_location().distance(spawn_loc) < check_radius:
                return False
        return True
    def step(self, action):
        self.step_counter += 1
        if self.step_counter % 50 == 0:
            print(f"Step {self.step_counter}")

      
        steer = self._map_steering_action(action[0])
        self._apply_control(steer, action[1])

        
        self.world.tick()

       
        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))

        if self.step_counter % 50 == 0:
            print(f"Vehicle speed: {kmh} km/h")

      
        distance_to_exit = self._get_distance_to_exit()
        lane_change_progress = self._calculate_lane_change_progress()

   
        cam = self.front_camera
        if self.SHOW_CAM:
            cv2.imshow('Sem Camera', cam)
            cv2.waitKey(1)

      
        reward = self._calculate_reward(kmh, distance_to_exit, lane_change_progress, steer)

        if self.step_counter % 50 == 0:
            print(f"Reward: {reward}, Distance to exit: {distance_to_exit:.1f}m")

       
        done = self._check_episode_end(distance_to_exit)

        return cam/255.0, reward, done, done, {}

    def _map_steering_action(self, steer):
        steer_map = {
            0: -0.9, 1: -0.25, 2: -0.1, 3: -0.05, 4: 0.0,
            5: 0.05, 6: 0.1, 7: 0.25, 8: 0.9
        }
        return steer_map[steer]

    def _apply_control(self, steer, throttle):
        throttle_map = {
            0: (0.0, 1.0),
            1: (0.3, 0.0),
            2: (0.7, 0.0),
            3: (1.0, 0.0)
        }
        throttle_val, brake_val = throttle_map[throttle]
        self.vehicle.apply_control(carla.VehicleControl(throttle=throttle_val, steer=steer, brake=brake_val))

    def _get_distance_to_exit(self):
        vehicle_location = self.vehicle.get_location()
        exit_distances = [exit_point.location.distance(vehicle_location)
                         for exit_point in self.exit_points]
        return min(exit_distances)

    def _calculate_lane_change_progress(self):
        try:
            current_wp = self.world.get_map().get_waypoint(self.vehicle.get_location())
            target_wp = self.target_exit_lane.transform.location
            lateral_distance = abs(current_wp.transform.location.distance(target_wp))
            return max(0, 1 - lateral_distance/5.0)
        except Exception as e:
            print(f"Error calculating lane change progress: {str(e)}")
            return 0

    def _calculate_reward(self, kmh, distance_to_exit, lane_change_progress, steer):
        reward = 0

       
        if 60 < kmh < 90:
            reward += 2
        elif kmh < 40:
            reward -= 2
        elif kmh > 100:
            reward -= 3

     
        reward += lane_change_progress * 3

      
        if distance_to_exit < self.initial_exit_distance:
            reward += 2

       
        if len(self.collision_hist) != 0:
            reward -= 300

        return reward

    def _check_episode_end(self, distance_to_exit):
        done = False

        if len(self.collision_hist) != 0:
            print("Episode ended due to collision")
            done = True
            self.cleanup()
        elif distance_to_exit < 10:
            print("Successfully reached exit!")
            done = True
            self.cleanup()
        elif self.episode_start + SECONDS_PER_EPISODE < time.time():
            print("Episode ended due to timeout")
            done = True
            self.cleanup()
        elif self.vehicle.get_location().z < -0.5:
            print("Episode ended - vehicle fell off road")
            done = True
            self.cleanup()

        return done

    def reset(self, seed=None):
        print("\nResetting environment...")
        self.cleanup()
        self.collision_hist = []
        self.actor_list = []
        self.vehicle = None
        self.front_camera = None

      
        max_reset_attempts = 3
        for reset_attempt in range(max_reset_attempts):
            try:
                print(f"Reset attempt {reset_attempt + 1}/{max_reset_attempts}")

                # Find a safe spawn point
                safe_spawn = self._find_safe_spawn_point()
                if not safe_spawn:
                    print("No safe spawn points found, retrying...")
                    continue

              
                for spawn_attempt in range(5):
                    try:
                        self.vehicle = self.world.spawn_actor(self.model_3, safe_spawn)
                        self.actor_list.append(self.vehicle)
                        print(f"Successfully spawned ego vehicle on attempt {spawn_attempt + 1}")
                        break
                    except Exception as e:
                        print(f"Spawn attempt {spawn_attempt + 1} failed: {str(e)}")
                        if spawn_attempt == 4:
                            raise RuntimeError("Failed to spawn ego vehicle after multiple attempts")
                        time.sleep(0.2)

                if not self.vehicle:
                    continue

                
                self.world.tick()
                time.sleep(0.1)  

               
                try:
                    self._setup_sensors()
                   
                    camera_init_timeout = 2.0 
                    start_time = time.time()
                    while self.front_camera is None:
                        self.world.tick()
                        if time.time() - start_time > camera_init_timeout:
                            raise TimeoutError("Camera initialization timeout")
                        time.sleep(0.1)
                except Exception as sensor_error:
                    print(f"Sensor setup failed: {str(sensor_error)}")
                    self.cleanup()
                    continue

     
                self.traffic_vehicles = self._spawn_traffic()
                self.actor_list.extend(self.traffic_vehicles)

               
                self.episode_start = time.time()
                self.step_counter = 0
                self.initial_exit_distance = self._get_distance_to_exit()

               
                if self.exit_points:
                    self.target_exit_lane = self.world.get_map().get_waypoint(
                        random.choice(self.exit_points).location)
                else:
                    print("Warning: No exit points found")
                    self.target_exit_lane = self.world.get_map().get_waypoint(
                        safe_spawn.location)

                print("Reset complete!")
                return self.front_camera/255.0, {}

            except Exception as e:
                print(f"Reset attempt {reset_attempt + 1} failed completely: {str(e)}")
                self.cleanup()
                time.sleep(1.0)

        raise RuntimeError("Failed to reset environment after multiple attempts")


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
