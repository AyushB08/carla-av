import carla
import time


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)


world = client.get_world()


spectator = world.get_spectator()

spectator_location = spectator.get_location()


map = world.get_map()

closest_waypoint = map.get_waypoint(spectator_location)


print(f"Closest waypoint: {closest_waypoint.transform.location}")

# visualizing waypoints in adjacent lanes
def visualize_other_lanes(current_waypoint, num_lanes=4):
    e
    blueprint = world.get_blueprint_library().find('static.prop.trafficcone01')
    if not blueprint:
        blueprint = world.get_blueprint_library().find('static.prop.box01')

    if blueprint:
       
        actor = world.spawn_actor(blueprint, current_waypoint.transform)
        time.sleep(1)  

      
        for i in range(1, num_lanes):
        
            next_waypoint = current_waypoint.next(2.0)
            if next_waypoint:
             
                next_right_waypoint = next_waypoint[0].get_right_lane()
                if next_right_waypoint:
             
                    actor = world.spawn_actor(blueprint, next_right_waypoint.transform)
                    time.sleep(1) 

                    current_waypoint = next_right_waypoint 

        # removes markers if needed
        time.sleep(10) 
        actor.destroy()  
    else:
        print("No suitable blueprint found.")

visualize_other_lanes(closest_waypoint)
