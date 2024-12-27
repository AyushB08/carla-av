 
import carla


client = carla.Client('localhost', 2000)
client.set_timeout(10.0)


map_name = 'Town04'  
world = client.load_world(map_name)

print(f"Loaded map: {map_name}")
