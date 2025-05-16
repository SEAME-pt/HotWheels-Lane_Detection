import carla
import random

client = carla.Client('localhost', 2000)

maps = client.get_available_maps()
print("Available maps:")
for map_name in maps:
	print(map_name)

world = client.get_world()
print("#################################################################")


print("Available blueprints:")
blueprints = [bp for bp in world.get_blueprint_library().filter('*')]
for blueprint in blueprints:
   print(blueprint.id)
   for attr in blueprint:
       print('  - {}'.format(attr))