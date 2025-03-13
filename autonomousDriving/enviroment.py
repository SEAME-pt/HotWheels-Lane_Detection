import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import subprocess
import tempfile
import random # for random starting position of the car
import time # for track when the car is stuck on a loop, by chasing its tail or stuck on a wall
import numpy as np # for the image processing
import math # for car speed
import cv2 # for image processing
import gymnasium
import tensorflow as tf # for the model
from gymnasium import spaces # for the action space
import carla # for the carla simulator
from tensorflow.keras.models import load_model # for the model
import logging

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # Suppress TensorFlow info and warning logs


SECONDS_PER_EPISODE = 15

N_CHANNELS = 3
HEIGHT = 240
WIDTH = 320

SPIN = 10 #angle of random spin

HEIGHT_REQUIRED_PORTION = 0.5 #bottom share, e.g. 0.1 is take lowest 10% of rows
WIDTH_REQUIRED_PORTION = 0.9

SHOW_PREVIEW = False

SEED = 123

class CarEnv(gymnasium.Env):
	SHOW_CAM = SHOW_PREVIEW
	STEER_AMT = 1.0
	im_width = WIDTH
	im_height = HEIGHT
	front_camera = None
	CAMERA_POS_Z = 1.3
	CAMERA_POS_X = 1.4
	PREFERRED_SPEED = 30
	SPEED_THRESHOLD = 2

	def __init__(self):
		super(CarEnv, self).__init__()
		self.action_space = spaces.MultiDiscrete([9])
		self.height_from = int(HEIGHT * (1 - HEIGHT_REQUIRED_PORTION))
		self.width_from = int((WIDTH - WIDTH * WIDTH_REQUIRED_PORTION) / 2)
		self.width_to = self.width_from + int(WIDTH_REQUIRED_PORTION * WIDTH)
		self.new_height = HEIGHT - self.height_from
		self.new_width = self.width_to - self.width_from
		self.image_from_CNN = None

		self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7, 18, 8), dtype=np.float32)
		
		self.client = carla.Client('localhost', 2000)
		self.client.set_timeout(4.0)
		self.world = self.client.get_world()
		#self.client.load_world('Town04') # select an specific map
		self.settings = self.world.get_settings()
		self.settings.no_rendering_mode = not self.SHOW_CAM
		self.world.apply_settings(self.settings)

		self.blueprint_library = self.world.get_blueprint_library()
		self.model_3 = self.blueprint_library.filter('model3')[0]
		self.cnn_model = load_model('lane_detector.keras', compile=False)
		self.cnn_model.compile()
		if self.SHOW_CAM:
			self.spectator = self.world.get_spectator()

	def clean_up(self):
		for sensor in self.world.get_actors().filter('*sensor*'):
			sensor.destroy()
		for actor in self.world.get_actors().filter('*vehicle*'):
			actor.destroy()
		cv2.destroyAllWindows()

	def maintain_speed(self,s):
			''' 
			this is a very simple function to maintan desired speed
			s arg is actual current speed
			'''
			if s >= self.PREFERRED_SPEED:
				return 0
			elif s < self.PREFERRED_SPEED - self.SPEED_THRESHOLD:
				return 0.7 # think of it as % of "full gas"
			else:
				return 0.3 # tweak this if the car is way over or under preferred speed 
			
	def apply_cnn(self, im):
		try:
			img = cv2.resize(im, (256, 256))  # Resize to (256, 256)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
			img = np.expand_dims(img, axis=-1)  # Add channel dimension
			img = np.float32(img) / 255.0  # Normalize to [0, 1]
			img = np.expand_dims(img, axis=0)  # Add batch dimension
			cnn_applied = self.cnn_model(img, training=False)
			cnn_applied = np.squeeze(cnn_applied)
			return cnn_applied
		except tf.errors.InvalidArgumentError as e:
			logging.error(f"Error applying CNN: {e}")
			return np.zeros((256, 256))  # Return a default value or handle the error as needed

	def step(self, action):
		trans = self.vehicle.get_transform()
		if self.SHOW_CAM:
			self.spectator.set_transform(carla.Transform(trans.location + carla.Location(z=20),carla.Rotation(yaw =-180, pitch=-90)))

		self.step_counter +=1
		steer = action[0]
		
		# map steering actions
		if steer ==0:
			steer = - 0.9
		elif steer ==1:
			steer = -0.25
		elif steer ==2:
			steer = -0.1
		elif steer ==3:
			steer = -0.05
		elif steer ==4:
			steer = 0.0 
		elif steer ==5:
			steer = 0.05
		elif steer ==6:
			steer = 0.1
		elif steer ==7:
			steer = 0.25
		elif steer ==8:
			steer = 0.9
		
		# optional - print steer and throttle every 50 steps
		if self.step_counter % 50 == 0:
			print('steer input from model:',steer)
		
		v = self.vehicle.get_velocity()
		kmh = int(3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2))
		estimated_throttle = self.maintain_speed(kmh)
		# map throttle and apply steer and throttle	
		self.vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer, brake = 0.0))


		distance_travelled = self.initial_location.distance(self.vehicle.get_location())

		# storing camera to return at the end in case the clean-up function destroys it
		cam = self.front_camera
		# showing image
		if self.SHOW_CAM:
			cv2.imshow('Sem Camera', cam)
			cv2.waitKey(1)

		# track steering lock duration to prevent "chasing its tail"
		lock_duration = 0
		if self.steering_lock == False:
			if steer<-0.6 or steer>0.6:
				self.steering_lock = True
				self.steering_lock_start = time.time()
		else:
			if steer<-0.6 or steer>0.6:
				lock_duration = time.time() - self.steering_lock_start
		
		# start defining reward from each step
		reward = 0
		done = False
		#punish for collision
		if len(self.collision_hist) != 0:
			done = True
			reward = reward - 300
			self.cleanup()
		if len(self.lane_invade_hist) != 0:
			done = True
			reward = reward - 300
			self.cleanup()
		# punish for steer lock up
		if lock_duration>3:
			reward = reward - 150
			done = True
			self.cleanup()
		elif lock_duration > 1:
			reward = reward - 20
		#reward for acceleration
		#if kmh < 10:
		#	reward = reward - 3
		#elif kmh <15:
		#	reward = reward -1
		#elif kmh>40:
		#	reward = reward - 10 #punish for going to fast
		#else:
		#	reward = reward + 1
		# reward for making distance
		if distance_travelled<30:
			reward = reward - 1
		elif distance_travelled<50:
			reward =  reward + 1
		else:
			reward = reward + 2
		# check for episode duration
		if self.episode_start + SECONDS_PER_EPISODE < time.time():
			done = True
			self.cleanup()
		self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:,self.width_from:self.width_to])

		return self.image_for_CNN, reward, done, done,{}	#curly brackets - empty dictionary required by SB3 format

	def reset(self, seed=SEED):
		self.collision_hist = []
		self.lane_invade_hist = []
		self.actor_list = []
		self.transform = random.choice(self.world.get_map().get_spawn_points())
		
		self.vehicle = None
		while self.vehicle is None:
			try:
        # connect
				self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
			except:
				pass
		self.actor_list.append(self.vehicle)



		self.initial_location = self.vehicle.get_location()
		self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
		self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
		self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
		self.sem_cam.set_attribute("fov", f"90")
		
		camera_init_trans = carla.Transform(carla.Location(z=self.CAMERA_POS_Z,x=self.CAMERA_POS_X))
		self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.sensor)
		self.sensor.listen(lambda data: self.process_img(data))

		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		time.sleep(2)
		
		# now apply random yaw so the RL does not guess to go straight
		angle_adj = random.randrange(-SPIN, SPIN, 1)
		trans = self.vehicle.get_transform()
		trans.rotation.yaw = trans.rotation.yaw + angle_adj
		self.vehicle.set_transform(trans)
		
		
		colsensor = self.blueprint_library.find("sensor.other.collision")
		self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.colsensor)
		self.colsensor.listen(lambda event: self.collision_data(event))

		lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
		self.lanesensor = self.world.spawn_actor(lanesensor, camera_init_trans, attach_to=self.vehicle)
		self.actor_list.append(self.lanesensor)
		self.lanesensor.listen(lambda event: self.lane_data(event))

		while self.front_camera is None:
			time.sleep(0.01)
		
		self.episode_start = time.time()
		self.steering_lock = False
		self.steering_lock_start = None # this is to count time in steering lock and start penalising for long time in steering lock
		self.step_counter = 0
		self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
		self.image_for_CNN = self.apply_cnn(self.front_camera[self.height_from:,self.width_from:self.width_to])
		return (self.image_for_CNN,{})

	def process_img(self, image):
		image.convert(carla.ColorConverter.CityScapesPalette)
		i = np.array(image.raw_data)
		i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3] # this is to ignore the 4th Alpha channel - up to 3
		self.front_camera = i

	def collision_data(self, event):
		self.collision_hist.append(event)
	def lane_data(self, event):
		self.lane_invade_hist.append(event)

def run_carla():
    env = os.environ.copy()
    env["DISPLAY"] = os.getenv("DISPLAY")
    with tempfile.TemporaryDirectory() as temp_runtime_dir:
        env["XDG_RUNTIME_DIR"] = temp_runtime_dir

    command = [
        "sudo", "docker", "run", "--privileged", "--gpus", "all", "--net=host",
        "-e", "DISPLAY=:0",
        "carlasim/carla:0.9.15", "/bin/bash", "./CarlaUE4.sh"
    ]

    logging.info("Starting CARLA...")
    try:
        carla_process = subprocess.Popen(command)
        logging.info("CARLA process started.")
        time.sleep(10)

        # Check if CARLA server is running
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        try:
            client.get_world()
            logging.info("CARLA server is running.")
        except:
            logging.error("CARLA server is not running. Exiting.")
            carla_process.terminate()
            return

        # Initialize the CarEnv environment
        env = CarEnv()
        logging.info("Environment initialized.")

        # Run episodes
        for episode in range(10):  # Adjust the number of episodes as needed
            logging.info(f"Starting episode {episode + 1}")
            obs = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = env.action_space.sample()  # Replace with your model's action
                logging.info(f"Action taken: {action}")
                obs, reward, done, _, _ = env.step(action)
                logging.info(f"Observation: {obs}, Reward: {reward}, Done: {done}")
                total_reward += reward

            logging.info(f"Episode {episode + 1} finished with total reward: {total_reward}")
            env.clean_up()
            logging.info(f"Environment cleaned up after episode {episode + 1}")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error running CARLA: {e}")
    except KeyboardInterrupt:
        logging.info("CARLA interrupted by user.")
    finally:
        env.clean_up()
        carla_process.terminate()
        logging.info("CARLA process terminated.")

# Call run_carla to start the simulation
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_carla()
