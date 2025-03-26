import subprocess
import os
import carla
import time
import pygame
import tempfile
#from traffic import spawn_traffic, spawn_pedestrians
#from process_image import sharpen_image, process_and_visualize, process_image
from process_image import KerasLaneDetector

# Initialize Pygame for keyboard input
pygame.init()
screen = pygame.display.set_mode((400, 300))
pygame.display.set_caption("CARLA Controls")

# Define the camera mode to track the current camera
camera_mode = "third_person"  # 'third_person' or 'inside_camera'

# Global camera reference
camera = None
capture_flag = False
captured_image = None

# Desired size for displaying the image
desired_width = 400
desired_height = 300

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

    try:
        print("Starting CARLA...")
        subprocess.Popen(command)
        time.sleep(10)
        spawn_vehicle_and_camera()
    except subprocess.CalledProcessError as e:
        print(f"Error running CARLA: {e}")
    except KeyboardInterrupt:
        print("CARLA interrupted by user.")

def spawn_vehicle_and_camera():
    global camera
    try:
        client = carla.Client("localhost", 2000)
        client.set_timeout(10.0)

        world = client.load_world('Town05')
        blueprint_library = world.get_blueprint_library()

        # Choose a vehicle blueprint
        vehicle_bp = blueprint_library.find("vehicle.lincoln.mkz_2020")
        spawn_points = world.get_map().get_spawn_points()
        spawn_point = spawn_points[0] if spawn_points else carla.Transform()

        vehicle = world.spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
            print("Failed to spawn vehicle!")
            return

        print(f"Vehicle spawned: {vehicle.id}")
        vehicle.set_autopilot(True)

        # 📷 Attach a camera sensor
        camera_bp = blueprint_library.find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", "512")
        camera_bp.set_attribute("image_size_y", "256")
        camera_bp.set_attribute("fov", "100")

        # Camera position: Slightly above and behind the car
        camera_transform = carla.Transform(carla.Location(x=0, z=2))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)

        # Start the control loop
        control_vehicle(vehicle, world.get_spectator())

    except Exception as e:
        print(f"Error spawning vehicle or setting camera: {e}")

def capture_image():
    global captured_image
    """Continuously capture and process images from the camera."""
    if not camera:
        print("No camera found!")
        return

    detector = KerasLaneDetector('lane_detector_final.keras')

    # Camera listener: Process image on every frame
    def process_frame(image):
        global captured_image
        # Convert CARLA image to OpenCV format and save it
        image_array = detector.process_image(image)  # Process and update segmentation result
        captured_image = detector.process_and_visualize(image_array)  # Get the processed surface

    camera.listen(process_frame)  # Start continuous listening

def control_vehicle(vehicle, spectator):
    global camera_mode, camera, captured_image
    """Control the car using arrow keys."""
    clock = pygame.time.Clock()

    # Vehicle control variables
    throttle = 0.0
    steer = 0.0
    brake = 0.0

    if camera:
        capture_image()

    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    print("\nExiting...")
                    return

            keys = pygame.key.get_pressed()

            # Press 'C' to switch between the cameras
            if keys[pygame.K_c]:
                if camera_mode == "third_person":
                    camera_mode = "inside_camera"
                else:
                    camera_mode = "third_person"

            # Press 'P' to capture a frame
            if keys[pygame.K_p] and camera:
                capture_image()

            # 🚀 **Throttle & Brake**
            if keys[pygame.K_UP]:
                throttle = min(throttle + 0.05, 1.0)  # Accelerate smoothly
                brake = 0.0
            elif keys[pygame.K_DOWN]:
                brake = min(brake + 0.1, 1.0)  # Apply brake
                throttle = 0.0
            else:
                throttle *= 0.9  # Slowly reduce speed when no key is pressed
                brake *= 0.9

            # 🎮 **Steering**
            if keys[pygame.K_LEFT]:
                steer = max(steer - 0.05, -1.0)  # Turn left
            elif keys[pygame.K_RIGHT]:
                steer = min(steer + 0.05, 1.0)  # Turn right
            else:
                steer *= 0.8  # Slowly straighten the wheels

            # Apply control to the car
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))

            # Change camera mode
            if camera_mode == "third_person":
                transform = vehicle.get_transform()

                # Fixed offset behind and above the car
                camera_offset = carla.Location(x=-6, z=2)

                # Move the spectator accordingly
                camera_location = transform.location + transform.get_forward_vector().make_unit_vector() * camera_offset.x
                camera_location.z = transform.location.z + camera_offset.z
                camera_rotation = carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)

                spectator.set_transform(carla.Transform(camera_location, camera_rotation))

            elif camera_mode == "inside_camera":
                transform = vehicle.get_transform()

                # Inside view: Slightly above the car
                camera_offset = carla.Location(x=0, z=2)

                # Move the spectator accordingly
                camera_location = transform.location + transform.get_forward_vector().make_unit_vector() * camera_offset.x
                camera_location.z = transform.location.z + camera_offset.z
                camera_rotation = carla.Rotation(pitch=-15, yaw=transform.rotation.yaw)

                spectator.set_transform(carla.Transform(camera_location, camera_rotation))

            # Display the captured image when it's available
            if captured_image:
                screen.blit(captured_image, (0, 0))  # Draw the image on screen
                pygame.display.flip()  # Update the display

            clock.tick(60)  # Limit to 60 FPS for smoother control

    except KeyboardInterrupt:
        print("\nExiting vehicle control.")

if __name__ == "__main__":
    run_carla()
