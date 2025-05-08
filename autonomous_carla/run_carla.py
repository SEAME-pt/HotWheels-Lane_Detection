import subprocess
import os
import sys
import time
import tempfile

# Desired size for displaying the image
desired_width = 400
desired_height = 300

def run_carla():
    env = os.environ.copy()
    env["DISPLAY"] = os.getenv("DISPLAY")
    env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"  # Explicitly set XDG_RUNTIME_DIR
    env["PULSE_SERVER"] = "unix:/run/user/1001/pulse/native"

    command = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--net=host",
        "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
        "-v", "/dev/snd:/dev/snd",  # Pass audio devices
        "-e", f"DISPLAY={env['DISPLAY']}",
        "-e", f"XDG_RUNTIME_DIR={env['XDG_RUNTIME_DIR']}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
        "-e", f"PULSE_SERVER={env['PULSE_SERVER']}",  # Use the correct PulseAudio server
        "-v", "/run/user/1001/pulse:/run/user/1001/pulse",  # Correct PulseAudio socket
        "carlasim/carla:0.9.15",
        "/bin/bash", "-c", "./CarlaUE4.sh -quality-level=Low"
    ]

    try:
        print("Starting CARLA...")
        subprocess.Popen(command, env=env)  # Pass the updated environment
        time.sleep(10)
        # spawn_vehicle_and_camera()
    except subprocess.CalledProcessError as e:
        print(f"Error running CARLA: {e}")
    except KeyboardInterrupt:
        print("CARLA interrupted by user.")
    finally:
        print("Cleaning up...")
        # Perform any necessary cleanup here        
if __name__ == "__main__":
    run_carla()
