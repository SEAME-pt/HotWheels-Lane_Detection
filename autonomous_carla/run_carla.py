"""
run_carla.py

Este script é responsável por iniciar o simulador CARLA em um contêiner Docker. Ele configura as variáveis de ambiente necessárias e executa o comando Docker para iniciar o simulador.

Funções:
- run_carla: Configura o ambiente e executa o simulador CARLA em um contêiner Docker.

Dependências:
- Docker instalado e configurado no sistema.
- A imagem Docker `carlasim/carla:0.9.15` disponível localmente ou no repositório Docker.

Como usar:
- Execute este script diretamente para iniciar o simulador CARLA.

"""

import subprocess
import os
import time

def run_carla():
    """
    Configura o ambiente e executa o simulador CARLA em um contêiner Docker.

    O método define variáveis de ambiente necessárias para o funcionamento do CARLA, como DISPLAY, XDG_RUNTIME_DIR e PULSE_SERVER. Em seguida, executa o comando Docker para iniciar o simulador.

    Exceções tratadas:
    - subprocess.CalledProcessError: Caso ocorra um erro ao executar o comando Docker.
    - KeyboardInterrupt: Caso o usuário interrompa a execução manualmente.

    """
    env = os.environ.copy()
    env["DISPLAY"] = os.getenv("DISPLAY")
    env["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}" 
    env["PULSE_SERVER"] = "unix:/run/user/1001/pulse/native"
    #TODO: Codigo Principal.
    command = [
        "docker", "run", "--rm",
        "--gpus", "all",
        "--net=host",
        "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
        "-v", "/dev/snd:/dev/snd",  # Passa dispositivos de áudio
        "-e", f"DISPLAY={env['DISPLAY']}",
        "-e", f"XDG_RUNTIME_DIR={env['XDG_RUNTIME_DIR']}",
        "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
        "-e", f"PULSE_SERVER={env['PULSE_SERVER']}",  # Usa o servidor PulseAudio correto
        "-v", "/run/user/1001/pulse:/run/user/1001/pulse",  # Socket PulseAudio correto
        "carlasim/carla:0.9.15",
        "/bin/bash", "-c", "./CarlaUE4.sh -quality-level=Low -windowed -ResX=640 -ResY=360"
    ]
    #! Para SSH
    # command = [
    #     "docker", "run", "--rm",
    #     "--gpus", "all",
    #     "--net=host",
    #     "-v", "/tmp/.X11-unix:/tmp/.X11-unix",
    #     "-v", "/dev/snd:/dev/snd",
    #     "-e", f"DISPLAY={env['DISPLAY']}",
    #     "-e", f"XDG_RUNTIME_DIR={env['XDG_RUNTIME_DIR']}",
    #     "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
    #     "-e", f"PULSE_SERVER={env['PULSE_SERVER']}",
    #     "-v", "/run/user/1001/pulse:/run/user/1001/pulse",
    #     "carlasim/carla:0.9.15",
    #     "/bin/bash", "-c", "./CarlaUE4.sh -carla-server -quality-level=Low -RenderOffScreen"
    # ]
    
    try:
        print("Iniciando o CARLA...")
        subprocess.Popen(command, env=env)  # Passa o ambiente atualizado
        time.sleep(10)
    except subprocess.CalledProcessError as e:
        print(f"Erro ao executar o CARLA: {e}")
    except KeyboardInterrupt:
        print("Execução do CARLA interrompida pelo usuário.")

if __name__ == "__main__":
    run_carla()
