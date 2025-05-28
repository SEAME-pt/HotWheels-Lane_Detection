import subprocess
import time
import os

def check_carla_running():
    """Verifica se já existe um container CARLA rodando"""
    try:
        result = subprocess.run(
            ["docker", "ps", "--filter", "ancestor=carlasim/carla:0.9.15", "--format", "{{.ID}}"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip() != ""
    except subprocess.CalledProcessError:
        return False

def get_stopped_carla_container():
    """Obtém ID de um container CARLA parado"""
    try:
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", "ancestor=carlasim/carla:0.9.15", 
             "--filter", "status=exited", "--format", "{{.ID}}", "-n", "1"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None

def run_carla():
    print("Verificando status do CARLA...")
    
    # Verificar se já está rodando
    if check_carla_running():
        print("CARLA já está rodando!")
        return
    
    # Verificar se existe container parado para reutilizar
    stopped_container = get_stopped_carla_container()
    
    if stopped_container:
        print(f"Reutilizando container existente: {stopped_container}")
        subprocess.run(["docker", "start", stopped_container])
        print("Container CARLA reiniciado!")
    else:
        print("Criando novo container CARLA...")
        # Seu comando docker run existente, mas com --rm para auto-limpeza
        command = [
            "docker", "run", "--rm", "-d",  # Adicionar --rm e -d
            "--privileged",
            "--gpus", "all",
            "-e", "DISPLAY=$DISPLAY",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix:rw",
            "-p", "2000-2002:2000-2002",
            "carlasim/carla:0.9.15",
            "/bin/bash", "./CarlaUE4.sh", "-world-port=2000", "-resx=800", "-resy=600"
        ]
        
        env = os.environ.copy()
        subprocess.Popen(command, env=env)
        print("Novo container CARLA criado!")
    
    # Aguardar CARLA inicializar
    print("Aguardando CARLA inicializar...")
    time.sleep(15)

if __name__ == "__main__":
    run_carla()
    print("CARLA está pronto para uso!")