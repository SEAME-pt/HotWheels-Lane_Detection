#!/usr/bin/env python3

import subprocess
import time
import signal
import sys
import os

class MPCTester:
    def __init__(self):
        self.process = None
        self.running = True

    def signal_handler(self, signum, frame):
        print("\nInterrompendo teste...")
        if self.process:
            self.process.terminate()
        self.running = False
        sys.exit(0)

    def run_test(self):
        print("=== TESTE AUTOMÁTICO DO MPC COM PYTHON ===")
        print("Iniciando sistema MPC...")
        
        # Mudar para o diretório correto
        os.chdir("/home/jetson/Documents/MPC")
        
        # Configurar handler para Ctrl+C
        signal.signal(signal.SIGINT, self.signal_handler)
        
        try:
            # Iniciar o processo
            self.process = subprocess.Popen(
                ['./main'],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Arquivo para salvar output
            output_file = open('outputs/python_mpc_test.txt', 'w')
            
            # Lista de comandos para enviar
            commands = [
                (5, 's'),    # Status inicial após 5s
                (2, 'd'),    # Ativar logs detalhados após 2s
                (15, 'm'),   # Ativar MPC após 15s
                (20, 's'),   # Status final após 20s
                (10, 'q')    # Sair após 10s
            ]
            
            command_index = 0
            start_time = time.time()
            next_command_time = start_time + commands[0][0]
            
            print("Sistema iniciado. Enviando comandos automaticamente...")
            
            while self.running and self.process.poll() is None:
                # Ler output do processo
                try:
                    line = self.process.stdout.readline()
                    if line:
                        print(line.strip())
                        output_file.write(line)
                        output_file.flush()
                        
                        # Verificar se precisa ativar MPC após ver lane detection
                        if "LaneDetection" in line and command_index == 2:
                            print("Detectando frames de lane detection...")
                    
                except:
                    pass
                
                # Enviar próximo comando se for a hora
                current_time = time.time()
                if command_index < len(commands) and current_time >= next_command_time:
                    delay, cmd = commands[command_index]
                    print(f"\n>>> Enviando comando: '{cmd}'")
                    
                    try:
                        self.process.stdin.write(cmd + '\n')
                        self.process.stdin.flush()
                    except:
                        print("Erro ao enviar comando")
                    
                    command_index += 1
                    if command_index < len(commands):
                        next_command_time = current_time + commands[command_index][0]
                
                time.sleep(0.1)  # Small delay to prevent excessive CPU usage
            
            output_file.close()
            
            print("\n=== TESTE CONCLUÍDO ===")
            print("Resultados salvos em: outputs/python_mpc_test.txt")
            
            # Mostrar dados relevantes
            print("\n=== DADOS DE PROJEÇÃO ENCONTRADOS ===")
            with open('outputs/python_mpc_test.txt', 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if any(keyword in line for keyword in ['MPC TRAJECTORY', 'MPC CONTROL', 'LANE DETECTION DATA', 'Generated trajectory']):
                        # Mostrar esta linha e as próximas 3
                        for j in range(min(4, len(lines) - i)):
                            print(lines[i + j].strip())
                        print("---")
            
        except Exception as e:
            print(f"Erro durante o teste: {e}")
        finally:
            if self.process:
                self.process.terminate()

if __name__ == "__main__":
    tester = MPCTester()
    tester.run_test()
