#!/bin/bash

# Script para testar dados de projeção do MPC automaticamente

echo "=== TESTE AUTOMÁTICO DOS DADOS DE PROJEÇÃO DO MPC ==="
echo "Este script irá executar o sistema MPC e enviar comandos automaticamente"
echo "Aguarde... o processo levará cerca de 1 minuto"

# Mudar para o diretório correto
cd /home/jetson/Documents/MPC/scripts

# Criar um pipe nomeado para comunicação
PIPE_NAME="/tmp/mpc_commands"
mkfifo "$PIPE_NAME" 2>/dev/null || true

# Função para enviar comandos
send_commands() {
    echo "Enviando comandos para o sistema MPC..."
    sleep 5   # Aguardar inicialização
    
    echo "s" > "$PIPE_NAME"  # Mostrar status inicial
    sleep 2
    
    echo "d" > "$PIPE_NAME"  # Ativar logs detalhados
    sleep 2
    
    echo "Aguardando frames de lane detection..."
    sleep 15  # Aguardar alguns frames serem processados
    
    echo "m" > "$PIPE_NAME"  # Ativar modo MPC
    sleep 2
    
    echo "Modo MPC ativado, coletando dados..."
    sleep 20  # Aguardar dados de MPC
    
    echo "s" > "$PIPE_NAME"  # Mostrar status final
    sleep 5
    
    echo "q" > "$PIPE_NAME"  # Sair
    sleep 2
}

# Executar comandos em background
send_commands &
COMMANDS_PID=$!

# Executar o sistema principal redirecionando entrada do pipe
echo "Iniciando sistema MPC..."
timeout 90s ./main < "$PIPE_NAME" > outputs/mpc_test_output.txt 2>&1

# Cleanup
kill $COMMANDS_PID 2>/dev/null || true
rm -f "$PIPE_NAME"

echo "=== TESTE CONCLUÍDO ==="
echo "Resultados salvos em: outputs/mpc_test_output.txt"
echo ""
echo "Para ver os dados de projeção do MPC:"
echo "cat outputs/mpc_test_output.txt | grep -A5 -B5 'MPC TRAJECTORY\|MPC CONTROL\|LANE DETECTION'"
