#!/bin/bash

# Script para iniciar debug remoto automaticamente
# Usage: ./start_remote_debug.sh <jetson_ip> [jetson_user] [binary_name]

JETSON_IP=${1:-"hotwheels-car.netbird.cloud"}
JETSON_USER=${2:-"jetson"}
BINARY_NAME=${3:-"main"}
JETSON_HOME="/home/${JETSON_USER}/Documents/MPC"
REMOTE_BINARY="${JETSON_HOME}/${BINARY_NAME}"
GDB_PORT="2345"

echo "=== Iniciando Debug Remoto ==="
echo "IP do Jetson: ${JETSON_IP}"
echo "Usuário: ${JETSON_USER}"
echo "Binário: ${BINARY_NAME}"
echo "=============================="

# Função para cleanup ao sair
cleanup() {
    echo ""
    echo "🧹 Limpando processos..."
    # Matar túnel SSH se existir
    pkill -f "ssh.*-L.*${GDB_PORT}.*${JETSON_IP}"
    # Matar gdbserver no Jetson se existir
    ssh "${JETSON_USER}@${JETSON_IP}" "pkill -f gdbserver" 2>/dev/null
    echo "✅ Cleanup concluído!"
    exit 0
}

# Capturar Ctrl+C
trap cleanup SIGINT SIGTERM

# Verificar se já existe um túnel SSH
if pgrep -f "ssh.*-L.*${GDB_PORT}.*${JETSON_IP}" > /dev/null; then
    echo "⚠️  Túnel SSH já existe. Matando processo anterior..."
    pkill -f "ssh.*-L.*${GDB_PORT}.*${JETSON_IP}"
    sleep 2
fi

# Verificar se gdbserver já está rodando no Jetson
echo "🔍 Verificando se gdbserver já está rodando no Jetson..."
ssh "${JETSON_USER}@${JETSON_IP}" "pkill -f gdbserver" 2>/dev/null

# Criar túnel SSH em background
echo "🌉 Criando túnel SSH para debug..."
ssh -L ${GDB_PORT}:localhost:${GDB_PORT} -N "${JETSON_USER}@${JETSON_IP}" &
SSH_TUNNEL_PID=$!

# Aguardar um pouco para o túnel ser estabelecido
sleep 3

# Verificar se o túnel foi criado com sucesso
if ! ps -p $SSH_TUNNEL_PID > /dev/null; then
    echo "❌ Erro ao criar túnel SSH!"
    exit 1
fi

echo "✅ Túnel SSH criado (PID: $SSH_TUNNEL_PID)"

# Configurar DISPLAY para X11 forwarding
export DISPLAY=${DISPLAY:-:0}

# Iniciar gdbserver no Jetson em background
echo "🐛 Iniciando gdbserver no Jetson Nano..."
echo "   Comando: gdbserver localhost:${GDB_PORT} ${REMOTE_BINARY}"
echo "   DISPLAY: ${DISPLAY}"

# Executar gdbserver com X11 forwarding
ssh -X "${JETSON_USER}@${JETSON_IP}" "DISPLAY=${DISPLAY} gdbserver localhost:${GDB_PORT} ${REMOTE_BINARY}" &
GDBSERVER_PID=$!

echo "✅ gdbserver iniciado!"
echo ""
echo "🎯 Agora você pode:"
echo "   1. Abrir VS Code"
echo "   2. Ir para o menu Run and Debug (Ctrl+Shift+D)"
echo "   3. Selecionar 'Debug Remote SSH (Jetson Nano)'"
echo "   4. Pressionar F5 para iniciar o debug"
echo ""
echo "💡 O gdbserver está aguardando conexão na porta ${GDB_PORT}"
echo "🔗 Túnel SSH: localhost:${GDB_PORT} -> ${JETSON_IP}:${GDB_PORT}"
echo ""
echo "Pressione Ctrl+C para parar o debug e limpar os processos."

# Aguardar até que o usuário pressione Ctrl+C
wait
