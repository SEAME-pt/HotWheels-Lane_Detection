#!/bin/bash

# Script simples para debug remoto
# Execute este script antes de pressionar F5 no VS Code

JETSON_HOST="hotwheels-car.netbird.cloud"
JETSON_USER="jetson" 
JETSON_PATH="/home/jetson/Documents/MPC"
GDB_PORT="2345"

echo "🚀 Configurando debug remoto..."

# 1. Verificar se o binário existe no Jetson
echo "1️⃣ Verificando binário no Jetson..."
if ssh $JETSON_USER@$JETSON_HOST "test -f $JETSON_PATH/main"; then
    echo "✅ Binário 'main' encontrado no Jetson"
else
    echo "❌ Binário 'main' não encontrado. Execute Deploy primeiro!"
    exit 1
fi

# 2. Matar processos antigos
echo "2️⃣ Limpando processos antigos..."
pkill -f "ssh.*-L.*$GDB_PORT" 2>/dev/null || true
ssh $JETSON_USER@$JETSON_HOST "pkill -f gdbserver" 2>/dev/null || true

# 3. Criar túnel SSH em background
echo "3️⃣ Criando túnel SSH..."
ssh -L $GDB_PORT:localhost:$GDB_PORT -N $JETSON_USER@$JETSON_HOST &
SSH_TUNNEL_PID=$!
sleep 2

if ps -p $SSH_TUNNEL_PID > /dev/null; then
    echo "✅ Túnel SSH criado (PID: $SSH_TUNNEL_PID)"
else
    echo "❌ Falha ao criar túnel SSH"
    exit 1
fi

# 4. Aguardar e iniciar gdbserver
echo "4️⃣ Iniciando gdbserver no Jetson..."
echo ""
echo "🎯 Execute este comando em outro terminal:"
echo "ssh $JETSON_USER@$JETSON_HOST \"cd $JETSON_PATH && gdbserver localhost:$GDB_PORT ./main\""
echo ""
echo "📍 Ou execute:"
echo "./scripts/start_gdbserver_only.sh"
echo ""
echo "✨ Após iniciar o gdbserver:"
echo "   1. Vá para o VS Code"
echo "   2. Pressione F5 ou Ctrl+Shift+D"
echo "   3. Escolha uma das configurações disponíveis:"
echo "      • 'Debug Remote Jetson' (padrão)"
echo "      • 'Debug Remote Jetson (Alternative)' (se houver problemas)"
echo ""
echo "🛑 Para parar: Pressione Ctrl+C aqui para limpar túnel SSH"

# Aguardar Ctrl+C
trap "echo ''; echo '🧹 Limpando túnel SSH...'; kill $SSH_TUNNEL_PID 2>/dev/null; echo '✅ Cleanup concluído!'" EXIT
wait
