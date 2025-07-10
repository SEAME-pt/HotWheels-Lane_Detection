#!/bin/bash

# Teste de conexão do debug remoto
echo "🧪 Testando conexão de debug remoto..."

JETSON_HOST="hotwheels-car.netbird.cloud"
JETSON_USER="jetson"
GDB_PORT="2345"

# 1. Verificar se o túnel SSH está ativo
echo "1️⃣ Verificando túnel SSH..."
if netstat -tln | grep -q ":$GDB_PORT.*LISTEN"; then
    echo "✅ Túnel SSH ativo na porta $GDB_PORT"
else
    echo "❌ Túnel SSH não encontrado. Execute:"
    echo "   ssh -L $GDB_PORT:localhost:$GDB_PORT -N $JETSON_USER@$JETSON_HOST &"
    exit 1
fi

# 2. Verificar se gdbserver está rodando no Jetson
echo "2️⃣ Verificando gdbserver no Jetson..."
if ssh $JETSON_USER@$JETSON_HOST "pgrep -f gdbserver" >/dev/null 2>&1; then
    echo "✅ gdbserver está rodando no Jetson"
else
    echo "❌ gdbserver não está rodando. Execute:"
    echo "   ./scripts/start_gdbserver_only.sh"
    exit 1
fi

# 3. Testar conexão direta com gdb
echo "3️⃣ Testando conexão GDB..."
echo "target extended-remote localhost:$GDB_PORT" | timeout 5 gdb-multiarch -batch > /tmp/gdb_test.log 2>&1

if grep -q "Remote debugging using" /tmp/gdb_test.log; then
    echo "✅ Conexão GDB funcional"
else
    echo "❌ Falha na conexão GDB. Log:"
    cat /tmp/gdb_test.log
    exit 1
fi

# 4. Verificar se o binário local tem símbolos
echo "4️⃣ Verificando símbolos do binário..."
if file build/main | grep -q "with debug_info"; then
    echo "✅ Binário local tem símbolos de debug"
else
    echo "❌ Binário local não tem símbolos de debug"
    echo "   Recompile com símbolos: make clean && make"
    exit 1
fi

echo ""
echo "🎉 Tudo pronto para debug!"
echo "📍 Agora você pode:"
echo "   1. Abrir VS Code"
echo "   2. Pressionar F5"
echo "   3. Escolher 'Debug Remote Jetson'"

rm -f /tmp/gdb_test.log
