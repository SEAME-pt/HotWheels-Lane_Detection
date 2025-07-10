#!/bin/bash

# Script de diagnóstico para debug remoto
echo "🔍 Diagnóstico do Debug Remoto"
echo "================================="

# Verificar configurações locais
echo "📊 Configurações Locais:"
echo "  gdb-multiarch: $(which gdb-multiarch || echo 'NÃO ENCONTRADO')"
echo "  Versão GDB: $(gdb-multiarch --version 2>/dev/null | head -1 || echo 'ERRO')"
echo ""

# Verificar arquivos de configuração
echo "📁 Arquivos de Configuração:"
if [ -f ".vscode/launch.json" ]; then
    echo "  ✅ launch.json existe"
    if python3 -m json.tool .vscode/launch.json >/dev/null 2>&1; then
        echo "  ✅ launch.json válido"
    else
        echo "  ❌ launch.json inválido"
    fi
else
    echo "  ❌ launch.json não encontrado"
fi

if [ -f ".vscode/tasks.json" ]; then
    echo "  ✅ tasks.json existe"
    if python3 -m json.tool .vscode/tasks.json >/dev/null 2>&1; then
        echo "  ✅ tasks.json válido"
    else
        echo "  ❌ tasks.json inválido"
    fi
else
    echo "  ❌ tasks.json não encontrado"
fi
echo ""

# Verificar scripts
echo "🔧 Scripts de Debug:"
for script in "setup_debug.sh" "start_gdbserver_only.sh" "deploy_rsync.sh"; do
    if [ -f "scripts/$script" ]; then
        if [ -x "scripts/$script" ]; then
            echo "  ✅ $script (executável)"
        else
            echo "  ⚠️  $script (não executável)"
        fi
    else
        echo "  ❌ $script não encontrado"
    fi
done
echo ""

# Verificar binário local
echo "📦 Binários:"
if [ -f "build/main" ]; then
    echo "  ✅ build/main existe"
    file_info=$(file build/main)
    if echo "$file_info" | grep -q "ARM aarch64"; then
        echo "  ✅ Compilado para ARM64"
    else
        echo "  ⚠️  Arquitetura: $file_info"
    fi
else
    echo "  ❌ build/main não encontrado"
fi
echo ""

# Testar conectividade com Jetson
echo "🌐 Conectividade com Jetson:"
JETSON_HOST="hotwheels-car.netbird.cloud"
JETSON_USER="jetson"

if timeout 5 ssh -o ConnectTimeout=5 -o BatchMode=yes $JETSON_USER@$JETSON_HOST "echo 'OK'" >/dev/null 2>&1; then
    echo "  ✅ SSH conecta com $JETSON_HOST"
    
    # Verificar binário remoto
    if ssh $JETSON_USER@$JETSON_HOST "test -f /home/jetson/Documents/MPC/main" 2>/dev/null; then
        echo "  ✅ Binário remoto existe"
    else
        echo "  ❌ Binário remoto não encontrado"
    fi
    
    # Verificar gdbserver remoto
    if ssh $JETSON_USER@$JETSON_HOST "which gdbserver" >/dev/null 2>&1; then
        echo "  ✅ gdbserver instalado no Jetson"
    else
        echo "  ❌ gdbserver não encontrado no Jetson"
    fi
else
    echo "  ❌ Não consegue conectar com SSH"
fi
echo ""

# Verificar portas em uso
echo "🔌 Portas:"
if netstat -tlnp 2>/dev/null | grep -q ":2345"; then
    echo "  ⚠️  Porta 2345 em uso:"
    netstat -tlnp 2>/dev/null | grep ":2345"
else
    echo "  ✅ Porta 2345 disponível"
fi
echo ""

echo "🏁 Diagnóstico concluído!"
echo ""
echo "💡 Próximos passos recomendados:"
echo "   1. Se algum item está ❌, corrija antes de prosseguir"
echo "   2. Execute: ./scripts/setup_debug.sh"
echo "   3. Em outro terminal: ./scripts/start_gdbserver_only.sh"
echo "   4. No VS Code: F5 -> 'Debug Remote Jetson'"
