#!/bin/bash

# Configurações
JETSON_USER="jetson"
JETSON_HOST="hotwheels-car.netbird.cloud"
PROJECT_PATH="/home/jetson/Documents/MPC"
LOCAL_PATH="/home/michel/Documents/other"
TMUX_SESSION="MPC"

echo "� Capturando output da sessão tmux '$TMUX_SESSION' na Jetson..."

# Criar diretório local para outputs se não existir
mkdir -p "$LOCAL_PATH/outputs"

# Verificar se já existe arquivo local
if [ -f "$LOCAL_PATH/outputs/output.txt" ]; then
    echo "⚠️  Arquivo local existente será substituído"
fi

echo "📡 Conectando na Jetson e capturando conteúdo do tmux..."

# Conectar via SSH, capturar todo o conteúdo da sessão tmux e salvar em output.txt
ssh $JETSON_USER@$JETSON_HOST "cd $PROJECT_PATH && tmux capture-pane -p -S- -t $TMUX_SESSION > ./output.txt"

if [ $? -ne 0 ]; then
    echo "❌ Erro ao capturar conteúdo da sessão tmux '$TMUX_SESSION'"
    echo "💡 Verifique se a sessão tmux existe na Jetson"
    exit 1
fi

echo "📥 Copiando arquivo via rsync..."

# Usar rsync para copiar o arquivo
rsync -avz $JETSON_USER@$JETSON_HOST:$PROJECT_PATH/output.txt $LOCAL_PATH/outputs/

if [ $? -eq 0 ]; then
    echo "🧹 Limpando buffer do tmux..."
    
    # Limpar o buffer da sessão tmux após cópia bem-sucedida
    ssh $JETSON_USER@$JETSON_HOST "tmux clear-history -t $TMUX_SESSION"
    
    if [ $? -eq 0 ]; then
        echo "✅ Buffer do tmux limpo com sucesso!"
    else
        echo "⚠️  Aviso: Não foi possível limpar o buffer do tmux"
    fi
fi

if [ $? -eq 0 ]; then
    echo "✅ Output capturado e baixado com sucesso!"
    echo "📍 Local: $LOCAL_PATH/outputs/output.txt"
    
    # Mostrar informações do arquivo
    echo ""
    echo "📜 Informações do arquivo:"
    echo "========================="
    ls -lh "$LOCAL_PATH/outputs/output.txt"
    echo ""
    
    # Mostrar as últimas linhas do arquivo
    echo "📄 Últimas 20 linhas do arquivo:"
    echo "================================"
    tail -20 "$LOCAL_PATH/outputs/output.txt"
	echo "==============================="
	echo "📄 Fim da Linha"
else
    echo "❌ Erro ao copiar o arquivo output.txt via rsync"
    exit 1
fi
