#!/bin/bash

# Script para enviar apenas os binários compilados para o Jetson Nano
# Autor: Auto-gerado
# Data: July 3, 2025

set -e  # Exit on any error

# Configurações
JETSON_USER="jetson"
JETSON_HOST="hotwheels-car.netbird.cloud"
JETSON_PATH="/home/jetson/Documents/MPC"
LOCAL_BASE_PATH="/home/michel/Documents/other"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Deploy de Binários para Jetson Nano ===${NC}"

# Função para verificar se o arquivo existe e é executável
check_binary() {
    local file="$1"
    local name="$2"
    
    if [ -f "$file" ]; then
        if [ -x "$file" ]; then
            echo -e "${GREEN}✓ $name encontrado e é executável${NC}"
            return 0
        else
            echo -e "${YELLOW}⚠ $name encontrado mas não é executável${NC}"
            chmod +x "$file"
            echo -e "${GREEN}✓ Permissões corrigidas para $name${NC}"
            return 0
        fi
    else
        echo -e "${RED}✗ $name não encontrado em $file${NC}"
        return 1
    fi
}

# Verificar se os binários existem
echo -e "${BLUE}Verificando binários locais...${NC}"

MAIN_BINARY="$LOCAL_BASE_PATH/build/main"
CAR_CONTROLS_BINARY="$LOCAL_BASE_PATH/car_controls/build/car-controls-qt"

check_binary "$MAIN_BINARY" "main (MPC)" || exit 1
check_binary "$CAR_CONTROLS_BINARY" "car-controls-qt" || exit 1

# Verificar conectividade com o Jetson
echo -e "${BLUE}Verificando conectividade com Jetson Nano...${NC}"
if ! ping -c 1 "$JETSON_HOST" > /dev/null 2>&1; then
    echo -e "${RED}✗ Não foi possível conectar ao Jetson ($JETSON_HOST)${NC}"
    echo -e "${YELLOW}Verifique se:${NC}"
    echo -e "${YELLOW}  - O Jetson está ligado e conectado à rede${NC}"
    echo -e "${YELLOW}  - O IP está correto ($JETSON_HOST)${NC}"
    exit 1
fi
echo -e "${GREEN}✓ Conectividade OK${NC}"

# Criar diretório no Jetson se não existir
echo -e "${BLUE}Criando diretório de destino no Jetson...${NC}"
ssh "$JETSON_USER@$JETSON_HOST" "mkdir -p $JETSON_PATH" || {
    echo -e "${RED}✗ Falha ao criar diretório no Jetson${NC}"
    exit 1
}
echo -e "${GREEN}✓ Diretório criado/verificado${NC}"

# Copiar binários
echo -e "${BLUE}Copiando binários para o Jetson...${NC}"

echo -e "${YELLOW}Enviando main (MPC)...${NC}"
scp "$MAIN_BINARY" "$JETSON_USER@$JETSON_HOST:$JETSON_PATH/main" || {
    echo -e "${RED}✗ Falha ao copiar main${NC}"
    exit 1
}
echo -e "${GREEN}✓ main copiado com sucesso${NC}"

echo -e "${YELLOW}Enviando car-controls-qt...${NC}"
scp "$CAR_CONTROLS_BINARY" "$JETSON_USER@$JETSON_HOST:$JETSON_PATH/car-controls-qt" || {
    echo -e "${RED}✗ Falha ao copiar car-controls-qt${NC}"
    exit 1
}
echo -e "${GREEN}✓ car-controls-qt copiado com sucesso${NC}"

# Verificar permissões no Jetson
echo -e "${BLUE}Configurando permissões no Jetson...${NC}"
ssh "$JETSON_USER@$JETSON_HOST" "chmod +x $JETSON_PATH/main $JETSON_PATH/car-controls-qt" || {
    echo -e "${RED}✗ Falha ao configurar permissões${NC}"
    exit 1
}
echo -e "${GREEN}✓ Permissões configuradas${NC}"

# Verificar tamanhos dos arquivos
echo -e "${BLUE}Verificando integridade dos arquivos...${NC}"
LOCAL_MAIN_SIZE=$(stat -c%s "$MAIN_BINARY")
LOCAL_CAR_SIZE=$(stat -c%s "$CAR_CONTROLS_BINARY")

REMOTE_MAIN_SIZE=$(ssh "$JETSON_USER@$JETSON_HOST" "stat -c%s $JETSON_PATH/main" 2>/dev/null || echo "0")
REMOTE_CAR_SIZE=$(ssh "$JETSON_USER@$JETSON_HOST" "stat -c%s $JETSON_PATH/car-controls-qt" 2>/dev/null || echo "0")

if [ "$LOCAL_MAIN_SIZE" -eq "$REMOTE_MAIN_SIZE" ]; then
    echo -e "${GREEN}✓ main: $LOCAL_MAIN_SIZE bytes${NC}"
else
    echo -e "${RED}✗ main: tamanhos diferentes (local: $LOCAL_MAIN_SIZE, remoto: $REMOTE_MAIN_SIZE)${NC}"
fi

if [ "$LOCAL_CAR_SIZE" -eq "$REMOTE_CAR_SIZE" ]; then
    echo -e "${GREEN}✓ car-controls-qt: $LOCAL_CAR_SIZE bytes${NC}"
else
    echo -e "${RED}✗ car-controls-qt: tamanhos diferentes (local: $LOCAL_CAR_SIZE, remoto: $REMOTE_CAR_SIZE)${NC}"
fi

# Mostrar informações finais
echo -e "${BLUE}=== Deploy Concluído ===${NC}"
echo -e "${GREEN}Binários disponíveis no Jetson em:${NC}"
echo -e "${YELLOW}  $JETSON_PATH/main${NC}"
echo -e "${YELLOW}  $JETSON_PATH/car-controls-qt${NC}"
echo ""
echo -e "${BLUE}Para executar no Jetson:${NC}"
echo -e "${YELLOW}  ssh $JETSON_USER@$JETSON_HOST${NC}"
echo -e "${YELLOW}  cd $JETSON_PATH/bin${NC}"
echo -e "${YELLOW}  ./main  # ou ./car-controls-qt${NC}"
echo ""
echo -e "${GREEN}✓ Deploy realizado com sucesso!${NC}"
