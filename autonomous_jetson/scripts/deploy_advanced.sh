#!/bin/bash

# Script avançado para deploy de binários com opções de configuração
# Autor: Auto-gerado
# Data: July 3, 2025

set -e

# Configurações padrão (podem ser sobrescritas por arquivo de config)
JETSON_USER="jetson"
JETSON_HOST="192.168.1.100"
JETSON_PATH="/home/jetson/car_controls"
LOCAL_BASE_PATH="/home/michel/Documents/other"
BACKUP_ENABLED=true
VERIFY_CHECKSUMS=true

# Arquivo de configuração
CONFIG_FILE="$HOME/.jetson_deploy_config"

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m'

# Função para mostrar ajuda
show_help() {
    echo -e "${BLUE}Deploy de Binários para Jetson Nano${NC}"
    echo ""
    echo "Uso: $0 [opções]"
    echo ""
    echo "Opções:"
    echo "  -h, --help              Mostra esta ajuda"
    echo "  -c, --config            Configura conexão com Jetson"
    echo "  -n, --no-backup         Não faz backup dos binários existentes"
    echo "  -f, --force             Força deploy mesmo com diferenças de checksum"
    echo "  -v, --verbose           Modo verboso"
    echo "  -t, --test-connection   Testa apenas a conexão"
    echo "  --main-only             Envia apenas o binário main (MPC)"
    echo "  --car-only              Envia apenas o binário car-controls-qt"
    echo ""
    echo "Configuração:"
    echo "  O script pode ser configurado editando: $CONFIG_FILE"
    echo "  Ou executando: $0 --config"
    echo ""
}

# Carregar configuração se existir
load_config() {
    if [ -f "$CONFIG_FILE" ]; then
        source "$CONFIG_FILE"
        echo -e "${GREEN}✓ Configuração carregada de $CONFIG_FILE${NC}"
    fi
}

# Configurar conexão interativamente
configure_connection() {
    echo -e "${BLUE}=== Configuração da Conexão Jetson ===${NC}"
    
    echo -e "${YELLOW}Configuração atual:${NC}"
    echo "  Usuário: $JETSON_USER"
    echo "  Host: $JETSON_HOST"
    echo "  Caminho: $JETSON_PATH"
    echo ""
    
    read -p "Usuário do Jetson [$JETSON_USER]: " new_user
    [ -n "$new_user" ] && JETSON_USER="$new_user"
    
    read -p "IP/hostname do Jetson [$JETSON_HOST]: " new_host
    [ -n "$new_host" ] && JETSON_HOST="$new_host"
    
    read -p "Caminho no Jetson [$JETSON_PATH]: " new_path
    [ -n "$new_path" ] && JETSON_PATH="$new_path"
    
    # Salvar configuração
    cat > "$CONFIG_FILE" << EOF
# Configuração do Deploy Jetson
JETSON_USER="$JETSON_USER"
JETSON_HOST="$JETSON_HOST"
JETSON_PATH="$JETSON_PATH"
BACKUP_ENABLED=$BACKUP_ENABLED
VERIFY_CHECKSUMS=$VERIFY_CHECKSUMS
EOF
    
    echo -e "${GREEN}✓ Configuração salva em $CONFIG_FILE${NC}"
    
    # Testar conexão
    echo -e "${BLUE}Testando conexão...${NC}"
    if ssh -o ConnectTimeout=5 "$JETSON_USER@$JETSON_HOST" "echo 'Conexão OK'" 2>/dev/null; then
        echo -e "${GREEN}✓ Conexão estabelecida com sucesso${NC}"
    else
        echo -e "${RED}✗ Falha na conexão${NC}"
        echo -e "${YELLOW}Verifique se:${NC}"
        echo -e "${YELLOW}  - O Jetson está ligado e acessível${NC}"
        echo -e "${YELLOW}  - As chaves SSH estão configuradas${NC}"
        echo -e "${YELLOW}  - O usuário e host estão corretos${NC}"
    fi
}

# Testar apenas conexão
test_connection() {
    echo -e "${BLUE}Testando conexão com $JETSON_USER@$JETSON_HOST...${NC}"
    
    if ping -c 1 -W 3 "$JETSON_HOST" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Ping OK${NC}"
        
        if ssh -o ConnectTimeout=5 "$JETSON_USER@$JETSON_HOST" "uname -a" 2>/dev/null; then
            echo -e "${GREEN}✓ SSH OK${NC}"
            ssh "$JETSON_USER@$JETSON_HOST" "df -h $JETSON_PATH 2>/dev/null || echo 'Diretório $JETSON_PATH não existe'"
        else
            echo -e "${RED}✗ SSH falhou${NC}"
            return 1
        fi
    else
        echo -e "${RED}✗ Ping falhou${NC}"
        return 1
    fi
}

# Função para fazer backup
backup_existing() {
    if [ "$BACKUP_ENABLED" = false ]; then
        return 0
    fi
    
    echo -e "${BLUE}Fazendo backup dos binários existentes...${NC}"
    
    BACKUP_DIR="$JETSON_PATH/backup/$(date +%Y%m%d_%H%M%S)"
    
    ssh "$JETSON_USER@$JETSON_HOST" "
        if [ -f $JETSON_PATH/bin/main ] || [ -f $JETSON_PATH/bin/car-controls-qt ]; then
            mkdir -p $BACKUP_DIR
            [ -f $JETSON_PATH/bin/main ] && cp $JETSON_PATH/bin/main $BACKUP_DIR/ && echo 'main backed up'
            [ -f $JETSON_PATH/bin/car-controls-qt ] && cp $JETSON_PATH/bin/car-controls-qt $BACKUP_DIR/ && echo 'car-controls-qt backed up'
            echo 'Backup criado em $BACKUP_DIR'
        else
            echo 'Nenhum binário existente para backup'
        fi
    " 2>/dev/null || true
}

# Função principal de deploy
deploy_binary() {
    local local_path="$1"
    local remote_name="$2"
    local description="$3"
    
    if [ ! -f "$local_path" ]; then
        echo -e "${YELLOW}⚠ $description não encontrado em $local_path${NC}"
        return 1
    fi
    
    if [ ! -x "$local_path" ]; then
        echo -e "${YELLOW}⚠ $description não é executável, corrigindo...${NC}"
        chmod +x "$local_path"
    fi
    
    echo -e "${BLUE}Enviando $description...${NC}"
    
    # Calcular checksum local
    local local_checksum=$(sha256sum "$local_path" | cut -d' ' -f1)
    local file_size=$(stat -c%s "$local_path")
    
    echo -e "${YELLOW}  Tamanho: $(numfmt --to=iec $file_size)${NC}"
    echo -e "${YELLOW}  SHA256: ${local_checksum:0:16}...${NC}"
    
    # Copiar arquivo
    scp "$local_path" "$JETSON_USER@$JETSON_HOST:$JETSON_PATH/bin/$remote_name" || {
        echo -e "${RED}✗ Falha ao copiar $description${NC}"
        return 1
    }
    
    # Verificar checksum remoto se habilitado
    if [ "$VERIFY_CHECKSUMS" = true ]; then
        local remote_checksum=$(ssh "$JETSON_USER@$JETSON_HOST" "sha256sum $JETSON_PATH/bin/$remote_name | cut -d' ' -f1" 2>/dev/null)
        
        if [ "$local_checksum" = "$remote_checksum" ]; then
            echo -e "${GREEN}✓ $description verificado (checksum OK)${NC}"
        else
            echo -e "${RED}✗ $description: checksum mismatch${NC}"
            if [ "$FORCE_DEPLOY" != true ]; then
                return 1
            fi
        fi
    fi
    
    # Definir permissões
    ssh "$JETSON_USER@$JETSON_HOST" "chmod +x $JETSON_PATH/bin/$remote_name"
    
    echo -e "${GREEN}✓ $description implantado com sucesso${NC}"
    return 0
}

# Parse dos argumentos
DEPLOY_MAIN=true
DEPLOY_CAR=true
VERBOSE=false
FORCE_DEPLOY=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -c|--config)
            load_config
            configure_connection
            exit 0
            ;;
        -n|--no-backup)
            BACKUP_ENABLED=false
            shift
            ;;
        -f|--force)
            FORCE_DEPLOY=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -t|--test-connection)
            load_config
            test_connection
            exit $?
            ;;
        --main-only)
            DEPLOY_CAR=false
            shift
            ;;
        --car-only)
            DEPLOY_MAIN=false
            shift
            ;;
        *)
            echo -e "${RED}Opção desconhecida: $1${NC}"
            show_help
            exit 1
            ;;
    esac
done

# Carregar configuração
load_config

echo -e "${PURPLE}=== Deploy de Binários para Jetson Nano ===${NC}"

# Verificar conectividade
test_connection || exit 1

# Criar diretórios necessários
echo -e "${BLUE}Preparando ambiente no Jetson...${NC}"
ssh "$JETSON_USER@$JETSON_HOST" "mkdir -p $JETSON_PATH/bin $JETSON_PATH/backup"

# Fazer backup
backup_existing

# Deploy dos binários
success_count=0
total_count=0

if [ "$DEPLOY_MAIN" = true ]; then
    total_count=$((total_count + 1))
    if deploy_binary "$LOCAL_BASE_PATH/build/main" "main" "MPC Binary"; then
        success_count=$((success_count + 1))
    fi
fi

if [ "$DEPLOY_CAR" = true ]; then
    total_count=$((total_count + 1))
    if deploy_binary "$LOCAL_BASE_PATH/car_controls/build/car-controls-qt" "car-controls-qt" "Car Controls Binary"; then
        success_count=$((success_count + 1))
    fi
fi

# Relatório final
echo -e "${PURPLE}=== Relatório Final ===${NC}"
echo -e "${GREEN}Binários implantados: $success_count/$total_count${NC}"

if [ $success_count -eq $total_count ] && [ $total_count -gt 0 ]; then
    echo -e "${GREEN}✓ Deploy concluído com sucesso!${NC}"
    echo ""
    echo -e "${BLUE}Para executar no Jetson:${NC}"
    echo -e "${YELLOW}  ssh $JETSON_USER@$JETSON_HOST${NC}"
    echo -e "${YELLOW}  cd $JETSON_PATH/bin${NC}"
    if [ "$DEPLOY_MAIN" = true ]; then
        echo -e "${YELLOW}  ./main${NC}"
    fi
    if [ "$DEPLOY_CAR" = true ]; then
        echo -e "${YELLOW}  ./car-controls-qt${NC}"
    fi
    exit 0
else
    echo -e "${RED}✗ Deploy com falhas${NC}"
    exit 1
fi
