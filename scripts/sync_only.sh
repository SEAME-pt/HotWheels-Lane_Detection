#!/bin/bash

# Configurações
JETSON_USER="jetson"
JETSON_HOST="hotwheels-car.netbird.cloud"
PROJECT_PATH="/home/jetson/Documents/MPC"
LOCAL_PATH="/home/michel/Documents/other"

echo "Sincronizando apenas arquivos modificados..."
rsync -avz --update --exclude='.git' --exclude='build/' --exclude='*.o' --exclude='*.sh' --exclude='*.txt' --exclude='*.tmp' --dry-run $LOCAL_PATH/ $JETSON_USER@$JETSON_HOST:$PROJECT_PATH/

read -p "Confirmar sincronização? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rsync -avz --update --exclude='.git' --exclude='build/' --exclude='*.sh' --exclude='*.o' --exclude='*.tmp' $LOCAL_PATH/ $JETSON_USER@$JETSON_HOST:$PROJECT_PATH/
    echo "Sincronização concluída!"
else
    echo "Sincronização cancelada."
fi
