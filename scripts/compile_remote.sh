#!/bin/bash

# Configurações
JETSON_USER="jetson"
JETSON_HOST="hotwheels-car.netbird.cloud"
PROJECT_PATH="/home/jetson/Documents/MPC"
LOCAL_PATH="/home/michel-batista/Documents/SEA_ME/HotWheels-Lane_Detection"

echo "Sincronizando apenas arquivos modificados..."
rsync -avz --update --exclude='.git' --exclude='build/' --exclude='*.o' --exclude='*.txt' --exclude='*.tmp' $LOCAL_PATH/ $JETSON_USER@$JETSON_HOST:$PROJECT_PATH/

echo "Compilando na Jetson Nano..."
ssh $JETSON_USER@$JETSON_HOST "cd $PROJECT_PATH && make $1 -j$(nproc)"

echo "Compilação concluída!"