#!/bin/bash

# Configurações
JETSON_USER="jetson"
JETSON_HOST="hotwheels-car.netbird.cloud"
PROJECT_PATH="/home/jetson/Documents/MPC"
LOCAL_PATH="/home/michel/Documents/other"

echo "Sincronizando apenas arquivos modificados..."
rsync -avz --update --exclude='.git' --exclude='build/' --exclude='*.o' --exclude='*.txt' --exclude='*.tmp' $LOCAL_PATH/ $JETSON_USER@$JETSON_HOST:$PROJECT_PATH/

echo "Compilando na Jetson Nano..."
# ssh $JETSON_USER@$JETSON_HOST "cd $PROJECT_PATH && make clean && make $1 -j$(nproc)"
ssh $JETSON_USER@$JETSON_HOST "cd $PROJECT_PATH && rm -fr main && make $1"
# se aparecer algua linha "Error 1" compilacao falhou
if [ $? -ne 0 ]; then
	echo "❌ Compilação falhou!"
	exit 1
fi
echo "Compilação concluída!"