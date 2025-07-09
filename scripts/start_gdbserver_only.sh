#!/bin/bash

# Script para iniciar apenas o gdbserver no Jetson

JETSON_HOST="hotwheels-car.netbird.cloud"
JETSON_USER="jetson"
JETSON_PATH="/home/jetson/Documents/MPC"
GDB_PORT="2345"

echo "🐛 Iniciando gdbserver no Jetson..."
ssh -t $JETSON_USER@$JETSON_HOST "cd $JETSON_PATH && gdbserver localhost:$GDB_PORT ./main"

# Parar o gdbserver
echo "🛑 gdbserver parado. Pressione Ctrl+C para sair."