#!/bin/bash

# Script para deploy usando rsync
# Criado automaticamente para evitar problemas de escape de aspas

set -e

echo "=== Deploy via rsync ==="
echo "Criando diretório no Jetson..."
ssh jetson@hotwheels-car.netbird.cloud 'mkdir -p /home/jetson/Documents/MPC'

echo "Sincronizando binários..."
rsync -avz --update --ignore-missing-args \
    "$1/build/main" \
    "$1/car_controls/build/car-controls-qt" \
    jetson@hotwheels-car.netbird.cloud:/home/jetson/Documents/MPC/

echo "✅ Deploy concluído!"
