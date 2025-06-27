#!/bin/bash

# Script simples para testar o MPC manualmente

echo "=== TESTE MANUAL DO MPC ==="
echo ""
echo "INSTRUÇÕES:"
echo "1. O sistema será iniciado"
echo "2. Aguarde ver mensagens '[LaneDetection] Frame shape...'"
echo "3. Digite 's' para ver status"
echo "4. Digite 'd' para ativar logs detalhados"
echo "5. Digite 'm' para ativar modo MPC"
echo "6. Aguarde e observe os dados de projeção"
echo "7. Digite 'q' para sair"
echo ""
echo "Pressione ENTER para continuar..."
read

cd /home/jetson/Documents/MPC/scripts

echo "Iniciando sistema MPC..."
echo "Os dados serão salvos em outputs/manual_mpc_test.txt"

# Usar tee para mostrar saída e salvar ao mesmo tempo
./main 2>&1 | tee outputs/manual_mpc_test.txt

echo ""
echo "=== TESTE CONCLUÍDO ==="
echo "Dados salvos em: outputs/manual_mpc_test.txt"
