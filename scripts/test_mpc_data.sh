#!/bin/bash

# Script para testar dados de projeção do MPC automaticamente

echo "=== TESTE AUTOMÁTICO DOS DADOS DE PROJEÇÃO DO MPC ==="

# Executar o sistema em background
echo "Iniciando sistema MPC..."
cd /home/jetson/Documents/MPC

# Usar expect para automatizar a interação
expect << EOF
set timeout 60
spawn ./main

# Aguardar inicialização
expect "Digite comandos"

# Mostrar status inicial
send "s\r"
expect "="

# Ativar logs detalhados do MPC
send "d\r"  
expect "Logs detalhados"

# Aguardar alguns frames de lane detection serem processados
sleep 10

# Ativar modo MPC
send "m\r"
expect -re "(MPC ATIVADO|Modo MPC)"

# Aguardar dados de MPC
sleep 15

# Mostrar status final
send "s\r"
expect "="

# Aguardar mais dados
sleep 10

# Sair
send "q\r"
expect eof
EOF

echo "=== TESTE CONCLUÍDO ==="
