#!/bin/bash

# Script para limpeza de memória na Jetson Nano (versão adaptada)
echo "=== Jetson Memory Cleanup ==="

echo "Memória antes da limpeza:"
free -h

echo "Limpando cache do sistema..."
sudo sync
sudo sysctl vm.drop_caches=3

echo "Limpando buffers..."
sudo sync

echo "Liberando swap se necessário..."
sudo swapoff -a && sudo swapon -a

echo "Matando processos órfãos do OpenCV..."
sudo pkill -f opencv

echo "Limpando arquivos temporários..."
sudo rm -rf /tmp/* 2>/dev/null

echo "Forçando garbage collection do CUDA..."
# Não existe equivalente direto para resetar GPU no Jetson Nano,
# mas é possível limpar memória de processos específicos ou reiniciar o serviço CUDA.
# Aqui apenas um aviso:
echo "Aviso: não há comando nvidia-smi no Jetson Nano. Use tegrastats para monitorar GPU."

echo "Memória após limpeza:"
free -h

echo "=== Status da GPU (via tegrastats) ==="
# Exibe um snapshot das estatísticas da GPU (pressione Ctrl+C para sair)
sudo tegrastats --interval 1000 --count 1

echo "✅ Limpeza concluída!"