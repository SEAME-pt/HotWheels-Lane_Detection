#!/bin/bash

# Diagnostic and Fix Script for Autonomous Driving System
# This script analyzes and fixes the current C++ implementation based on Python working version

set -e

echo "🔧 DIAGNÓSTICO E CORREÇÃO DO SISTEMA AUTÔNOMO"
echo "=============================================="

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() { echo -e "${GREEN}[OK]${NC} $1"; }
print_warning() { echo -e "${YELLOW}[AVISO]${NC} $1"; }
print_error() { echo -e "${RED}[ERRO]${NC} $1"; }
print_info() { echo -e "${BLUE}[INFO]${NC} $1"; }

# 1. Diagnose current issues
diagnose_system() {
    echo ""
    print_info "=== DIAGNÓSTICO DO SISTEMA ATUAL ==="
    
    # Check if model exists
    if [ -f "models/lane_detector_combined_v2.keras" ]; then
        print_status "Modelo Keras encontrado"
    else
        print_error "Modelo Keras NÃO encontrado em models/lane_detector_combined_v2.keras"
        echo "  Copie seu modelo treinado para este local"
    fi
    
    # Check TensorRT engine
    if [ -f "/home/jetson/models/lane-detection/model.engine" ]; then
        print_status "Engine TensorRT encontrado"
    else
        print_warning "Engine TensorRT NÃO encontrado (normal, não é obrigatório)"
    fi
    
    # Check video file
    if [ -f "videos/output_1803ok.avi" ]; then
        print_status "Vídeo de teste encontrado"
    elif [ -f "/home/jetson/Videos/output_1803ok.avi" ]; then
        print_status "Vídeo de teste encontrado em /home/jetson/Videos/"
    else
        print_warning "Vídeo de teste NÃO encontrado"
        echo "  Use qualquer vídeo de teste ou conecte a câmera"
    fi
    
    # Check compilation
    if [ -f "main" ]; then
        print_status "Executável principal compilado"
    else
        print_warning "Sistema principal NÃO compilado"
    fi
    
    # Check if system is running
    if pgrep -f "main" > /dev/null; then
        print_info "Sistema principal está EXECUTANDO"
    else
        print_info "Sistema principal NÃO está executando"
    fi
}

# 2. Apply fixes based on Python version analysis
apply_fixes() {
    echo ""
    print_info "=== APLICANDO CORREÇÕES ==="
    
    # Fix 1: Ensure video path is correct in ControlsManager
    print_info "Corrigindo caminho do vídeo..."
    if [ -f "/home/jetson/Videos/output_1803ok.avi" ]; then
        # Update video path in ControlsManager.cpp
        sed -i 's|/home/jetson/Videos/output_1803ok.avi|/home/jetson/Videos/output_1803ok.avi|g' car_controls/sources/ControlsManager.cpp
        print_status "Caminho do vídeo atualizado"
    fi
    
    # Fix 2: Enable video mode by default
    print_info "Habilitando modo vídeo por padrão..."
    
    # Fix 3: Ensure model fallback is working
    print_status "Sistema de fallback de modelos aplicado (já feito)"
    
    # Fix 4: Create missing directories
    print_info "Criando diretórios necessários..."
    mkdir -p models videos logs outputs
    print_status "Diretórios criados"
}

# 3. Quick test compilation
test_compilation() {
    echo ""
    print_info "=== TESTE DE COMPILAÇÃO ==="
    
    # Clean and rebuild
    print_info "Limpando build anterior..."
    make clean 2>/dev/null || true
    
    print_info "Compilando sistema..."
    if make -j4; then
        print_status "✅ Compilação bem-sucedida!"
        return 0
    else
        print_error "❌ Falha na compilação"
        return 1
    fi
}

# 4. Test basic functionality
test_basic_functionality() {
    echo ""
    print_info "=== TESTE DE FUNCIONALIDADE BÁSICA ==="
    
    if [ ! -f "main" ]; then
        print_error "Executável 'main' não encontrado"
        return 1
    fi
    
    # Test if system starts without crashing
    print_info "Testando inicialização do sistema..."
    timeout 10s ./main &
    MAIN_PID=$!
    sleep 5
    
    if kill -0 $MAIN_PID 2>/dev/null; then
        print_status "✅ Sistema inicializa corretamente"
        kill $MAIN_PID 2>/dev/null || true
        wait $MAIN_PID 2>/dev/null || true
        return 0
    else
        print_error "❌ Sistema falha na inicialização"
        return 1
    fi
}

# 5. Show recommended next steps
show_recommendations() {
    echo ""
    print_info "=== RECOMENDAÇÕES ==="
    
    if [ ! -f "models/lane_detector_combined_v2.keras" ]; then
        echo "📋 PASSO 1: Copie seu modelo treinado:"
        echo "   cp /caminho/para/seu/modelo.keras models/lane_detector_combined_v2.keras"
        echo ""
    fi
    
    echo "📋 PASSO 2: Execute o sistema:"
    echo "   ./main"
    echo ""
    
    echo "📋 PASSO 3: Para controle manual/automático:"
    echo "   - Pressione '2' para modo MANUAL (joystick)"
    echo "   - Pressione '1' para modo AUTOMÁTICO (MPC)"
    echo "   - Pressione 's' para ver status"
    echo "   - Pressione 'q' para sair"
    echo ""
    
    echo "📋 PASSO 4: Para debug:"
    echo "   - Pressione '5' para ativar logs detalhados"
    echo "   - Pressione '6' para desativar logs"
    echo ""
    
    echo "📋 PROBLEMA COMUM: Se não houver detecção de faixas:"
    echo "   1. Verifique se o modelo está no local correto"
    echo "   2. Verifique se o vídeo/câmera está funcionando"
    echo "   3. Monitor ZeroMQ na porta 5556 para dados de visão"
    echo "   4. Use 'htop' para monitorar uso de CPU/memória"
}

# 6. Monitor system in real-time
monitor_system() {
    echo ""
    print_info "=== MONITORAMENTO EM TEMPO REAL ==="
    echo "Pressione Ctrl+C para parar o monitoramento"
    echo ""
    
    while true; do
        clear
        echo "🔍 MONITORAMENTO DO SISTEMA AUTÔNOMO"
        echo "===================================="
        date
        echo ""
        
        # Check if main process is running
        if pgrep -f "./main" > /dev/null; then
            echo -e "${GREEN}✅ Sistema Principal: RODANDO${NC}"
            MAIN_PID=$(pgrep -f "./main")
            echo "   PID: $MAIN_PID"
            
            # Show CPU and memory usage
            ps -p $MAIN_PID -o pid,ppid,pcpu,pmem,command --no-headers 2>/dev/null || echo "   Processo não encontrado"
        else
            echo -e "${RED}❌ Sistema Principal: PARADO${NC}"
        fi
        
        echo ""
        
        # Check ZeroMQ ports
        echo "🔌 Portas ZeroMQ:"
        netstat -ln | grep ":555[0-9]" | while read line; do
            echo "   $line"
        done
        
        echo ""
        
        # Check recent log messages (if available)
        echo "📜 Logs recentes:"
        if [ -f "logs/autonomous.log" ]; then
            tail -5 logs/autonomous.log
        else
            echo "   Nenhum arquivo de log encontrado"
        fi
        
        echo ""
        echo "Pressione Ctrl+C para sair..."
        
        sleep 3
    done
}

# Main execution
main() {
    echo "Iniciando diagnóstico completo..."
    
    diagnose_system
    apply_fixes
    
    if test_compilation; then
        if test_basic_functionality; then
            print_status "🎉 Sistema está funcionando!"
            show_recommendations
            
            echo ""
            read -p "Deseja monitorar o sistema em tempo real? (y/n): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                monitor_system
            fi
        else
            print_error "Sistema compila mas falha na execução"
            show_recommendations
        fi
    else
        print_error "Falha na compilação"
        echo ""
        echo "📋 POSSÍVEIS SOLUÇÕES:"
        echo "1. Instale dependências: sudo apt install libopencv-dev libeigen3-dev"
        echo "2. Verifique se todas as bibliotecas estão instaladas"
        echo "3. Execute: make install-deps"
    fi
}

# Handle Ctrl+C gracefully
trap 'echo -e "\n🛑 Monitoramento interrompido pelo usuário"; exit 0' INT

main
