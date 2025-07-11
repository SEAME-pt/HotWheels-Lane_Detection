#!/bin/bash

# Configurações (baseadas no seu script original)
JETSON_USER="michel"
JETSON_HOST="seame-michel.netbird.cloud"
PROJECT_PATH="/home/michel/Documents/other"
LOCAL_PATH="/home/michel-batista/Documents/SEA_ME/HotWheels-Lane_Detection"

# Pasta dedicada para arquivos .dot no host
DOT_DIR="$LOCAL_PATH/dot_analysis"

echo "🔍 Buscando todos os arquivos .dot na Jetson Nano..."

# Criar pasta local para arquivos .dot se não existir
mkdir -p "$DOT_DIR"

# Buscar arquivos .dot na Jetson Nano
echo "Executando busca remota por arquivos .dot..."
ssh $JETSON_USER@$JETSON_HOST "find $PROJECT_PATH -name '*.dot' -type f" > /tmp/dot_files_list.txt

# Verificar se foram encontrados arquivos
if [ ! -s /tmp/dot_files_list.txt ]; then
    echo "❌ Nenhum arquivo .dot encontrado na Jetson Nano"
    echo "💡 Dica: Gere o call graph primeiro:"
    echo "   cd $PROJECT_PATH/build && egypt *.expand > callgraph.dot"
    exit 1
fi

echo "📁 Arquivos .dot encontrados:"
cat /tmp/dot_files_list.txt

# Transferir arquivos .dot para o host
echo "📥 Transferindo arquivos .dot para $DOT_DIR..."
while IFS= read -r dot_file; do
    if [ -n "$dot_file" ]; then
        # Extrair apenas o nome do arquivo
        filename=$(basename "$dot_file")
        echo "  Copiando: $filename"
        rsync --update $JETSON_USER@$JETSON_HOST:"$dot_file" "$DOT_DIR/"
        
        # Obter informações do arquivo transferido
        if [ -f "$DOT_DIR/$filename" ]; then
            filesize=$(stat -c%s "$DOT_DIR/$filename" 2>/dev/null || echo "0")
            lines=$(wc -l < "$DOT_DIR/$filename" 2>/dev/null || echo "0")
            echo "    📊 Tamanho: $filesize bytes, Linhas: $lines"
        fi
    fi
done < /tmp/dot_files_list.txt

# Limpar arquivo temporário
rm -f /tmp/dot_files_list.txt

echo "✅ Transferência concluída!"
echo "📂 Arquivos .dot disponíveis em: $DOT_DIR"

# Contar arquivos transferidos
dot_count=$(ls -1 "$DOT_DIR"/*.dot 2>/dev/null | wc -l)
echo "📊 Total de arquivos .dot transferidos: $dot_count"

# Gerar visualizações para todos os arquivos .dot
if [ $dot_count -gt 0 ] && command -v dot >/dev/null 2>&1; then
    echo ""
    echo "🎨 Gerando visualizações para todos os call graphs..."
    
    cd "$DOT_DIR"
    
    for dot_file in *.dot; do
        if [ -f "$dot_file" ]; then
            echo "  Processando: $dot_file"
            base_name="${dot_file%.dot}"
            
            # Gerar PNG
            echo "    📊 Gerando PNG..."
            dot -Tpng "$dot_file" -o "${base_name}_architecture.png"
            
            # Gerar SVG interativo
            echo "    🌐 Gerando SVG..."
            dot -Tsvg "$dot_file" -o "${base_name}_architecture.svg"
            
            # Gerar PDF para documentação
            echo "    📄 Gerando PDF..."
            dot -Tpdf "$dot_file" -o "${base_name}_architecture.pdf"
            
            # Análise básica do call graph
            lines=$(wc -l < "$dot_file" 2>/dev/null || echo "0")
            relations=$(grep -c " -> " "$dot_file" 2>/dev/null || echo "0")
            functions=$(grep " -> " "$dot_file" 2>/dev/null | cut -d' ' -f1 | sort -u | wc -l || echo "0")
            
            echo "    📈 Análise: $lines linhas, $relations relações, $functions funções"
        fi
    done
    
    echo ""
    echo "🎯 Visualizações geradas para todos os arquivos:"
    ls -1 "$DOT_DIR"/*.png "$DOT_DIR"/*.svg "$DOT_DIR"/*.pdf 2>/dev/null | while read file; do
        echo "  📁 $(basename "$file")"
    done
    
elif [ $dot_count -gt 0 ]; then
    echo "⚠️  Graphviz não encontrado. Instale com: sudo apt-get install graphviz"
    echo "📊 Arquivos .dot salvos, mas visualizações não foram geradas"
fi

# Resumo final
echo ""
echo "📋 Resumo da operação:"
echo "  Arquivos .dot encontrados: $dot_count"
echo "  Pasta de destino: $DOT_DIR"
if command -v dot >/dev/null 2>&1; then
    visualizations=$((dot_count * 3))  # PNG, SVG, PDF para cada .dot
    echo "  Visualizações geradas: $visualizations"
fi
