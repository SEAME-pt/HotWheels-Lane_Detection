#!/bin/bash
# Script simples para extrair arquivos de um arquivo concatenado
# Uso: ./extract_files_simple.sh <arquivo_concatenado> [diretorio_base]

set -e

# Verificar argumentos
if [ $# -lt 1 ]; then
    echo "❌ Uso: $0 <arquivo_concatenado> [diretorio_base]"
    echo ""
    echo "Exemplos:"
    echo "  $0 codigo_completo_cppBack"
    echo "  $0 codigo_completo_cppBack /path/to/project"
    exit 1
fi

CONCATENATED_FILE="$1"
BASE_DIR="${2:-.}"  # Usar diretório atual se não especificado

# Verificar se arquivo existe
if [ ! -f "$CONCATENATED_FILE" ]; then
    echo "❌ Arquivo não encontrado: $CONCATENATED_FILE"
    exit 1
fi

echo "📖 Processando arquivo: $CONCATENATED_FILE"
echo "📁 Diretório base: $BASE_DIR"
echo ""

# Variáveis para controle
current_file=""
in_file=false
files_processed=0

# Criar diretório base se não existir
mkdir -p "$BASE_DIR"

# Processar linha por linha
while IFS= read -r line; do
    # Verificar início de arquivo
    if [[ "$line" =~ ^===\ ARQUIVO:\ (.+)\ ===$ ]]; then
        current_file="${BASH_REMATCH[1]}"
        # Remover ./ do início se presente
        current_file="${current_file#./}"
        
        # Caminho completo
        full_path="$BASE_DIR/$current_file"
        
        # Criar diretório pai
        mkdir -p "$(dirname "$full_path")"
        
        # Fazer backup se arquivo existir
        if [ -f "$full_path" ]; then
            cp "$full_path" "$full_path.backup"
            echo "💾 Backup: $full_path → $full_path.backup"
        fi
        
        # Limpar arquivo (preparar para novo conteúdo)
        > "$full_path"
        
        in_file=true
        echo "📁 Extraindo: $current_file"
        continue
    fi
    
    # Verificar fim de arquivo
    if [[ "$line" =~ ^===\ FIM\ DO\ ARQUIVO:\ (.+)\ ===$ ]]; then
        if [ "$in_file" = true ]; then
            files_processed=$((files_processed + 1))
            echo "✅ Concluído: $current_file"
        fi
        in_file=false
        current_file=""
        continue
    fi
    
    # Adicionar linha ao arquivo atual
    if [ "$in_file" = true ] && [ -n "$current_file" ]; then
        echo "$line" >> "$BASE_DIR/$current_file"
    fi
    
done < "$CONCATENATED_FILE"

echo ""
echo "✅ Processamento concluído!"
echo "📊 Total de arquivos extraídos: $files_processed"
echo "📁 Arquivos salvos em: $BASE_DIR"

# Mostrar alguns arquivos extraídos como exemplo
echo ""
echo "📋 Alguns arquivos extraídos:"
find "$BASE_DIR" -type f \( -name "*.cpp" -o -name "*.hpp" -o -name "*.h" \) | head -10 | while read -r file; do
    echo "  - $file"
done

if [ $files_processed -gt 10 ]; then
    echo "  ... e mais $((files_processed - 10)) arquivo(s)"
fi
