#!/bin/bash

# Configurações
OUTPUT_FILE="codigo_completo_cpp.txt"
INCLUDE_HIDDEN=false
EXCLUDE_DIRS=".git build obj bin Debug Release scripts outputs docs tests examples"

# Função de ajuda
show_help() {
    echo "Uso: $0 [opções]"
    echo "Opções:"
    echo "  -o, --output FILE    Nome do arquivo de saída (padrão: codigo_completo.txt)"
    echo "  -h, --hidden         Incluir arquivos/diretórios ocultos"
    echo "  -e, --exclude DIRS   Diretórios a excluir (padrão: .git build obj bin Debug Release)"
    echo "  --help               Mostrar esta ajuda"
}

# Processa argumentos da linha de comando
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output)
            OUTPUT_FILE="$2"
            shift 2
            ;;
        -h|--hidden)
            INCLUDE_HIDDEN=true
            shift
            ;;
        -e|--exclude)
            EXCLUDE_DIRS="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Opção desconhecida: $1"
            show_help
            exit 1
            ;;
    esac
done

# Constrói o comando find com exclusões
build_find_command() {
    local pattern="$1"
    local cmd="find . -name \"$pattern\" -type f"
    
    # Adiciona exclusões de diretórios
    for dir in $EXCLUDE_DIRS; do
        cmd="$cmd -not -path \"./$dir/*\""
    done
    
    # Exclui arquivos ocultos se necessário
    if [[ "$INCLUDE_HIDDEN" == false ]]; then
        cmd="$cmd -not -path \"*/.*\""
    fi
    
    echo "$cmd"
}

# Função para processar arquivos
process_files() {
    local pattern="$1"
    local section_name="$2"
    
    local find_cmd=$(build_find_command "$pattern")
    local files=$(eval "$find_cmd" | sort)
    
    if [[ -n "$files" ]]; then
        echo "=== $section_name ===" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        
        echo "$files" | while read -r file; do
            if [[ -n "$file" ]]; then
                echo "=== ARQUIVO: $file ===" >> "$OUTPUT_FILE"
                echo "Tamanho: $(wc -l < "$file") linhas" >> "$OUTPUT_FILE"
                echo "Última modificação: $(stat -c %y "$file" 2>/dev/null || stat -f %Sm "$file" 2>/dev/null)" >> "$OUTPUT_FILE"
                echo "" >> "$OUTPUT_FILE"
                cat "$file" >> "$OUTPUT_FILE"
                echo "" >> "$OUTPUT_FILE"
                echo "=== FIM DO ARQUIVO: $file ===" >> "$OUTPUT_FILE"
                echo "" >> "$OUTPUT_FILE"
                echo "" >> "$OUTPUT_FILE"
            fi
        done
    fi
}

# Remove o arquivo de saída se já existir
rm -f "$OUTPUT_FILE"

# Adiciona cabeçalho
echo "CONCATENAÇÃO COMPLETA DO PROJETO" > "$OUTPUT_FILE"
echo "Gerado em: $(date)" >> "$OUTPUT_FILE"
echo "Diretório: $(pwd)" >> "$OUTPUT_FILE"
echo "Arquivos incluídos: *.cpp, *.hpp, *.h, *.c, *.cc, Makefile" >> "$OUTPUT_FILE"
echo "Diretórios excluídos: $EXCLUDE_DIRS" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"

# Processa diferentes tipos de arquivos
process_files "Makefile" "MAKEFILES"
process_files "makefile" "MAKEFILES (lowercase)"
process_files "*.mk" "ARQUIVOS MAKE"
process_files "*.hpp" "CABEÇALHOS C++ (.hpp)"
process_files "*.h" "CABEÇALHOS C (.h)"
process_files "*.cpp" "CÓDIGO FONTE C++ (.cpp)"
process_files "*.cc" "CÓDIGO FONTE C++ (.cc)"
process_files "*.c" "CÓDIGO FONTE C (.c)"

# Estatísticas finais
total_lines=$(wc -l < "$OUTPUT_FILE")
file_size=$(du -h "$OUTPUT_FILE" | cut -f1)

echo "" >> "$OUTPUT_FILE"
echo "========================================" >> "$OUTPUT_FILE"
echo "ESTATÍSTICAS FINAIS" >> "$OUTPUT_FILE"
echo "Total de linhas: $total_lines" >> "$OUTPUT_FILE"
echo "Tamanho do arquivo: $file_size" >> "$OUTPUT_FILE"
echo "Gerado em: $(date)" >> "$OUTPUT_FILE"

echo "Concatenação concluída!"
echo "Arquivo gerado: $OUTPUT_FILE"
echo "Total de linhas: $total_lines"
echo "Tamanho: $file_size"
