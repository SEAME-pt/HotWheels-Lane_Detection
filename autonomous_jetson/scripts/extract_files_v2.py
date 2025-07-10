#!/usr/bin/env python3
"""
Script para extrair arquivos de um arquivo concatenado.
Preserva exatamente a formatação original, removendo apenas linhas de metadados específicas.
"""
import os
import sys

def extract_files(input_file, output_dir=None, dry_run=False):
    """
    Extrai arquivos de um arquivo concatenado preservando formatação original.
    
    Args:
        input_file: Path do arquivo concatenado
        output_dir: Diretório de saída (opcional)
        dry_run: Se True, apenas mostra o que seria feito
    """
    if not os.path.exists(input_file):
        print(f"Erro: Arquivo {input_file} não encontrado")
        return False
    
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Processa linha por linha
    files = []
    current_file = None
    current_content = []
    i = 0
    
    while i < len(lines):
        line = lines[i].rstrip('\n\r')
        
        if line.startswith('=== ARQUIVO: '):
            # Novo arquivo começando
            if current_file:
                files.append((current_file, current_content))
            
            # Extrai o path do arquivo
            current_file = line.replace('=== ARQUIVO: ', '').replace(' ===', '').strip()
            if current_file.startswith('./'):
                current_file = current_file[2:]
            current_content = []
            
            # Pula linha de metadados (tamanho e data)
            j = i + 1
            while j < len(lines):
                next_line = lines[j].rstrip('\n\r')
                # Se é metadado, pula
                if (next_line.startswith('Tamanho: ') and 'linhas' in next_line) or \
                   (next_line.startswith('Última modificação: ') and '+' in next_line):
                    j += 1
                    continue
                else:
                    break
            i = j - 1  # -1 porque será incrementado no final do loop
            
        elif line.startswith('=== FIM DO ARQUIVO: '):
            # Fim do arquivo atual
            if current_file:
                files.append((current_file, current_content))
                current_file = None
                current_content = []
                
        elif current_file is not None:
            # Adiciona linha preservando exatamente como está (incluindo vazias)
            current_content.append(lines[i])
        
        i += 1
    
    # Se ainda há um arquivo em processamento
    if current_file:
        files.append((current_file, current_content))
    
    print(f"Encontrados {len(files)} arquivos para extrair")
    
    for file_path, file_lines in files:
        # Remove linha vazia final se existir
        while file_lines and file_lines[-1].strip() == '':
            file_lines.pop()
        
        if dry_run:
            print(f"[DRY RUN] Extrairia: {file_path} ({len(file_lines)} linhas)")
        else:
            # Determina o path de saída
            if output_dir:
                full_path = os.path.join(output_dir, file_path)
            else:
                full_path = file_path
            
            # Cria diretórios se necessário
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            
            # Escreve o arquivo preservando formatação original
            with open(full_path, 'w', encoding='utf-8') as f:
                for line in file_lines:
                    f.write(line)
            
            print(f"Extraído: {full_path}")
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Uso: python3 extract_files_v2.py <arquivo_concatenado> [diretorio_saida] [--dry-run]")
        return
    
    input_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    dry_run = '--dry-run' in sys.argv
    
    extract_files(input_file, output_dir, dry_run)

if __name__ == "__main__":
    main()
