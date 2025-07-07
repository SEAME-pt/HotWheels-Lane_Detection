#!/usr/bin/env python3
"""
Script para extrair arquivos individuais de um arquivo concatenado.

Formato esperado:
=== ARQUIVO: ./caminho/para/arquivo ===
[conteúdo do arquivo]
=== FIM DO ARQUIVO: ./caminho/para/arquivo ===

Uso: python3 extract_files.py [arquivo_concatenado] [diretorio_base]
"""

import sys
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple


def extract_files_from_concatenated(concatenated_file: str, base_dir: str = None) -> Dict[str, str]:
    """
    Extrai arquivos individuais de um arquivo concatenado.
    
    Args:
        concatenated_file: Caminho para o arquivo concatenado
        base_dir: Diretório base onde extrair os arquivos (opcional)
    
    Returns:
        Dicionário com caminho do arquivo como chave e conteúdo como valor
    """
    
    if not os.path.exists(concatenated_file):
        raise FileNotFoundError(f"Arquivo concatenado não encontrado: {concatenated_file}")
    
    files_content = {}
    current_file = None
    current_content = []
    in_file_content = False
    
    # Padrões regex para identificar início e fim de arquivos
    start_pattern = re.compile(r'^=== ARQUIVO: (.+) ===$')
    end_pattern = re.compile(r'^=== FIM DO ARQUIVO: (.+) ===$')
    
    print(f"📖 Lendo arquivo concatenado: {concatenated_file}")
    
    with open(concatenated_file, 'r', encoding='utf-8', errors='ignore') as f:
        line_number = 0
        for line in f:
            line_number += 1
            line = line.rstrip('\n\r')
            
            # Verificar início de arquivo
            start_match = start_pattern.match(line)
            if start_match:
                file_path = start_match.group(1)
                # Remover ./ do início se presente
                if file_path.startswith('./'):
                    file_path = file_path[2:]
                
                current_file = file_path
                current_content = []
                in_file_content = True
                print(f"📁 Encontrado arquivo: {file_path} (linha {line_number})")
                continue
            
            # Verificar fim de arquivo
            end_match = end_pattern.match(line)
            if end_match:
                if current_file and in_file_content:
                    # Juntar o conteúdo e remover linhas vazias no final
                    content = '\n'.join(current_content).rstrip() + '\n' if current_content else ''
                    files_content[current_file] = content
                    print(f"✅ Extraído: {current_file} ({len(current_content)} linhas)")
                
                current_file = None
                current_content = []
                in_file_content = False
                continue
            
            # Adicionar linha ao conteúdo atual, ignorando metadados
            if in_file_content and current_file:
                # Ignorar linhas de metadados (tamanho e data de modificação)
                if not (line.startswith('Tamanho: ') or 
                       line.startswith('Última modificação: ') or
                       line.strip() == ''):  # Também ignora linhas vazias após metadados
                    current_content.append(line)
                elif line.strip() != '' and len(current_content) > 0:  # Só adiciona linhas vazias se já temos conteúdo
                    current_content.append(line)
    
    print(f"\n📊 Total de arquivos extraídos: {len(files_content)}")
    return files_content


def write_files_to_disk(files_content: Dict[str, str], base_dir: str = None, 
                       backup: bool = True, dry_run: bool = False) -> List[str]:
    """
    Escreve os arquivos extraídos no disco.
    
    Args:
        files_content: Dicionário com caminho e conteúdo dos arquivos
        base_dir: Diretório base onde escrever os arquivos
        backup: Se deve criar backup dos arquivos existentes
        dry_run: Se deve apenas simular (não escrever realmente)
    
    Returns:
        Lista de arquivos escritos/que seriam escritos
    """
    
    written_files = []
    
    if base_dir:
        base_path = Path(base_dir)
    else:
        base_path = Path.cwd()
    
    print(f"\n💾 {'Simulando escrita' if dry_run else 'Escrevendo arquivos'} em: {base_path.absolute()}")
    
    for file_path, content in files_content.items():
        full_path = base_path / file_path
        
        # Criar diretórios necessários
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Fazer backup se arquivo existir
        if backup and full_path.exists() and not dry_run:
            backup_path = full_path.with_suffix(full_path.suffix + '.backup')
            print(f"💾 Backup: {full_path} → {backup_path}")
            full_path.rename(backup_path)
        
        if dry_run:
            status = "EXISTENTE" if full_path.exists() else "NOVO"
            print(f"🔍 [DRY-RUN] {status}: {full_path}")
        else:
            # Escrever arquivo
            try:
                with open(full_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                status = "✅ SOBRESCRITO" if backup else "✅ CRIADO"
                print(f"{status}: {full_path}")
                written_files.append(str(full_path))
                
            except Exception as e:
                print(f"❌ ERRO ao escrever {full_path}: {e}")
    
    return written_files


def show_statistics(files_content: Dict[str, str]):
    """Mostra estatísticas dos arquivos extraídos."""
    
    print(f"\n📈 ESTATÍSTICAS:")
    print(f"Total de arquivos: {len(files_content)}")
    
    # Agrupar por extensão
    extensions = {}
    for file_path in files_content.keys():
        ext = Path(file_path).suffix.lower()
        if not ext:
            ext = "(sem extensão)"
        extensions[ext] = extensions.get(ext, 0) + 1
    
    print("\nPor extensão:")
    for ext, count in sorted(extensions.items()):
        print(f"  {ext}: {count} arquivo(s)")
    
    # Agrupar por diretório
    directories = {}
    for file_path in files_content.keys():
        dir_path = str(Path(file_path).parent)
        if dir_path == '.':
            dir_path = "(raiz)"
        directories[dir_path] = directories.get(dir_path, 0) + 1
    
    print("\nPor diretório:")
    for dir_path, count in sorted(directories.items()):
        print(f"  {dir_path}: {count} arquivo(s)")


def main():
    """Função principal do script."""
    
    # Parsing de argumentos simples
    if len(sys.argv) < 2:
        print("❌ Uso: python3 extract_files.py <arquivo_concatenado> [diretorio_base] [opções]")
        print("\nOpções:")
        print("  --dry-run    : Simula a operação sem escrever arquivos")
        print("  --no-backup  : Não cria backup dos arquivos existentes")
        print("  --stats-only : Apenas mostra estatísticas")
        print("\nExemplo:")
        print("  python3 extract_files.py codigo_completo_cppBack")
        print("  python3 extract_files.py codigo_completo_cppBack /path/to/project --dry-run")
        sys.exit(1)
    
    concatenated_file = sys.argv[1]
    base_dir = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith('--') else None
    
    # Opções
    dry_run = '--dry-run' in sys.argv
    no_backup = '--no-backup' in sys.argv
    stats_only = '--stats-only' in sys.argv
    
    try:
        # Extrair arquivos
        files_content = extract_files_from_concatenated(concatenated_file, base_dir)
        
        if not files_content:
            print("⚠️  Nenhum arquivo encontrado no formato esperado!")
            return
        
        # Mostrar estatísticas
        show_statistics(files_content)
        
        if stats_only:
            print("\n📊 Apenas estatísticas solicitadas. Finalizando.")
            return
        
        # Confirmar antes de escrever (a menos que seja dry-run)
        if not dry_run:
            response = input(f"\n❓ Deseja {'sobrescrever' if base_dir else 'extrair'} {len(files_content)} arquivos? [y/N]: ")
            if response.lower() not in ['y', 'yes', 's', 'sim']:
                print("❌ Operação cancelada pelo usuário.")
                return
        
        # Escrever arquivos
        written_files = write_files_to_disk(
            files_content, 
            base_dir, 
            backup=not no_backup, 
            dry_run=dry_run
        )
        
        if dry_run:
            print(f"\n🔍 DRY-RUN completo. {len(files_content)} arquivos seriam processados.")
        else:
            print(f"\n✅ Operação concluída! {len(written_files)} arquivos processados.")
            
    except Exception as e:
        print(f"❌ ERRO: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
