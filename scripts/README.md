# Scripts de Extração de Arquivos Concatenados

Este diretório contém scripts para extrair arquivos individuais de um arquivo concatenado no formato usado pelo projeto HotWheels.

## Formato do Arquivo Concatenado

O arquivo concatenado segue o formato:

```
CONCATENAÇÃO COMPLETA DO PROJETO
Gerado em: [timestamp]
Diretório: [diretório_base]
...

=== ARQUIVO: ./caminho/para/arquivo ===
Tamanho: X linhas
Última modificação: [timestamp]

[conteúdo do arquivo]

=== FIM DO ARQUIVO: ./caminho/para/arquivo ===
```

## Scripts Disponíveis

### 1. `extract_files.py` (Recomendado)

Script Python com funcionalidades avançadas.

#### Uso Básico:
```bash
# Mostrar apenas estatísticas
python3 scripts/extract_files.py codigo_completo_cppBack --stats-only

# Simulação (dry-run) - ver o que seria feito
python3 scripts/extract_files.py codigo_completo_cppBack --dry-run

# Extrair arquivos no diretório atual
python3 scripts/extract_files.py codigo_completo_cppBack

# Extrair arquivos em diretório específico
python3 scripts/extract_files.py codigo_completo_cppBack /path/to/target/directory
```

#### Opções:
- `--dry-run`: Simula a operação sem escrever arquivos
- `--no-backup`: Não cria backup dos arquivos existentes
- `--stats-only`: Apenas mostra estatísticas dos arquivos

#### Características:
- ✅ Cria backup automático de arquivos existentes
- ✅ Criação automática de diretórios
- ✅ Estatísticas detalhadas
- ✅ Modo de simulação (dry-run)
- ✅ Confirmação antes de sobrescrever
- ✅ Tratamento robusto de erros
- ✅ Encoding UTF-8 com fallback

### 2. `extract_files_simple.sh`

Script shell mais simples e rápido.

#### Uso:
```bash
# Extrair no diretório atual
./scripts/extract_files_simple.sh codigo_completo_cppBack

# Extrair em diretório específico
./scripts/extract_files_simple.sh codigo_completo_cppBack /path/to/target/directory
```

#### Características:
- ✅ Rápido e simples
- ✅ Cria backup automático
- ✅ Criação automática de diretórios
- ✅ Resumo da operação
- ⚠️  Menos verificações de erro

## Exemplos Práticos

### Verificar conteúdo do arquivo concatenado:
```bash
python3 scripts/extract_files.py codigo_completo_cppBack --stats-only
```

### Testar extração sem modificar arquivos:
```bash
python3 scripts/extract_files.py codigo_completo_cppBack --dry-run
```

### Extrair todos os arquivos (com backup):
```bash
python3 scripts/extract_files.py codigo_completo_cppBack
```

### Extrair sem fazer backup:
```bash
python3 scripts/extract_files.py codigo_completo_cppBack --no-backup
```

### Extrair para um novo diretório:
```bash
mkdir -p ~/extracted_project
python3 scripts/extract_files.py codigo_completo_cppBack ~/extracted_project
```

## Segurança

### Backups Automáticos
Por padrão, os scripts criam backup de arquivos existentes:
- `arquivo.cpp` → `arquivo.cpp.backup`
- `header.hpp` → `header.hpp.backup`

### Verificações
- Verificação de existência do arquivo de entrada
- Criação automática de diretórios necessários
- Tratamento de caracteres especiais no conteúdo
- Preservação de permissões de arquivos

## Estatísticas do Arquivo Atual

Baseado no `codigo_completo_cppBack`:

**Total de arquivos**: 58
- **29** arquivos `.cpp`
- **28** arquivos `.hpp`
- **1** Makefile

**Diretórios principais**:
- `car_controls/includes/`: 12 headers
- `car_controls/sources/`: 9 implementações
- `car_controls/includes/inference/`: 10 headers de inferência
- `car_controls/sources/inference/`: 8 implementações de inferência
- `car_controls/tests/unit/`: 5 testes unitários
- `ZeroMQ/`: 4 arquivos de comunicação

## Solução de Problemas

### Erro de permissão:
```bash
chmod +x scripts/extract_files.py
chmod +x scripts/extract_files_simple.sh
```

### Arquivo não encontrado:
Verifique se o caminho está correto:
```bash
ls -la codigo_completo_cppBack
```

### Problemas de encoding:
O script Python trata automaticamente problemas de encoding com fallback.

### Restaurar backups:
```bash
# Restaurar um arquivo específico
mv arquivo.cpp.backup arquivo.cpp

# Restaurar todos os backups
find . -name "*.backup" | while read backup; do
    original="${backup%.backup}"
    mv "$backup" "$original"
done
```

## Criação de Arquivo Concatenado

Se precisar criar um novo arquivo concatenado, use o script `concatenador.sh`:

```bash
./concatenador.sh
```

Isso gerará um novo `codigo_completo_cpp.txt` com todos os arquivos atuais do projeto.
