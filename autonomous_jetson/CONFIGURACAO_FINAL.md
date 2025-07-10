# Configuração Final - Debug Remoto

## ✅ O que foi corrigido:

1. **Arquivos de configuração ajustados:**
   - `launch.json` com 3 configurações diferentes
   - `tasks.json` com task de setup atualizada
   - Mapeamento de diretórios corrigido

2. **Scripts atualizados:**
   - `setup_debug.sh` - Setup automático 
   - `start_gdbserver_only.sh` - Apenas gdbserver
   - `diagnose_debug.sh` - Diagnóstico completo

3. **Configurações disponíveis no VS Code:**
   - **"Debug Remote Jetson"** - Configuração padrão com sourceFileMap
   - **"Debug Remote Jetson (Alternative)"** - Com logging verbose
   - **"Debug Remote Jetson (Simple)"** - Configuração minimalista

## 🚀 Como usar agora:

### Método 1 - Automático (Recomendado)
```bash
# Terminal 1
./scripts/setup_debug.sh

# Terminal 2 (após setup rodar)
./scripts/start_gdbserver_only.sh

# VS Code: F5 -> Escolher configuração
```

### Método 2 - Diagnóstico primeiro
```bash
# Verificar se tudo está OK
./scripts/diagnose_debug.sh

# Então seguir Método 1
```

### Método 3 - Manual
```bash
# Terminal 1 - Túnel SSH
ssh -L 2345:localhost:2345 -N jetson@hotwheels-car.netbird.cloud

# Terminal 2 - GDBServer
ssh jetson@hotwheels-car.netbird.cloud "cd /home/jetson/Documents/MPC && gdbserver localhost:2345 ./main"

# VS Code: F5
```

## 🔧 Se ainda houver problemas:

1. **Teste a configuração "Simple" primeiro**
2. **Verifique o output do diagnóstico**
3. **Use a configuração "Alternative" para logs detalhados**
4. **Certifique-se de que o gdbserver está rodando antes de F5**

## 📁 Arquivos principais:
- `.vscode/launch.json` - 3 configurações de debug
- `.vscode/tasks.json` - Tasks de build e deploy
- `scripts/diagnose_debug.sh` - Diagnóstico completo
- `scripts/setup_debug.sh` - Setup automático
- `scripts/start_gdbserver_only.sh` - Apenas gdbserver
