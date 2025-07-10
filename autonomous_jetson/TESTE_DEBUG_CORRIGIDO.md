# 🚨 SOLUÇÃO DO ERRO DE DEBUG

## ❌ Problema identificado:
O VS Code estava tentando usar `/home/jetson/Documents/MPC/main` (caminho remoto) localmente.

## ✅ Correção aplicada:
Mudei todas as configurações para usar `${workspaceFolder}/build/main` (caminho local).

## 🧪 Teste agora:

### 1. Terminal 1 - Setup do túnel:
```bash
./scripts/setup_debug.sh
```

### 2. Terminal 2 - Gdbserver (após setup concluir):
```bash
./scripts/start_gdbserver_only.sh
```

### 3. Terminal 3 - Teste de conexão (opcional):
```bash
./scripts/test_debug_connection.sh
```

### 4. VS Code:
- Pressione `F5`
- Escolha **"Debug Remote Jetson"** (primeira opção)
- Se não funcionar, tente **"Debug Remote Jetson (Alternative)"** (segunda opção)

## 📋 Configurações disponíveis:

1. **"Debug Remote Jetson"** - Configuração principal (launch)
2. **"Debug Remote Jetson (Alternative)"** - Modo attach com logging
3. **"Debug Remote Jetson (Simple)"** - Configuração minimalista

## 🔧 Mudanças importantes:

- ✅ `program`: Agora usa `${workspaceFolder}/build/main` (local)
- ✅ `architecture`: Mudou para `aarch64` (correto para Jetson Nano)
- ✅ `sourceFileMap`: Mapeia diretórios corretamente
- ✅ Uma configuração usa `attach` em vez de `launch`

## 🎯 O que esperar:

- O GDB deve conectar em `localhost:2345`
- Deve carregar símbolos do binário local `build/main`
- Deve mapear arquivos fonte corretamente
- Breakpoints devem funcionar

## 🆘 Se ainda houver erro:

1. Execute o teste: `./scripts/test_debug_connection.sh`
2. Use a configuração "Alternative" (com logging)
3. Verifique o painel OUTPUT -> C/C++ no VS Code
4. Certifique-se de que o gdbserver está rodando no Jetson

**A principal mudança:** Agora o VS Code usa o binário local com símbolos para debug, conectando remotamente ao processo no Jetson!
