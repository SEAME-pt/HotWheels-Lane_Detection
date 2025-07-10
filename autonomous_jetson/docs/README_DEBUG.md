# Debug Remoto - Jetson Nano

## Processo Simplificado

### 1. Compilação e Deploy
```bash
# No VS Code: Ctrl+Shift+P -> "Tasks: Run Task" -> "Deploy Binaries"
# Ou manual:
cd build
~/new_qtjetson/qt5.15/bin/qmake ../MPC.pro
make -j$(nproc)
./scripts/deploy_rsync.sh /home/michel/Documents/other
```

### 2. Debug Remoto - Método Simples

**Terminal 1:**
```bash
./scripts/setup_debug.sh
```

**Terminal 2 (após setup_debug rodar):**
```bash
./scripts/start_gdbserver_only.sh
```

**VS Code:**
- Pressione `F5` ou `Ctrl+Shift+D`
- Selecione uma das configurações:
  - **"Debug Remote Jetson"** (padrão)
  - **"Debug Remote Jetson (Alternative)"** (se houver problemas)

### 3. Debug Remoto - Método Manual

**Terminal 1 - Túnel SSH:**
```bash
ssh -L 2345:localhost:2345 -N jetson@hotwheels-car.netbird.cloud
```

**Terminal 2 - GDB Server:**
```bash
ssh jetson@hotwheels-car.netbird.cloud "cd /home/jetson/Documents/MPC && gdbserver localhost:2345 ./main"
```

**VS Code:**
- Pressione `F5` ou `Ctrl+Shift+D`
- Escolha a configuração de debug desejada

### Configuração do Jetson
- Host: `hotwheels-car.netbird.cloud`
- Usuário: `jetson`
- Pasta: `/home/jetson/Documents/MPC`
- Porta Debug: `2345`

### Scripts Disponíveis
- `scripts/deploy_rsync.sh` - Deploy via rsync
- `scripts/setup_debug.sh` - Configuração automática do debug
- `scripts/start_gdbserver_only.sh` - Inicia apenas o gdbserver

## Troubleshooting

### Problemas Comuns

**Erro "Unable to connect to gdbserver":**
1. Verifique se o túnel SSH está ativo: `ps aux | grep "ssh.*2345"`
2. Verifique se o gdbserver está rodando no Jetson
3. Tente a configuração alternativa: "Debug Remote Jetson (Alternative)"

**Erro de arquitetura:**
- Use a configuração alternativa que inclui `targetArchitecture: "arm"`
- Verifique se `gdb-multiarch` está instalado localmente

**Debug não para nos breakpoints:**
- Certifique-se de que o binário foi compilado com símbolos de debug (`-g`)
- Verifique se o caminho do programa está correto no `launch.json`

**Performance lenta:**
- Use `stopAtEntry: false` para não parar no main
- Desabilite logging verbose se não necessário
- `scripts/setup_debug.sh` - Configuração completa de debug
- `scripts/start_gdbserver_only.sh` - Apenas gdbserver
