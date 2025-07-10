# Organização Final - Debug Remoto

## Arquivos de Configuração VS Code
- `.vscode/launch.json` - Configuração "Debug Remote Jetson"
- `.vscode/tasks.json` - Tasks para compilação e deploy

## Scripts (pasta scripts/)
- `deploy_binaries.sh` - Deploy automático dos binários
- `start_remote_debug.sh` - Configuração completa do debug remoto

## Documentação (pasta docs/)
- `README_DEBUG.md` - Instruções simplificadas

## Processo Completo
1. **F5 no VS Code** → Executa automaticamente:
   - QMake Configure
   - Make Build  
   - Deploy Binaries
   - Prepara debug remoto

2. **Manual**: `./scripts/start_remote_debug.sh` + F5

## Configurações
- **Jetson**: `hotwheels-car.netbird.cloud`
- **Usuário**: `jetson`
- **Pasta**: `/home/jetson/Documents/MPC`
- **Porta Debug**: `2345`
