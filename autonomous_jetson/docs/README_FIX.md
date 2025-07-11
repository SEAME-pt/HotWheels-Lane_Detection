# 🚗 Correção do Sistema Autônomo - Jetson Nano

## Problema Identificado

Após análise comparativa entre o código Python (funcionando) e C++ (não funcionando), identifiquei os seguintes problemas:

### 1. **Pipeline Complexo Demais**
- **Python**: Pipeline simples e direto: `camera → keras → polyfit → MPC → control`
- **C++**: Pipeline complexo com múltiplas threads e ZeroMQ entre componentes

### 2. **Dependência TensorRT**
- **Problema**: Sistema tentando carregar `/home/jetson/models/lane-detection/model.engine` que pode não existir
- **Solução**: Implementei fallback para ONNX → Keras

### 3. **Dados de Visão Não Chegando ao MPC**
- **Problema**: Múltiplas threads com sincronização complexa
- **Solução**: Simplificação do fluxo de dados

## Soluções Implementadas

### ✅ 1. Script de Diagnóstico e Correção
```bash
./fix_autonomous.sh
```

### ✅ 2. Fallback de Modelos
- Agora tenta TensorRT → ONNX → Keras automaticamente
- Compatível com seu modelo existente

### ✅ 3. Teste Simplificado
```bash
make -f Makefile.test
./test_autonomous_simple models/lane_detector_combined_v2.onnx
```

## Passos para Resolução

### Passo 1: Execute o Diagnóstico
```bash
./fix_autonomous.sh
```
Este script vai:
- ✅ Identificar problemas no sistema atual
- ✅ Aplicar correções automáticas
- ✅ Testar a compilação
- ✅ Verificar funcionalidade básica
- ✅ Mostrar próximos passos

### Passo 2: Prepare o Modelo
Se você ainda não tem o modelo em formato compatível:

**Opção A - Usar Keras diretamente:**
```bash
cp /seu/caminho/lane_detector_combined_v2.keras models/
```

**Opção B - Converter para ONNX (recomendado):**
```bash
python3 convert_model.py
```

### Passo 3: Execute o Sistema
```bash
# Sistema completo
./main

# OU teste simplificado
./test_autonomous_simple
```

### Passo 4: Controle do Sistema
No terminal do sistema, você pode:
- `1` - Ativar modo AUTOMÁTICO (MPC)
- `2` - Ativar modo MANUAL (joystick) 
- `5` - Ativar logs detalhados
- `s` - Mostrar status
- `q` - Sair

## Diferenças Críticas Python vs C++

| Aspecto | Python (Funcionando) | C++ (Original) | C++ (Corrigido) |
|---------|---------------------|----------------|-----------------|
| **Pipeline** | Simples, single-thread | Complexo, multi-thread | Simplificado |
| **Modelo** | Keras direto | TensorRT obrigatório | TensorRT → ONNX → Keras |
| **Processamento** | Síncrono | Assíncrono (ZeroMQ) | Híbrido |
| **Rate** | 20Hz direto | Múltiplas rates | 20Hz unificado |

## Monitoramento

Para verificar se o sistema está funcionando:

```bash
# Monitor em tempo real
./fix_autonomous.sh  # Escolha opção de monitoramento

# Verificar portas ZeroMQ
netstat -ln | grep 555

# Verificar processos
ps aux | grep main
```

## Debugging

### Se não há detecção de faixas:
1. ✅ Verifique se o modelo está carregando: veja logs de inicialização
2. ✅ Verifique se há dados na porta 5556: `nc -l 5556`
3. ✅ Ative logs detalhados: pressione `5` no sistema

### Se MPC não está controlando:
1. ✅ Verifique se há waypoints: veja logs com `5`
2. ✅ Verifique modo: deve estar em `1` (AUTOMÁTICO)
3. ✅ Verifique se `m_autonomousMode = true`

### Se há problemas de memória:
1. ✅ Use `htop` para monitorar
2. ✅ Pressione `7` para limpeza de memória
3. ✅ Reinicie o sistema se necessário

## Próximos Passos Após Funcionar

1. **Validar Detecção**: Confirme que as faixas são detectadas corretamente
2. **Calibrar MPC**: Ajuste parâmetros no `MPCPlanner.cpp`
3. **Testar Hardware**: Conecte servos/motores e teste comandos
4. **Otimizar**: Habilite GPU acceleration após validação básica

## Arquivos Importantes

- `fix_autonomous.sh` - Script principal de diagnóstico e correção
- `test_autonomous_simple.cpp` - Versão simplificada para teste
- `car_controls/sources/ControlsManager.cpp` - Sistema principal (corrigido)
- `car_controls/sources/inference/CameraStreamer.cpp` - Captura e inferência (corrigido)

## Suporte

Se ainda houver problemas após executar `./fix_autonomous.sh`, verifique:

1. **Logs detalhados**: Pressione `5` no sistema rodando
2. **Arquivos de log**: `logs/` (se existir)
3. **Status das portas**: `netstat -ln | grep 555`
4. **Uso de recursos**: `htop` ou `nvidia-smi`

---

**🎯 Objetivo**: Fazer o carro andar de forma autônoma como no CARLA

**🔑 Chave**: Simplificar o pipeline complexo para seguir o padrão Python que funciona
