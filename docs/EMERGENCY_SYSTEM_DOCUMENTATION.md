# Sistema de Emergência - MPC Integrated System

## Visão Geral

O sistema MPC Integrated agora possui um **sistema de failsafe completo** que garante a parada segura dos motores em caso de emergência ou fechamento da aplicação.

## Características do Sistema de Emergência

### 🛡️ **Parada Automática**
- **No fechamento da aplicação**: Motores são parados automaticamente no destrutor
- **Em sinais do sistema**: SIGINT (Ctrl+C) e SIGTERM disparam parada imediata
- **Antes da limpeza**: Motores são parados ANTES de qualquer cleanup de CUDA/TensorRT

### ⚡ **Comandos de Emergência**

#### Teclas na Janela OpenCV:
- **`e` ou `E`**: Parada imediata dos motores

#### Comandos no Terminal:
- **`e` ou `emergency`**: Parada imediata dos motores
- **`test` ou `emergency-test`**: Teste controlado do sistema de emergência

### 🎯 **Integração com ControlsManager**

O sistema utiliza o método `emergencyStop()` integrado do ControlsManager:
```cpp
g_emergency_controls->emergencyStop();
```

### 📊 **Indicadores Visuais**

#### Na Interface Gráfica:
- **Status de Emergência**: "EMERGENCY: Ready" (verde) ou "EMERGENCY: N/A" (vermelho)
- **Instruções**: "e:EMERGENCY" nas instruções compactas

#### No Terminal:
- Seção "SEGURANÇA" nos comandos disponíveis
- Aviso destacado sobre emergency stop

## Como Usar

### Parada de Emergência Imediata
1. **Na janela OpenCV**: Pressione `e`
2. **No terminal**: Digite `e` e pressione Enter
3. **Ctrl+C**: Dispara parada automática + fechamento

### Teste do Sistema
1. No terminal, digite: `test`
2. Confirme com `s` quando solicitado
3. Sistema executará parada de teste e mostrará resultado

### Retomar Operação
Após uma parada de emergência:
1. Pressione `2` para ativar modo manual
2. Ou use joystick para controle manual

## Fluxo de Segurança

```
1. Evento de Emergência
   ↓
2. emergencyMotorStop() chamada
   ↓
3. ControlsManager.emergencyStop()
   ↓
4. MPC desativado (se ativo)
   ↓
5. Sistema em estado PARADO
   ↓
6. Usuário deve retomar manualmente
```

## Casos de Uso

### ✅ **Funcionamento Normal**
- Aplicação roda normalmente
- Emergency ready (verde) na interface
- Comandos de emergência disponíveis

### ⚠️ **Emergência Ativada**
- Motores parados imediatamente
- MPC desativado automaticamente
- Mensagem de confirmação exibida
- Sistema aguarda comando manual para retomar

### 🔄 **Fechamento da Aplicação**
- Signal handler captura Ctrl+C
- Emergency stop ativado automaticamente
- Cleanup seguro sem CUDA durante signal
- Aplicação encerra de forma segura

## Arquitetura do Failsafe

### Componentes
1. **Global Emergency Pointer**: `g_emergency_controls`
2. **Signal Handler**: Captura SIGINT/SIGTERM
3. **Emergency Function**: `emergencyMotorStop()`
4. **Destructor Safety**: Parada no `~MPCIntegratedApp()`

### Thread Safety
- Variável global atômica: `std::atomic<bool> g_running`
- Ponteiro global para acesso rápido em emergências
- Cleanup ordenado com timers parados primeiro

## Testes Recomendados

1. **Teste básico**: `test` → confirmar → verificar parada
2. **Teste Ctrl+C**: Durante operação normal
3. **Teste de tecla**: Pressionar `e` durante MPC ativo
4. **Teste de fechamento**: Fechar janela OpenCV

## Logs de Emergência

O sistema produz logs claros durante emergências:
```
[EMERGENCY] Stopping all motors...
[EMERGENCY] Motors stopped successfully
*** EMERGENCY STOP ACTIVATED ***
MPC DESATIVADO por parada de emergência
*** Veículo em modo PARADO ***
*** Pressione '2' para retomar controle manual ***
```

## Compatibilidade

- ✅ **Jetson Nano**: ARM64, Ubuntu 20.04
- ✅ **Cross-compilation**: Testado e funcional
- ✅ **Qt Integration**: Compatível com QTimer e QObject
- ✅ **OpenCV**: Integrado com visualização
- ✅ **ZeroMQ**: Não interfere com comunicação

---

## Resumo

**O sistema de emergência garante que os motores sejam parados de forma segura em qualquer situação, protegendo o veículo e proporcionando controle total ao operador.**
