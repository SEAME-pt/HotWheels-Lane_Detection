# Correções dos Problemas MPC + Camera

## Problemas Identificados:

1. **Camera/Video para quando MPC é ativado**
2. **Threads não param quando sai (SIGINT/Ctrl+C)**

## Correções Implementadas:

### 1. Correção das Threads não Pararem (SIGINT)

#### `main.cpp`:
- **Signal Handler melhorado**: Aumentou timeout para 500ms e adicionou `std::exit(0)` como failsafe
- **Shutdown Timer melhorado**: Agora reseta o `integrated_app` antes de sair
- **Global flag**: Todas as threads agora verificam `g_running`

#### `ControlsManager.cpp`:
- **Loops de threads**: Adicionada verificação de `extern std::atomic<bool> g_running` em:
  - `autonomousControlLoop()`
  - Thread do joystick (`m_subscriberJoystickThread`)
- **Destructor melhorado**: Adicionados timeouts de 3 segundos para `wait()` e `terminate()` como fallback

#### `CameraStreamer.cpp`:
- **CaptureLoop**: Adicionada verificação de `extern std::atomic<bool> g_running`
- **Loop principal**: Agora usa `while(m_running && g_running)`

### 2. Correção da Camera Parar Durante MPC

#### `ControlsManager.cpp`:
- **Comentou `showVisionDebug()`**: Esta função estava competindo com o streaming da câmera
- **Frequência reduzida**: Autonomous control loop agora não interfere tanto

#### `main.cpp`:
- **Frequência MPC reduzida**: De 20Hz (50ms) para 10Hz (100ms) para reduzir interferência
- **Debug melhorado**: Contador de frames da câmera para monitorar se está funcionando
- **Visualização melhorada**: Status da câmera mostra se está recebendo frames

## Como Testar:

1. **Compile o projeto**:
   ```bash
   cd car_controls
   make clean && make
   cd ..
   make
   ```

2. **Execute**:
   ```bash
   ./main
   ```

3. **Teste Camera Durante MPC**:
   - Ative MPC com 'm'
   - Verifique se "Cam: REAL #número" continua incrementando na visualização
   - Verifique se o video continua passando na janela "Camera Feed"

4. **Teste Saída com SIGINT**:
   - Pressione Ctrl+C
   - Deve mostrar "Received signal 2. Shutting down gracefully..."
   - Programa deve terminar em alguns segundos (máximo 4-5 segundos)

## Logs Importantes:

- `[main] Shutting down due to signal...` - Indica que o signal foi recebido
- `Cam: REAL #número` - Contador deve continuar incrementando durante MPC
- `Running autonomous control loop...` - Deve aparecer menos frequentemente agora
- `[~MPCIntegratedApp] Cleanup complete` - Indica limpeza bem-sucedida

## Notas:

- Se ainda houver problemas, pode ser necessário ajustar os timeouts ou adicionar mais verificações de `g_running`
- O problema pode estar relacionado ao OpenCV + CUDA + X11 forwarding via SSH
- Para testes remotos via SSH, considere usar `export DISPLAY=:0` se necessário
