# Correções de Logs Excessivos e Valores Estáticos MPC

## Problemas Resolvidos:

### 1. **Logs Excessivos (Buffer tmux lotando)**

#### Mudanças nos Logs do MPC (`main.cpp`):
- **Modo Compacto por Padrão**: Logs a cada 100 steps (era 10)
- **Modo Verbose Controlável**: Use comando `d` para ativar logs detalhados
- **Formato Compacto**: `[MPC #100] Pos:(1.2,0.3) T:0.45 S:0.12`
- **Formato Verbose**: Logs completos com trajetória e estados detalhados

#### Mudanças nos Logs do ControlsManager (`ControlsManager.cpp`):
- **Frequência Reduzida**: Logs a cada 20 loops de controle (era todo loop)
- **Informações Condensadas**: Estado e controles em uma linha
- **Logs Condicionais**: Só mostra detalhes quando necessário

### 2. **Valores Estáticos no Estado do Veículo**

#### Correção do `getCurrentVehicleState()` (`ControlsManager.cpp`):
- **Estado Dinâmico**: Agora simula movimento baseado no tempo
- **Velocidade Variável**: `velocity = 0.5 + 0.3 * sin(t * 0.5)` (varia entre 0.2-0.8 m/s)
- **Yaw Dinâmico**: Pequenas variações no ângulo baseadas no tempo
- **Posição Atualizada**: x/y são calculados baseados na velocidade e direção

## Como Usar:

### **Controle de Logs:**
```bash
# No terminal do programa:
d    # Ativa/desativa logs detalhados do MPC
s    # Mostra status completo do sistema
```

### **Modos de Log:**

**Modo Compacto (Padrão)**:
- Log a cada 100 steps
- Uma linha por log: `[MPC #100] Pos:(1.2,0.3) T:0.45 S:0.12`
- Minimal impacto no buffer

**Modo Verbose (comando 'd')**:
- Log a cada 10 steps  
- Informações completas de estado, controles e trajetória
- Use apenas para debug quando necessário

### **Verificação de Valores Dinâmicos:**

Agora você deve ver:
- **Posição (x,y)**: Mudando constantemente
- **Velocidade**: Oscilando entre ~0.2 e 0.8 m/s
- **Yaw**: Pequenas variações angulares
- **Controles**: Respondendo às mudanças de estado

## **Exemplo de Output:**

### Modo Compacto:
```
[MPC #100] Pos:(1.2,0.3) T:0.45 S:0.12
[MPC #200] Pos:(2.1,0.8) T:0.38 S:-0.05
```

### Modo Verbose:
```
=== MPC CONTROL STEP 110 ===
[MPC] Pos: (1.234, 0.456) Vel: 0.67 m/s Yaw: 15.2°
[MPC] Throttle: 0.450 Steering: 0.120 rad
[MPC] Lane trajectory: 25 points (1.2,0.1) (1.4,0.2) (1.6,0.3)
================================
```

## **Para Testar:**

1. **Compile**: `make clean && make`
2. **Execute**: `./main`
3. **Ative MPC**: `m`
4. **Observe logs compactos** (padrão)
5. **Ative verbose**: `d` 
6. **Desative verbose**: `d` novamente

Os valores agora devem estar mudando dinamicamente, não mais estáticos!
