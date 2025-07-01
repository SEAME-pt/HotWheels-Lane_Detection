# 🛡️ **CONFIGURAÇÕES DE SEGURANÇA PARA SERVO FRÁGIL**

## ⚠️ **ATENÇÃO CRÍTICA**
O servomotor do sistema é **EXTREMAMENTE FRÁGIL** e pode quebrar facilmente com:
- Movimentos muito rápidos
- Ângulos muito grandes
- Mudanças bruscas de direção
- Uso repetitivo excessivo

## 🔧 **Proteções Implementadas**

### **1. Limitação de Velocidade (main.cpp)**
```cpp
// Velocidade MUITO reduzida para proteger o servo
#define DEFAULT_CONSTANT_SPEED_KMH 0.5 // Apenas 0.5 km/h (muito lento)
#define DEFAULT_CONSTANT_THROTTLE 0.08  // Apenas 8% de potência
#define MAX_SAFE_SPEED_KMH 2.0         // Máximo 2.0 km/h
```

### **2. Proteção de Ângulo (ControlsManager.cpp)**
```cpp
// Ângulo limitado a ±15° (muito menor que os ±45° originais)
int steer_angle = static_cast<int>(std::clamp(control.steer * 15, -15.0, 15.0));

// Rate limiting: máximo 2° por iteração
int max_servo_change = 2;
```

### **3. Proteção Temporal (EngineController.cpp)**
```cpp
// Limite temporal: mínimo 50ms entre comandos
auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_servo_time).count();
if(elapsed < 50) {
    // Muito rápido, usar último ângulo
    angle = last_angle;
}

// Rate limiting por mudança: máximo 3° por comando
int max_change = 3;
```

### **4. Soft Start Extremo (ControlsManager.hpp)**
```cpp
// Aceleração muito gradual
double max_throttle_change_per_step = 0.002; // Apenas 0.2% por iteração
double initial_throttle_limit = 0.02;        // Máximo 2% inicial
double warmup_duration_seconds = 8.0;        // 8 segundos de aquecimento
```

---

## 📊 **Comparação: Antes vs Agora**

| Parâmetro | Valor Original | Valor Atual | Proteção |
|-----------|----------------|-------------|----------|
| **Velocidade máxima** | 7 km/h | 0.5 km/h | 🛡️ 14x mais lento |
| **Throttle máximo** | 50% | 20% | 🛡️ 2.5x menor |
| **Ângulo máximo** | ±45° | ±15° | 🛡️ 3x menor |
| **Rate limit ângulo** | ±5° | ±2° | 🛡️ 2.5x mais suave |
| **Delay entre comandos** | 0ms | 50ms | 🛡️ Proteção temporal |
| **Soft start inicial** | 5% | 2% | 🛡️ 2.5x mais suave |
| **Duração aquecimento** | 3s | 8s | 🛡️ 2.7x mais longo |

---

## 🎯 **Valores Seguros para Teste**

### **Para testes iniciais (MUITO seguro):**
```cpp
DEFAULT_CONSTANT_SPEED_KMH 0.3     // 0.3 km/h - crawling speed
DEFAULT_CONSTANT_THROTTLE 0.05     // 5% potência
```

### **Para testes normais (seguro):**
```cpp
DEFAULT_CONSTANT_SPEED_KMH 0.5     // 0.5 km/h - valor atual
DEFAULT_CONSTANT_THROTTLE 0.08     // 8% potência
```

### **Para testes avançados (ainda seguro):**
```cpp
DEFAULT_CONSTANT_SPEED_KMH 1.0     // 1 km/h - limite recomendado
DEFAULT_CONSTANT_THROTTLE 0.12     // 12% potência
```

---

## 🚨 **NUNCA EXCEDER:**
- **Velocidade:** 2 km/h
- **Throttle:** 20%
- **Ângulo:** ±15°
- **Tempo entre comandos:** < 50ms

---

## 🔍 **Monitoramento em Tempo Real**

O sistema agora exibe logs de proteção:
```
[SERVO PROTECTION] Limitando mudança de 8° para 2°
[SERVO] Ângulo: 5° (PWM: 350)
[SOFT START] Elapsed: 2.2s, Target: 8.00%, Limited: 2.32%
```

---

## 📋 **Checklist de Segurança**

Antes de cada teste:
- [ ] Verificar se DEFAULT_CONSTANT_SPEED_KMH ≤ 1.0
- [ ] Verificar se DEFAULT_CONSTANT_THROTTLE ≤ 0.12
- [ ] Soft start habilitado (`soft` para verificar)
- [ ] Emergency stop funcionando (`test` para verificar)
- [ ] Servo centralizado antes de iniciar

---

## 🛠️ **Como Alterar Configurações**

1. **Para mudanças rápidas:** Edite as macros no início de `main.cpp`
2. **Para proteções do servo:** Edite `EngineController.cpp`
3. **Para soft start:** Edite `ControlsManager.hpp`
4. **Sempre recompilar:** `make clean && make`

---

## ⚡ **Comando de Emergência**
**Sempre tenha o dedo na tecla `e` para EMERGENCY STOP imediato!**
