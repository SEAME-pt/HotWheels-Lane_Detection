# 🔧 Proteção do Servo (Steering) - COMPONENTE FRÁGIL

## ⚠️ **IMPORTANTE: Servo vs Motor**

### **🔴 SERVO (Steering/Direção) - FRÁGIL**
- **Componente**: 1x Servomotor I2C para direção
- **Função**: Controla o ângulo das rodas dianteiras (steering)
- **Características**: Extremamente frágil, quebra facilmente
- **Proteções implementadas**: Múltiplas camadas de segurança
- **Cuidados especiais**: Rate limiting, amplitude limitada, tempo entre comandos

### **🟢 MOTOR (Throttle/Aceleração) - ROBUSTO**
- **Componente**: 2x Motores DC I2C para aceleração
- **Função**: Controla a velocidade das rodas traseiras (throttle)
- **Características**: Robustos, aguentam comandos extremos
- **Proteções**: Apenas as básicas de segurança
- **Uso**: Pode receber valores de 0% a 50% sem problemas

### **🟢 MOTOR (Throttle/Aceleração) - ROBUSTO**  
- **Componentes**: 2x Motores I2C para aceleração individual das rodas
- **Características**: Robustos, aceitam inputs extremos
- **Proteções**: Apenas proteções básicas de segurança
- **Funcionamento**: Normal, sem limitações especiais

---

## 🛡️ **Proteções Implementadas para o SERVO**

### **1. Limitação de Amplitude**
```cpp
static const int SAFE_MAX_ANGLE = 20; // Máximo ±20° (ao invés de ±45°)
```
- **Objetivo**: Proteger mecanismo do servo de movimentos extremos
- **Original**: ±45°
- **Protegido**: ±20°

### **2. Rate Limiting Temporal**
```cpp
if(elapsed < 80) return; // Mínimo 80ms entre comandos
```
- **Objetivo**: Evitar comandos muito rápidos que podem danificar o servo
- **Frequência máxima**: ~12 Hz (ao invés de 50+ Hz)

### **3. Rate Limiting por Mudança**
```cpp
int max_change = 5; // Máximo 5° por comando
```
- **Objetivo**: Evitar mudanças bruscas de direção
- **Comportamento**: Transições suaves e graduais

### **4. Logs de Proteção**
- Sistema informa quando limita comandos para proteger o servo
- Facilita debug e monitoramento da saúde do servo

---

## 📊 **Configurações Atuais**

| Parâmetro | Valor Original | Valor Protegido | Motivo |
|-----------|----------------|-----------------|---------|
| **Ângulo máximo** | ±45° | ±20° | Proteger mecanismo |
| **Frequência máxima** | Ilimitada | 12.5 Hz | Evitar sobrecarga |
| **Mudança máxima** | Ilimitada | 5°/comando | Suavizar movimento |
| **Tempo mínimo entre comandos** | 0ms | 80ms | Proteger eletrônica |

---

## 🚗 **Configuração Atual de Velocidade**

### **Motor (Throttle) - Funciona Normalmente**
```cpp
#define DEFAULT_CONSTANT_SPEED_KMH 0.5   // 0.5 km/h (0.14 m/s) - muito lento
#define DEFAULT_CONSTANT_THROTTLE 0.08   // 8% potência - bem baixo
```

### **Soft Start para Motor**
```cpp
max_throttle_change_per_step = 0.005;    // 0.5% por iteração  
initial_throttle_limit = 0.03;           // 3% máximo inicial
warmup_duration_seconds = 5.0;           // 5 segundos de aquecimento
```

---

## 🔧 **Como Ajustar para Testes**

### **Para Testes Mais Lentos (Servo Ultra-Protegido)**
```cpp
// main.cpp - linha 30
#define DEFAULT_CONSTANT_SPEED_KMH 0.2   // 0.2 km/h - extremamente lento

// EngineController.cpp - linha ~175
static const int SAFE_MAX_ANGLE = 15;   // Apenas ±15°
if(elapsed < 100) return;               // 100ms entre comandos
int max_change = 3;                     // Apenas 3° por comando
```

### **Para Testes Mais Rápidos (Mantendo Servo Protegido)**
```cpp
// main.cpp - linha 30  
#define DEFAULT_CONSTANT_SPEED_KMH 1.0   // 1 km/h - ainda bem lento

// EngineController.cpp - manter proteções do servo iguais
```

---

## 🚨 **Sinais de Problema no Servo**

### **Sintomas de Servo Sobrecarregado:**
- Ruídos estranhos durante movimentação
- Movimento irregular ou "travando"
- Aquecimento excessivo
- Perda de precisão na direção

### **Se Detectar Problemas:**
1. **Parar testes imediatamente** (`e` - Emergency Stop)
2. **Reduzir ainda mais as proteções**
3. **Verificar conexões I2C**
4. **Inspecionar mecanismo físico**

---

## 📝 **Debug do Servo**

### **Logs Importantes:**
```
[SERVO PROTECTION] Comando muito rápido (45ms) - aguardando para proteger servo
[SERVO PROTECTION] Limitando mudança de 15° para 5° (protegendo servo)  
[SERVO] Ângulo: 12° (PWM: 1650)
```

### **Comandos de Teste:**
- `1` - Ativar MPC (teste com servo)
- `8` - Ativar velocidade constante (motor apenas)
- `e` - Emergency Stop (parar tudo imediatamente)

---

## ✅ **Resumo das Diferenças**

| Componente | Tratamento | Proteções | Comandos |
|------------|------------|-----------|----------|
| **🔴 SERVO** | Frágil | Rate limiting, amplitude limitada, delays | `steering`, `steer` |
| **🟢 MOTOR** | Robusto | Apenas básicas | `throttle`, `speed` |
