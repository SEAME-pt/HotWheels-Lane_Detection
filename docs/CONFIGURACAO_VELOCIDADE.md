# 🚗 Configuração Rápida de Velocidade Constante

## 📝 **Como Alterar Velocidade Rapidamente**

Para alterar os valores de velocidade constante sem precisar procucar no código, simplesmente modifique as **MACROS** no início dos arquivos:

### **1. Arquivo Principal: `main.cpp`**

```cpp
// Linhas 19-21 no main.cpp
#define DEFAULT_CONSTANT_SPEED_KMH 2.0   // ← ALTERE AQUI para mudar velocidade alvo (em km/h)
#define DEFAULT_CONSTANT_SPEED (DEFAULT_CONSTANT_SPEED_KMH / 3.6) // Conversão automática para m/s
#define DEFAULT_CONSTANT_THROTTLE 0.15   // ← ALTERE AQUI para mudar potência do motor
```

### **2. Arquivo de Header: `car_controls/includes/ControlsManager.hpp`**

```cpp
// Linhas 43-48 no ControlsManager.hpp
#define DEFAULT_CONSTANT_SPEED_KMH 2.0   // ← ALTERE AQUI também (em km/h)
#define DEFAULT_CONSTANT_SPEED (DEFAULT_CONSTANT_SPEED_KMH / 3.6) // Conversão automática para m/s
#define DEFAULT_CONSTANT_THROTTLE 0.15   // ← ALTERE AQUI também
```

---

## ⚠️ **IMPORTANTE: Motor vs Servo**

### **🔋 MOTOR (Throttle) - ROBUSTO:**
- Controla a **aceleração** (velocidade das rodas)
- **Aguenta comandos extremos** - pode usar valores altos sem problema
- Valores seguros: 0.1 a 0.5 (10% a 50% de potência)

### **🎛️ SERVO (Steering) - FRÁGIL:**
- Controla a **direção** (ângulo das rodas)
- **MUITO DELICADO** - movimentos bruscos podem quebrar
- Precisa de proteções especiais (implementadas no código)

---

## ⚡ **Valores Recomendados (em km/h)**

### **Para Testes Normais:**
```cpp
#define DEFAULT_CONSTANT_SPEED_KMH 2.0   // 2 km/h - velocidade normal (0.56 m/s)
#define DEFAULT_CONSTANT_THROTTLE 0.15   // 15% potência - normal para motores
```

### **Para Testes Mais Rápidos:**
```cpp
#define DEFAULT_CONSTANT_SPEED_KMH 4.0   // 4 km/h - velocidade moderada (1.11 m/s)
#define DEFAULT_CONSTANT_THROTTLE 0.3    // 30% potência - motores aguentam tranquilo
```

### **Para Testes de Desempenho:**
```cpp
#define DEFAULT_CONSTANT_SPEED_KMH 6.0   // 6 km/h - velocidade alta (1.67 m/s)
#define DEFAULT_CONSTANT_THROTTLE 0.4    // 40% potência - próximo do limite seguro
```
#define DEFAULT_CONSTANT_SPEED_KMH 1.0   // 1 km/h - lento (0.28 m/s)
#define DEFAULT_CONSTANT_THROTTLE 0.12   // 12% potência
```

### **Para Testes de Desenvolvimento (Servo Ainda Protegido):**
```cpp
#define DEFAULT_CONSTANT_SPEED_KMH 2.0   // 2 km/h - moderado (0.56 m/s)
#define DEFAULT_CONSTANT_THROTTLE 0.15   // 15% potência
```

---

## ⚠️ **Limites de Segurança**

| Parâmetro | Mínimo Seguro | Máximo Seguro | Observações |
|-----------|---------------|---------------|-------------|
| **DEFAULT_CONSTANT_SPEED_KMH** | 0.2 km/h | 2.0 km/h | Servo é frágil - velocidades altas são perigosas |
| **DEFAULT_CONSTANT_THROTTLE** | 0.05 | 0.2 | Motor é robusto, mas servo precisa de proteção |

**⚠️ ATENÇÃO: SERVO FRÁGIL!**
- **SERVO (Steering)**: Componente frágil, quebra facilmente
- **MOTOR (Throttle)**: Robusto, aceita inputs extremos  
- **Proteções**: Apenas o servo tem limitações especiais

**Conversão Automática:**
- 0.2 km/h = 0.06 m/s ← **Extremamente lento**
- 0.5 km/h = 0.14 m/s ← **Valor atual**
- 1.0 km/h = 0.28 m/s ← **Lento seguro**
- 2.0 km/h = 0.56 m/s ← **Moderado**

---

## 🔧 **Como Aplicar as Mudanças**

1. **Edite as macros** nos arquivos indicados
2. **Recompile o projeto:**
   ```bash
   cd /home/michel/Documents/other
   make clean && make
   ```
3. **Execute o programa** e teste com comando `8` (Ativar Velocidade Constante)

---

## 🎮 **Comandos de Teste**

Após recompilar, use estes comandos no programa:

- `8` - Ativar modo velocidade constante
- `9` - Desativar modo velocidade constante  
- `0` - Ajustar velocidade manualmente (sobrescreve a macro)
- `1` - Ativar modo MPC com velocidade constante

---

## 📊 **Monitoramento**

O sistema mostra na tela:
- **Velocidade alvo atual**
- **Throttle fixo aplicado**
- **Status do modo velocidade constante**

---

## 🛡️ **Sistema de Segurança**

- **Emergency Stop (`e`)** funciona independente da velocidade configurada
- **Soft Start** limita aceleração inicial do motor
- **Proteção do Servo**: Rate limiting e amplitude limitada para proteger servo frágil
- **Motor Robusto**: Throttle funciona normalmente, aceita mudanças rápidas

### **🔴 SERVO (Steering) - FRÁGIL:**
- Limitado a ±20° (ao invés de ±45°)
- Mínimo 80ms entre comandos  
- Máximo 5° de mudança por comando
- Logs de proteção quando limita comandos

### **🟢 MOTOR (Throttle) - ROBUSTO:**
- Sem limitações especiais de rate limiting
- Aceita mudanças rápidas de velocidade
- Apenas soft start para suavidade inicial

---

## 📝 **Exemplo de Configuração Personalizada**

Para um teste específico com velocidade de 0.8 m/s:

1. **Edite `main.cpp` linha 19:**
   ```cpp
   #define DEFAULT_CONSTANT_SPEED 0.8
   ```

2. **Edite `main.cpp` linha 20:**
   ```cpp
   #define DEFAULT_CONSTANT_THROTTLE 0.2  // 20% potência para 0.8 m/s
   ```

3. **Edite `ControlsManager.hpp` linhas 43-44** com os mesmos valores

4. **Recompile e teste!**
