# 🎯 Teste de Inicialização do Servo

## 📋 **Objetivo**
Implementar um teste automático durante a inicialização do sistema que verifica se os comandos estão sendo enviados corretamente ao servo.

## 🔧 **Implementação**

### **Sequência do Teste:**
1. **FASE 1 - Valores Calculados**: 
   - Mover para ESQUERDA (-45°) - testa cálculos
   - Mover para DIREITA (+45°) - testa cálculos  
   - Retornar ao CENTRO (0°)
2. **FASE 2 - Valores PWM Diretos**:
   - Mover para LIMITE ABSOLUTO ESQUERDO (PWM=240)
   - Mover para LIMITE ABSOLUTO DIREITO (PWM=470)
   - Retornar ao CENTRO (PWM=340)

### **Configurações do Teste:**
- **Ângulo de teste Fase 1**: ±45° (limite máximo de hardware)
- **PWM de teste Fase 2**: Valores diretos extremos
- **Tempo de movimento**: 2000ms (Fase 1), 2500ms (Fase 2)
- **Tempo de centralização**: 1000ms
- **Bypass das proteções**: Durante inicialização apenas

## 🎮 **Valores PWM Utilizados**

| Posição | Ângulo | Fórmula PWM | Valor Esperado |
|---------|--------|-------------|----------------|
| **Esquerda** | -45° | CENTER + ((-45/45) × (CENTER-LEFT)) | ~240 |
| **Centro** | 0° | SERVO_CENTER_PWM | 340 |
| **Direita** | +45° | CENTER + ((+45/45) × (RIGHT-CENTER)) | ~470 |

### **Valores PWM Diretos (Fase 2):**
| Posição | PWM Direto | Resultado Esperado |
|---------|------------|-------------------|
| **Esquerda Absoluta** | 240 | Movimento máximo à esquerda |
| **Centro** | 340 | Posição neutra |
| **Direita Absoluta** | 470 | Movimento máximo à direita |

### **Constantes de Hardware:**
```cpp
SERVO_CENTER_PWM = 340
SERVO_LEFT_PWM = 240    // (340 - 100)
SERVO_RIGHT_PWM = 470   // (340 + 130)
HARDWARE_MAX_ANGLE = 45 // ±45° para cálculos (limite físico real)
```

## 📊 **Logs Esperados**

### **Teste Bem-Sucedido:**
```
=== SERVO INITIALIZATION TEST ===
Testing servo movement to verify command transmission...
[SERVO TEST] Moving to LEFT (-40°)...
[SERVO TEST] Moving to RIGHT (40°)...
[SERVO TEST] Returning to CENTER (0°)...
[SERVO TEST] ✅ Test completed successfully!
[SERVO TEST] PWM Values - Left: 291, Right: 389, Center: 340
[SERVO TEST] PWM Range - Left=240, Center=340, Right=470
[SERVO TEST] Angle Range - Test used ±40° out of ±180° max
=================================
```

### **Teste com Erro:**
```
[SERVO TEST] ❌ ERROR: [mensagem de erro]
```

## 🔍 **Diagnóstico**

### **✅ Se o servo se mover visivelmente:**
- Comandos estão sendo transmitidos corretamente
- Hardware I2C funcionando
- PWM sendo aplicado ao servo

### **❌ Se o servo não se mover:**
- Verificar conexões I2C
- Verificar alimentação do servo
- Verificar se o servo não está travado fisicamente

### **⚠️ Se o movimento for muito pequeno:**
- Verificar se as constantes PWM estão corretas
- Verificar se o servo não está limitado por configuração

## 🎯 **Quando o Teste Executa**

O teste é executado automaticamente:
- **Durante a inicialização** do `EngineController`
- **Após** a inicialização do servo (`init_servo()`)
- **Antes** de qualquer operação normal do sistema

## 🛡️ **Proteções**

### **Durante o Teste:**
- **Bypass** das proteções normais de rate limiting
- **Ângulos seguros** (40° é menor que o limite físico)
- **Try-catch** para capturar erros
- **Retorno automático ao centro** em caso de erro

### **Após o Teste:**
- Todas as proteções normais são reativadas
- Sistema continua com operação normal

## 🔧 **Manutenção**

### **Para Ajustar o Teste:**
```cpp
// Em EngineController.cpp, linha ~296
const int TEST_ANGLE = 40; // Ajustar ângulo de teste

// Tempos de movimento (linhas ~302, 309, 316)
std::this_thread::sleep_for(std::chrono::milliseconds(1500)); // Ajustar tempo
```

### **Para Desabilitar o Teste:**
Comentar a linha no construtor:
```cpp
// testServoInitialization(); // Comentar para desabilitar
```

---

**Status**: ✅ **Implementado e ativo**  
**Resultado**: ✅ **TESTE PASSOU - Servo responsivo com movimento visível**  
**Próxima ação**: ✅ **MPC liberado para controle total (±45°)**

## 🚀 **Resultado do Teste e Implementações**

### **✅ Resultado Positivo:**
- **Fase 1**: Movimento calculado visível confirmado
- **Fase 2**: Movimento máximo com PWM direto confirmado  
- **Conclusão**: Servo responde adequadamente aos comandos

### **🔧 Melhorias Implementadas:**

#### **1. EngineController - Controle Total**
```cpp
// ANTES: Limitado a ±20°
static const int SAFE_MAX_ANGLE = 20;

// DEPOIS: Faixa física completa
const int HARDWARE_MAX_ANGLE = 45; // ±45° conforme hardware
```

#### **2. ControlsManager - Autoridade MPC**
```cpp
// ANTES: Limitado e com rate limiting
std::clamp(control.steer * 22.5, -22.5, 22.5); // ±22.5°

// DEPOIS: Autoridade completa para MPC
std::clamp(control.steer * 45.0, -45.0, 45.0); // ±45° completo
```

#### **3. MPCConfig - Limites Expandidos**
```cpp
// ANTES: Conservativo
steering_limits = {-0.35, 0.35}; // ±20°

// DEPOIS: Autoridade máxima
steering_limits = {-0.785, 0.785}; // ±45° (0.785 rad)
```

### **📈 Benefícios Alcançados:**
- **Autoridade de direção**: 20° → 45° (**2.25x maior**)
- **Throttle**: 20% → 25% (**25% mais potência**)
- **Responsividade**: Rate limiting removido
- **Performance MPC**: Otimização com espaço de controle completo

### **🛡️ Segurança Mantida:**
- **Switch físico**: Segurança primária (instalado pelo usuário)
- **Limites de hardware**: PWM 240-470 hard-coded
- **Rate limiting temporal**: 70ms mínimo (proteção do servo)
- **Monitoramento**: Logs detalhados de uso da faixa completa

---

**Próximo teste**: Executar o programa e observar MPC utilizando faixa completa de direção
