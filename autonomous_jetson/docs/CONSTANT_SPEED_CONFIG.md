# 🚗 Configuração de Velocidade Constante - Jetracer

## 📖 **Como Alterar Rapidamente os Valores de Velocidade**

Para alterar os valores de velocidade constante, edite as macros no arquivo `main.cpp`:

## 🚗 **Valores Padrão Atuais**

```cpp
#define DEFAULT_CONSTANT_SPEED 0.56    // m/s (equivale a 2 km/h - muito seguro)
#define DEFAULT_CONSTANT_THROTTLE 0.15  // 15% de potência (muito conservador)
```

## 🔧 **Conversões Úteis (km/h para m/s)**

| km/h | m/s   | Uso Recomendado                    |
|------|-------|------------------------------------|
| 1    | 0.28  | Testes iniciais (muito lento)     |
| 2    | 0.56  | **PADRÃO ATUAL** - Muito seguro    |
| 3    | 0.83  | Testes normais                     |
| 4    | 1.11  | Testes mais rápidos                |
| 5    | 1.39  | Velocidade máxima segura           |
| 6    | 1.67  | Apenas para ambientes controlados  |

**Fórmula:** `velocidade_m_s = velocidade_km_h ÷ 3.6`

## ⚡ **Valores Recomendados de Throttle**

| Throttle | Potência | Velocidade Esperada | Uso                      |
|----------|----------|---------------------|--------------------------|
| 0.10     | 10%      | 1-2 km/h           | Testes muito lentos      |
| 0.15     | 15%      | 2-3 km/h           | **PADRÃO** - Muito seguro|
| 0.20     | 20%      | 3-4 km/h           | Testes normais           |
| 0.25     | 25%      | 4-5 km/h           | Velocidade moderada      |
| 0.30     | 30%      | 5-6 km/h           | **MÁXIMO SEGURO**        |

⚠️ **ATENÇÃO**: Valores acima de 0.30 (30%) podem ser perigosos!

## 🎯 **Exemplos de Configuração**

### Para testes muito lentos (1 km/h):
```cpp
#define DEFAULT_CONSTANT_SPEED 0.28    // 1 km/h
#define DEFAULT_CONSTANT_THROTTLE 0.10  // 10% potência
```

### Para testes normais (3 km/h):
```cpp
#define DEFAULT_CONSTANT_SPEED 0.83    // 3 km/h
#define DEFAULT_CONSTANT_THROTTLE 0.20  // 20% potência
```

### Para testes mais rápidos (5 km/h):
```cpp
#define DEFAULT_CONSTANT_SPEED 1.39    // 5 km/h
#define DEFAULT_CONSTANT_THROTTLE 0.25  // 25% potência
```

## 🔄 **Como Aplicar as Mudanças**

1. **Edite as macros** no arquivo `main.cpp` (linhas ~32-33)
2. **Recompile** o projeto: `make`
3. **Execute** o sistema
4. **Ative o modo velocidade constante** com o comando `8`

## 📋 **Comandos Relacionados no Sistema**

- `8`: Ativar modo velocidade constante
- `9`: Desativar modo velocidade constante  
- `0`: Ajustar velocidade dinamicamente (via interface)
- `s`: Mostrar status atual (inclui modo velocidade constante)

## 🛡️ **Características de Segurança**

- **Emergency Stop ('e')** sempre funciona, independente do modo
- **Soft Start** aplicado automaticamente em modo autônomo
- **Limitação automática** de mudanças bruscas de steering
- **Validação de ranges** para evitar valores perigosos

## 📊 **Dimensões do Jetracer (para referência)**

- **Wheelbase**: 150mm
- **Distância entre rodas**: 170mm  
- **Diâmetro das rodas**: 65mm
- **Ângulo máximo de steering**: ±22.5°

Essas dimensões afetam como a velocidade e steering se traduzem em movimento real.
