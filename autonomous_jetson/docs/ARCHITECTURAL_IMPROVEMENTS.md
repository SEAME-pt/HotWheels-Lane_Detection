# Melhorias Arquiteturais - Integração MPC/Polyfitter/TensorRT

## Problema Identificado
O MPC não interage diretamente com o Polyfitter, que por sua vez não herda do TensorRTInferencer, criando um pipeline fragmentado e ineficiente.

## Proposta de Solução

### Arquitetura Atual (Problemática)
```
TensorRT → ZeroMQ → MPC (separado)
Polyfitter (isolado, processamento manual de máscaras)
```

### Arquitetura Proposta 1: Pipeline Integrado
```
TensorRT → Polyfitter (integrado) → MPC (direto)
```

### Arquitetura Proposta 2: Herança Especializada
```
IInferencer
    ├── TensorRTInferencer
    └── PolyfitterInferencer : TensorRTInferencer
            └── Enhanced Lane Processing → MPC
```

## Implementação Sugerida

### 1. Criar PolyfitterInferencer
- Herdar de TensorRTInferencer
- Integrar processamento avançado de pistas
- Eliminar dependência do ZeroMQ para dados críticos do MPC

### 2. Pipeline Direto MPC
- MPCPlanner recebe dados diretamente do PolyfitterInferencer
- Eliminar latência de comunicação assíncrona
- Garantir sincronização temporal

### 3. Otimizações de Performance
- Processamento GPU end-to-end
- Cache de coeficientes polinomiais
- Predição de trajetória integrada

## Benefícios Esperados
1. **Redução de Latência**: Eliminação do overhead do ZeroMQ
2. **Melhor Sincronização**: Dados sempre atualizados para o MPC
3. **Arquitetura Limpa**: Pipeline linear e compreensível
4. **Performance**: Processamento GPU otimizado
5. **Manutenibilidade**: Menos componentes desacoplados

## Próximos Passos
1. Implementar PolyfitterInferencer
2. Modificar MPCPlanner para usar dados diretos
3. Manter ZeroMQ apenas para visualização/debug
4. Benchmarking de performance
