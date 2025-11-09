# Problema de Evasão Escolar - Otimização Combinatória

Trabalho MO824 Tópicos em Otimização Combinatória - Pesquisa Operacional
## Sobre

Implementação e comparação de métodos de otimização para o problema de alocação de intervenções para prevenção de evasão escolar:
- **ILP**: Programação Linear Inteira (Gurobi)
  
- **GRASP**: Metaheurística construtiva com busca local
- **GRASP + Reactive + VND**: Versão avançada com alpha reativo e busca local multi-vizinhança

- **Tabu Search**:

## Arquivos

### Geração de Instâncias
- `generate_risks_markov.py` - Gera riscos iniciais dos alunos
- `generate_instance.py` - Gera instância completa do problema

### Solvers
- `solve_ilp.py` - Resolve via PLI com Gurobi
- `GRASP.py` - GRASP padrão
- `GRASP_Reactive_VND.py` - GRASP com Reactive e VND

### Automação de Experimentos
- `run_experiments.py` - Executa todos os experimentos

## Como Usar

### Executar Todos os Experimentos
```bash
# Sequencial
python3 run_experiments.py
```

## Resultados

Os experimentos geram:
- `instances/` - Instâncias geradas
- `results/` - Métricas de cada execução
- `results/results.csv` - CSV consolidado com todas as métricas

### Métricas no CSV
- Valor objetivo (risco final)
- Redução percentual do risco
- Tempo de execução
- Uso do orçamento
- Cobertura de alunos
- MIP gap (ILP)

## Tamanhos de Instância

- `classroom`: 40 alunos
- `school`: 1.000 alunos
- `neighborhood`: 10.000 alunos
- `rural_city`: 75.000 alunos
- `country_city`: 150.000 alunos
- `urban_city`: 250.000 alunos
- `state`: 1.000.000 alunos

## Requisitos

- Python 3.7+
- Licença Gurobi
- Bibliotecas: numpy, pandas, gurobipy

---
