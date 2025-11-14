#!/usr/bin/env python3
"""
Executa múltiplas rodadas do Tabu Search com parâmetros adaptativos.
"""

import subprocess
from pathlib import Path

# Lista de instâncias com número de alunos estimado
instances = [
    #("instance_classroom.txt", 40),
    #("instance_school.txt", 1000),
    #("instance_neighborhood.txt", 10000),
    #("instance_rural_city.txt", 75000),
    #("instance_urban_city.txt", 250000),
    #("instance_country_city.txt", 150000),
    ("instance_state.txt", 1000000)
]

# Pasta para salvar resultados
results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

print("=" * 70)
print("RODANDO EXPERIMENTOS TABU SEARCH")
print("=" * 70)

for filename, n_students in instances:
    instance_path = f"instances/instances/{filename}"

    # Define parâmetros adaptativos
    max_iterations = min(75, max(50, int(n_students * 0.002)))
    tabu_tenure = max(10, int(n_students * 0.01))

    print(f"\n>>> Rodando Tabu Search para {filename}")
    print(f"Alunos: {n_students}, Iterações: {max_iterations}, Tabu Tenure: {tabu_tenure}")

    cmd = [
        "python3", "TABU.py",
        "--instance", instance_path,
        "--max-iterations", str(max_iterations),
        "--tabu-tenure", str(tabu_tenure)
    ]
    subprocess.run(cmd)

print("\n" + "=" * 70)
print("TODOS OS EXPERIMENTOS CONCLUÍDOS!")
print("=" * 70)
