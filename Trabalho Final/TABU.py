#!/usr/bin/env python3
"""
Tabu Search para o Problema de Evasão Escolar
"""

import argparse
import random
import time
from pathlib import Path
import sys
import csv
import os

from GRASP import SchoolDropoutInstance, Solution, IncrementalEvaluator  # Reuso das classes

class TabuSearch:
    """
    Implementação da Tabu Search para o Problema de Evasão Escolar.
    """

    def __init__(self, instance, max_iterations=500, tabu_tenure=10, max_candidates=None, seed=42):
        """
        Args:
            instance: objeto SchoolDropoutInstance
            max_iterations: número máximo de iterações
            tabu_tenure: tamanho da lista tabu
            max_candidates: número máximo de movimentos candidatos por iteração
            seed: semente aleatória
        """
        self.instance = instance
        self.max_iterations = max_iterations
        self.tabu_tenure = tabu_tenure
        
        if max_candidates is None:
            self.max_candidates = min(2500, int(self.instance.n_students * 0.1))

        #self.max_candidates = max_candidates
        random.seed(seed)

        self.best_solution = None
        self.best_objective = float('inf')

    def solve(self):
        """Executa a busca tabu."""
        print(f"\n{'='*70}")
        print(f"TABU SEARCH - School Dropout Problem")
        print(f"{'='*70}")
        print(f"Max iterations: {self.max_iterations}")
        print(f"Tabu tenure: {self.tabu_tenure}")
        print(f"Max candidates per iteration: {self.max_candidates}")
        print(f"{'='*70}\n")

        start_time = time.time()



        current_solution = Solution(self.instance)
        current_solution.evaluate()
        evaluator = IncrementalEvaluator(self.instance, current_solution)


        tabu_list = {}
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1


            candidates = self._generate_candidates(current_solution, evaluator)

            if not candidates:
                print(f"Iteração {iteration}: Nenhum movimento viável encontrado.")
                break


            best_move = None
            best_delta = float('inf')

            for move in candidates:
                delta_risk, student_id, new_subset_idx = move
                move_key = (student_id, new_subset_idx)

            
                if move_key in tabu_list and delta_risk >= 0 and current_solution.objective_value + delta_risk >= self.best_objective:
                    continue

                if delta_risk < best_delta:
                    best_delta = delta_risk
                    best_move = move

            if best_move is None:
                print(f"Iteração {iteration}: Todos movimentos tabu.")
                break

           
            _, student_id, new_subset_idx = best_move
            evaluator.apply_move(student_id, new_subset_idx)
            current_solution.evaluate()

            
            tabu_list[(student_id, new_subset_idx)] = iteration + self.tabu_tenure
            
            tabu_list = {k: v for k, v in tabu_list.items() if v > iteration}

            
            if current_solution.is_feasible and current_solution.objective_value < self.best_objective:
                self.best_solution = current_solution.copy()
                self.best_objective = current_solution.objective_value
                improvement = (1 - self.best_objective / sum(self.instance.risks)) * 100
                print(f"Iter {iteration:3d}: Nova melhor = {self.best_objective:.6f} (Redução risco: {improvement:.2f}%)")

        elapsed_time = time.time() - start_time
        print(f"\n{'='*70}")
        print(f"Tabu Search concluída em {elapsed_time:.2f} segundos")
        print(f"Melhor valor objetivo: {self.best_objective:.6f}")
        print(f"{'='*70}")

        return self.best_solution, elapsed_time


    def _generate_candidates(self, solution, evaluator):
        """
        Gera movimentos candidatos (troca de subset para alunos).
        """
        candidates = []
        
        sample_size = min(self.instance.n_students, max(100, int(self.instance.n_students * 0.5)))
        students = random.sample(range(self.instance.n_students), sample_size)

        #students = random.sample(range(self.instance.n_students), min(self.instance.n_students, 100))

        for student_id in students:
            current_subset_idx = solution.assignment[student_id]
            eligible_subsets = [i for i in range(len(self.instance.subsets[student_id])) if i != current_subset_idx]
            eligible_subsets = random.sample(eligible_subsets, min(len(eligible_subsets), self.max_candidates))

            for new_subset_idx in eligible_subsets:
                delta_risk, _, feasible = evaluator.evaluate_delta(student_id, new_subset_idx)
                if feasible and delta_risk < 0:
                #if feasible and (delta_risk <= 0 or dentro_do_orcamento):
                    candidates.append((delta_risk, student_id, new_subset_idx))

        return candidates

def save_metrics(solution, elapsed_time, output_file):
    """Salva métricas da solução em TXT."""
    instance = solution.instance
    initial_risk = sum(instance.risks)
    final_risk = solution.objective_value
    risk_reduction = initial_risk - final_risk
    risk_reduction_pct = (risk_reduction / initial_risk) * 100

    cost_details = solution.get_cost_details()
    students_served = sum(1 for s in range(instance.n_students) if solution.assignment[s] != 0)
    students_served_pct = (students_served / instance.n_students) * 100

    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("TABU SEARCH SOLUTION METRICS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Solve time: {elapsed_time:.2f} seconds\n")
        f.write(f"Final risk: {final_risk:.6f}\n")
        f.write(f"Risk reduction: {risk_reduction_pct:.2f}%\n")
        f.write(f"Budget used: {cost_details['total_cost']:.2f} / {instance.budget:.2f}\n")
        f.write(f"Students served: {students_served} ({students_served_pct:.2f}%)\n")
        f.write("=" * 70 + "\n")

    print(f"\nMétricas salvas em: {output_file}")


def save_results_csv(solution, elapsed_time, output_csv, max_iterations, tabu_tenure):
    """Salva resultados em formato CSV com parâmetros e métricas."""
    instance = solution.instance
    initial_risk = sum(instance.risks)
    final_risk = solution.objective_value
    risk_reduction_pct = ((initial_risk - final_risk) / initial_risk) * 100

    cost_details = solution.get_cost_details()
    students_served = sum(1 for s in range(instance.n_students) if solution.assignment[s] != 0)
    students_served_pct = (students_served / instance.n_students) * 100


    header = [
        "solver", "instance", "max_iterations", "tabu_tenure",
        "final_risk", "risk_reduction_pct", "solve_time",
        "budget", "total_cost", "budget_used_pct",
        "students_served", "students_served_pct"
    ]


    row = [
        "TabuSearch",
        Path(instance.filename).stem,
        max_iterations,
        tabu_tenure,
        f"{final_risk:.6f}",
        f"{risk_reduction_pct:.2f}",
        f"{elapsed_time:.2f}",
        f"{instance.budget:.2f}",
        f"{cost_details['total_cost']:.2f}",
        f"{(cost_details['total_cost'] / instance.budget) * 100:.2f}",
        students_served,
        f"{students_served_pct:.2f}"
    ]


    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)


    file_exists = Path(output_csv).exists()
    with open(output_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)

    print(f"\nResultados salvos no CSV: {output_csv}")


def main():
    parser = argparse.ArgumentParser(description="Tabu Search para Evasão Escolar")
    parser.add_argument("--instance", type=str, required=True)
    parser.add_argument("--max-iterations", type=int, default=500)
    parser.add_argument("--tabu-tenure", type=int, default=10)
    parser.add_argument("--output-metrics", type=str, default=None)
    args = parser.parse_args()

    instance_basename = Path(args.instance).stem
    if args.output_metrics is None:
        args.output_metrics = (
            f"results/metrics_tabu_{instance_basename}_it{args.max_iterations}_tenure{args.tabu_tenure}.txt"
        )

    print("=" * 70)
    print("TABU SEARCH - SCHOOL DROPOUT PROBLEM")
    print("=" * 70)
    print(f"Instância: {args.instance}")
    print(f"Iterações: {args.max_iterations}")
    print(f"Tabu tenure: {args.tabu_tenure}")

    instance = SchoolDropoutInstance(args.instance)
    tabu = TabuSearch(instance, max_iterations=args.max_iterations, tabu_tenure=args.tabu_tenure)
    solution, elapsed_time = tabu.solve()

    if solution is None or not solution.is_feasible:
        print("\nERRO: Nenhuma solução viável encontrada")
        return 1

    # Salvar métricas TXT
    save_metrics(solution, elapsed_time, args.output_metrics)

    # Salvar também em CSV consolidado
    output_csv = "results/results_tabu.csv"
    save_results_csv(solution, elapsed_time, output_csv, args.max_iterations, args.tabu_tenure)

    print("\n" + "=" * 70)
    print("TABU SEARCH CONCLUÍDA COM SUCESSO")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())