#!/usr/bin/env python3
"""
Reactive GRASP + VND for School Dropout Problem
"""

import argparse
import random
import time
from pathlib import Path
from collections import defaultdict
import sys


class SchoolDropoutInstance:
    """Represents a School Dropout Problem instance."""
    
    def __init__(self, filename):
        """Load instance from file."""
        self.filename = filename
        self.n_students = 0
        self.n_interventions = 0
        self.budget = 0.0
        
        self.fixed_costs = []
        self.capacities = []
        self.risks = []
        self.var_costs = {}  
        self.subsets = {}    
        self.effectiveness = {} 
        
        self._load_instance()
        self._precompute_data()
    
    def _load_instance(self):
        """Parse instance file."""
        print(f"Loading instance: {self.filename}")
        
        with open(self.filename, 'r') as f:
            lines = f.readlines()
        
        # Parse header
        for line in lines[:10]:
            if line.startswith("# Students:"):
                self.n_students = int(line.split(":")[1].strip())
            elif line.startswith("# Interventions:"):
                self.n_interventions = int(line.split(":")[1].strip())
            elif line.startswith("# Budget:"):
                self.budget = float(line.split(":")[1].strip())
        
        # Parse data
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            parts = line.split()
            
            if parts[0] == "INTERVENTION":
                idx = int(parts[1])
                fixed_cost = float(parts[4])
                capacity = int(parts[5])
                self.fixed_costs.append(fixed_cost)
                self.capacities.append(capacity)
            
            elif parts[0] == "BUDGET":
                self.budget = float(parts[1])
            
            elif parts[0] == "RISK":
                student_id = int(parts[1])
                risk = float(parts[2])
                self.risks.append(risk)
            
            elif parts[0] == "VAR_COST":
                intervention_id = int(parts[1])
                student_id = int(parts[2])
                cost = float(parts[3])
                self.var_costs[(intervention_id, student_id)] = cost
            
            elif parts[0] == "SUBSETS":
                student_id = int(parts[1])
                num_subsets = int(parts[2])
                
                student_subsets = []
                idx = 3
                for _ in range(num_subsets):
                    subset_size = int(parts[idx])
                    idx += 1
                    
                    if subset_size == 0:
                        student_subsets.append(tuple())
                    else:
                        subset = tuple(int(parts[idx + j]) for j in range(subset_size))
                        student_subsets.append(subset)
                        idx += subset_size
                
                self.subsets[student_id] = student_subsets
            
            elif parts[0] == "EFFECTIVENESS":
                student_id = int(parts[1])
                subset_idx = int(parts[2])
                eff = float(parts[3])
                
                if student_id not in self.effectiveness:
                    self.effectiveness[student_id] = []
                self.effectiveness[student_id].append(eff)
        
        print(f"  Students: {self.n_students:,}")
        print(f"  Interventions: {self.n_interventions}")
        print(f"  Budget: ${self.budget:,.2f}")
    
    def _precompute_data(self):
        """Precompute useful data structures."""
        # Precompute subset costs for each student
        self.subset_costs = {}  # (student_id, subset_idx) -> total_cost
        
        for student_id in range(self.n_students):
            for subset_idx, subset in enumerate(self.subsets[student_id]):
                total_cost = 0.0
                for intervention_id in subset:
                    if (intervention_id, student_id) in self.var_costs:
                        total_cost += self.var_costs[(intervention_id, student_id)]
                self.subset_costs[(student_id, subset_idx)] = total_cost
        
        # Sort students by risk (descending) for risk-based prioritization
        self.students_by_risk = sorted(
            range(self.n_students),
            key=lambda s: self.risks[s],
            reverse=True
        )


class IncrementalEvaluator:
    """
    Incremental evaluator for fast delta calculations.
    """
    
    def __init__(self, instance, solution):
        self.instance = instance
        self.solution = solution
        
        # Current state tracking
        self.intervention_usage = [0] * instance.n_interventions
        self.activated_interventions = set()
        self.total_var_cost = 0.0
        self.total_risk = 0.0
        
        self._initialize_state()
    
    def _initialize_state(self):
        """Initialize state from current solution."""
        instance = self.instance
        
        for student_id in range(instance.n_students):
            subset_idx = self.solution.assignment[student_id]
            subset = instance.subsets[student_id][subset_idx]
            
            # Update intervention usage
            for intervention_id in subset:
                self.intervention_usage[intervention_id] += 1
                self.activated_interventions.add(intervention_id)
            
            # Update variable cost
            self.total_var_cost += instance.subset_costs[(student_id, subset_idx)]
            
            # Update residual risk
            risk = instance.risks[student_id]
            effectiveness = instance.effectiveness[student_id][subset_idx]
            self.total_risk += risk * (1.0 - effectiveness)
    
    def evaluate_delta(self, student_id, new_subset_idx):
        """
        Evaluate the delta (change) of assigning new_subset_idx to student_id.
        Returns: (delta_risk, delta_cost, is_feasible)
        """
        instance = self.instance
        old_subset_idx = self.solution.assignment[student_id]
        
        if old_subset_idx == new_subset_idx:
            return 0.0, 0.0, True
        
        old_subset = instance.subsets[student_id][old_subset_idx]
        new_subset = instance.subsets[student_id][new_subset_idx]
        
        # Calculate delta risk
        risk = instance.risks[student_id]
        old_eff = instance.effectiveness[student_id][old_subset_idx]
        new_eff = instance.effectiveness[student_id][new_subset_idx]
        
        old_residual = risk * (1.0 - old_eff)
        new_residual = risk * (1.0 - new_eff)
        delta_risk = new_residual - old_residual
        
        # Calculate delta variable cost
        old_cost = instance.subset_costs[(student_id, old_subset_idx)]
        new_cost = instance.subset_costs[(student_id, new_subset_idx)]
        delta_var_cost = new_cost - old_cost
        
        # Check capacity constraints (incremental)
        temp_usage = self.intervention_usage.copy()
        for intervention_id in old_subset:
            temp_usage[intervention_id] -= 1
        for intervention_id in new_subset:
            temp_usage[intervention_id] += 1
        
        for intervention_id in new_subset:
            if temp_usage[intervention_id] > instance.capacities[intervention_id]:
                return delta_risk, delta_var_cost, False
        
        # Calculate delta fixed cost
        delta_fixed_cost = 0.0
        
        # Interventions being removed
        for intervention_id in old_subset:
            if intervention_id not in new_subset:
                # Check if this intervention becomes unused
                if temp_usage[intervention_id] == 0:
                    delta_fixed_cost -= instance.fixed_costs[intervention_id]
        
        # Interventions being added
        for intervention_id in new_subset:
            if intervention_id not in old_subset:
                # Check if this intervention was not active before
                if intervention_id not in self.activated_interventions:
                    delta_fixed_cost += instance.fixed_costs[intervention_id]
        
        delta_cost = delta_var_cost + delta_fixed_cost
        
        # Check budget constraint
        new_total_cost = self.get_current_cost() + delta_cost
        if new_total_cost > instance.budget + 1e-6:
            return delta_risk, delta_cost, False
        
        return delta_risk, delta_cost, True
    
    def apply_move(self, student_id, new_subset_idx):
        """Apply a move and update internal state."""
        instance = self.instance
        old_subset_idx = self.solution.assignment[student_id]
        old_subset = instance.subsets[student_id][old_subset_idx]
        new_subset = instance.subsets[student_id][new_subset_idx]
        
        # Update intervention usage
        for intervention_id in old_subset:
            self.intervention_usage[intervention_id] -= 1
            if self.intervention_usage[intervention_id] == 0:
                self.activated_interventions.discard(intervention_id)
        
        for intervention_id in new_subset:
            if self.intervention_usage[intervention_id] == 0:
                self.activated_interventions.add(intervention_id)
            self.intervention_usage[intervention_id] += 1
        
        # Update variable cost
        old_cost = instance.subset_costs[(student_id, old_subset_idx)]
        new_cost = instance.subset_costs[(student_id, new_subset_idx)]
        self.total_var_cost += (new_cost - old_cost)
        
        # Update risk
        risk = instance.risks[student_id]
        old_eff = instance.effectiveness[student_id][old_subset_idx]
        new_eff = instance.effectiveness[student_id][new_subset_idx]
        self.total_risk += risk * ((1.0 - new_eff) - (1.0 - old_eff))
        
        # Update solution
        self.solution.assignment[student_id] = new_subset_idx
    
    def get_current_cost(self):
        """Get current total cost."""
        fixed_cost = sum(self.instance.fixed_costs[i] for i in self.activated_interventions)
        return self.total_var_cost + fixed_cost
    
    def get_current_risk(self):
        """Get current total residual risk."""
        return self.total_risk


class Solution:
    """Represents a solution to the School Dropout Problem."""
    
    def __init__(self, instance):
        self.instance = instance
        # student_id -> subset_index
        self.assignment = {}
        self.objective_value = float('inf')
        self.is_feasible = False
        
        # Initialize all students with empty subset (index 0)
        for student_id in range(instance.n_students):
            self.assignment[student_id] = 0
    
    def copy(self):
        """Create a deep copy of this solution."""
        new_sol = Solution(self.instance)
        new_sol.assignment = self.assignment.copy()
        new_sol.objective_value = self.objective_value
        new_sol.is_feasible = self.is_feasible
        return new_sol
    
    def evaluate(self):
        """Evaluate solution: calculate objective value and check feasibility."""
        evaluator = IncrementalEvaluator(self.instance, self)
        
        self.is_feasible = self._check_feasibility()
        
        if not self.is_feasible:
            self.objective_value = float('inf')
        else:
            self.objective_value = evaluator.get_current_risk()
        
        return self.objective_value
    
    def _check_feasibility(self):
        """Check if solution satisfies all constraints."""
        instance = self.instance
        
        # Count intervention usage
        intervention_count = [0] * instance.n_interventions
        activated_interventions = set()
        total_var_cost = 0.0
        
        for student_id in range(instance.n_students):
            subset_idx = self.assignment[student_id]
            subset = instance.subsets[student_id][subset_idx]
            
            for intervention_id in subset:
                intervention_count[intervention_id] += 1
                activated_interventions.add(intervention_id)
            
            total_var_cost += instance.subset_costs[(student_id, subset_idx)]
        
        # Check capacity constraints
        for i in range(instance.n_interventions):
            if intervention_count[i] > instance.capacities[i]:
                return False
        
        # Check budget constraint
        total_fixed_cost = sum(instance.fixed_costs[i] for i in activated_interventions)
        total_cost = total_var_cost + total_fixed_cost
        
        if total_cost > instance.budget + 1e-6:
            return False
        
        return True
    
    def get_cost_details(self):
        """Get detailed cost breakdown."""
        instance = self.instance
        
        activated_interventions = set()
        total_var_cost = 0.0
        intervention_count = [0] * instance.n_interventions
        
        for student_id in range(instance.n_students):
            subset_idx = self.assignment[student_id]
            subset = instance.subsets[student_id][subset_idx]
            
            for intervention_id in subset:
                activated_interventions.add(intervention_id)
                intervention_count[intervention_id] += 1
            
            total_var_cost += instance.subset_costs[(student_id, subset_idx)]
        
        total_fixed_cost = sum(instance.fixed_costs[i] for i in activated_interventions)
        total_cost = total_var_cost + total_fixed_cost
        
        return {
            'total_cost': total_cost,
            'variable_cost': total_var_cost,
            'fixed_cost': total_fixed_cost,
            'activated_interventions': len(activated_interventions),
            'intervention_count': intervention_count
        }


class ReactiveAlphaManager:
    """
    Manages alpha parameter reactively based on solution quality.
    """
    
    def __init__(self, alpha_values=None):
        """Initialize with a set of alpha values."""
        if alpha_values is None:
            self.alpha_values = [0.05, 0.10, 0.15, 0.25]
        else:
            self.alpha_values = alpha_values
        
        n = len(self.alpha_values)
        
        # Initialize uniform probabilities
        self.probabilities = [1.0 / n] * n
        
        # Track average quality for each alpha
        self.alpha_quality = [0.0] * n
        self.alpha_count = [0] * n
        
        # Smoothing parameter
        self.smooth = 0.9
    
    def select_alpha(self):
        """Select alpha based on adaptive probabilities."""
        return random.choices(self.alpha_values, weights=self.probabilities)[0]
    
    def update_statistics(self, alpha, solution_quality):
        """Update statistics after using an alpha value."""
        try:
            alpha_idx = self.alpha_values.index(alpha)
        except ValueError:
            return
        
        # Update count
        self.alpha_count[alpha_idx] += 1
        
        # Update quality with exponential smoothing
        if self.alpha_count[alpha_idx] == 1:
            self.alpha_quality[alpha_idx] = solution_quality
        else:
            self.alpha_quality[alpha_idx] = (
                self.smooth * self.alpha_quality[alpha_idx] + 
                (1 - self.smooth) * solution_quality
            )
        
        # Recalculate probabilities after warmup period
        if sum(self.alpha_count) >= 2 * len(self.alpha_values):
            self._update_probabilities()
    
    def _update_probabilities(self):
        """Update probabilities based on performance."""
        # Get quality values 
        qualities = []
        for i, q in enumerate(self.alpha_quality):
            if self.alpha_count[i] > 0:
                qualities.append(q)
            else:
                qualities.append(float('inf'))
        
        # Calculate fitness
        epsilon = 1e-6
        min_quality = min(q for q in qualities if q < float('inf'))
        
        fitness = []
        for q in qualities:
            if q < float('inf'):
                fitness.append(1.0 / (q - min_quality + epsilon))
            else:
                fitness.append(epsilon)
        
        # Normalize to probabilities
        total_fitness = sum(fitness)
        if total_fitness > 0:
            self.probabilities = [f / total_fitness for f in fitness]
        else:
            # Fallback to uniform
            n = len(self.alpha_values)
            self.probabilities = [1.0 / n] * n


class ReactiveGRASP_VND:
    """
    Reactive GRASP with VND for School Dropout Problem.
    """
    
    def __init__(self, instance, iterations=100, local_search='first',
                 max_candidates_per_student=10, initialization='varied', 
                 seed=42, max_time=600, max_iter_no_improvement=100):
        """
        Initialize Reactive GRASP + VND solver.
        
        Args:
            instance: SchoolDropoutInstance object
            iterations: Number of GRASP iterations
            local_search: 'first' or 'best' improving strategy
            max_candidates_per_student: Max subsets to evaluate per student
            initialization: 'empty', 'random', 'risk', 'varied'
            seed: Random seed
            max_time: Maximum execution time in seconds
            max_iter_no_improvement: Stop after N iterations without improvement
        """
        self.instance = instance
        self.iterations = iterations
        self.local_search = local_search
        self.max_candidates_per_student = max_candidates_per_student
        self.initialization = initialization
        self.max_time = max_time
        self.max_iter_no_improvement = max_iter_no_improvement
        
        random.seed(seed)
        
        # Initialize Reactive Alpha Manager
        self.alpha_manager = ReactiveAlphaManager()
        
        self.best_solution = None
        self.best_objective = float('inf')
    
    def solve(self):
        """Run Reactive GRASP + VND algorithm."""
        print(f"\n{'='*70}")
        print(f"REACTIVE GRASP + VND - School Dropout Problem")
        print(f"{'='*70}")
        print(f"Iterations: {self.iterations}")
        print(f"Alpha values: {self.alpha_manager.alpha_values}")
        print(f"Local search: VND with {self.local_search.upper()} improving")
        print(f"Initialization: {self.initialization.upper()}")
        print(f"Max candidates per student: {self.max_candidates_per_student}")
        print(f"Max iterations without improvement: {self.max_iter_no_improvement}")
        print(f"{'='*70}\n")
        
        start_time = time.time()
        iter_no_improvement = 0
        
        for iteration in range(self.iterations):
            if self.max_time is not None:
                elapsed = time.time() - start_time
                if elapsed >= self.max_time:
                    print(f"\n  TIME LIMIT REACHED: {elapsed:.1f}s")
                    break
            
            # Check iterations without improvement
            if iter_no_improvement >= self.max_iter_no_improvement:
                print(f"\n  STOPPING: {self.max_iter_no_improvement} iterations without improvement")
                break
            
            # Select alpha reactively
            current_alpha = self.alpha_manager.select_alpha()
            
            # Select initialization strategy
            if self.initialization == 'varied':
                if iteration % 3 == 0:
                    init_strategy = 'empty'
                elif iteration % 3 == 1:
                    init_strategy = 'random'
                else:
                    init_strategy = 'risk'
            else:
                init_strategy = self.initialization
            
            # Construction phase
            solution = self._construction_phase(init_strategy, current_alpha)
            
            # VND Local search phase
            solution = self._vnd_local_search(solution)
            
            # Update alpha statistics
            if solution.is_feasible:
                self.alpha_manager.update_statistics(current_alpha, solution.objective_value)
            
            # Update best solution
            if solution.is_feasible and solution.objective_value < self.best_objective:
                self.best_solution = solution.copy()
                self.best_objective = solution.objective_value
                iter_no_improvement = 0  # Reset counter
                
                # Calculate improvement percentage
                initial_risk = sum(self.instance.risks)
                improvement = (1 - self.best_objective / initial_risk) * 100
                
                print(f"  Iter {iteration+1:3d}: New best = {self.best_objective:12.6f} "
                      f"(α={current_alpha:.2f}, Risk reduction: {improvement:5.2f}%)")
            else:
                iter_no_improvement += 1
        
        elapsed_time = time.time() - start_time
        
        print(f"\n{'='*70}")
        print(f"REACTIVE GRASP + VND completed in {elapsed_time:.2f} seconds")
        print(f"Best objective value: {self.best_objective:.6f}")
        
        initial_risk = sum(self.instance.risks)
        if self.best_objective < float('inf'):
            final_improvement = (1 - self.best_objective / initial_risk) * 100
            print(f"Final risk reduction: {final_improvement:.2f}%")
        
        # Print alpha statistics
        print(f"\nAlpha Performance:")
        for i, alpha in enumerate(self.alpha_manager.alpha_values):
            count = self.alpha_manager.alpha_count[i]
            prob = self.alpha_manager.probabilities[i]
            print(f"  α={alpha:.2f}: used {count:3d} times, P={prob:.3f}")
        
        print(f"{'='*70}")
        
        return self.best_solution, elapsed_time
    
    def _find_subset_by_interventions(self, student_id, target_interventions):
        """
        Find the index of a subset that matches the target interventions.
        
        Args:
            student_id: Student ID
            target_interventions: Set or tuple of intervention IDs
        
        Returns:
            Subset index if found, None otherwise
        """
        target_set = set(target_interventions)
        for idx, subset in enumerate(self.instance.subsets[student_id]):
            if set(subset) == target_set:
                return idx
        return None
    
    def _neighborhood_intervention_add_remove(self, solution, evaluator, first_improving=True):
        """
        N3: Intervention Add/Remove neighborhood.
        For each student, try adding or removing a single intervention.
        This provides finer-grained changes compared to full subset swaps.
        
        Args:
            solution: Current solution
            evaluator: Incremental evaluator
            first_improving: If True, accept first improvement; otherwise find best
        
        Returns:
            True/False if first_improving, or best_move tuple otherwise
        """
        n_students_sample = min(self.instance.n_students, 150)
        students = random.sample(range(self.instance.n_students), n_students_sample)
        
        best_move = None
        best_delta = 0.0
        
        for student_id in students:
            current_subset_idx = solution.assignment[student_id]
            current_subset = set(self.instance.subsets[student_id][current_subset_idx])
            
            # Try ADDING interventions not in current subset
            for intervention_id in range(self.instance.n_interventions):
                if intervention_id not in current_subset:
                    # Create new subset with this intervention added
                    new_subset = current_subset | {intervention_id}
                    new_subset_idx = self._find_subset_by_interventions(student_id, new_subset)
                    
                    if new_subset_idx is not None:
                        delta_risk, _, is_feasible = evaluator.evaluate_delta(
                            student_id, new_subset_idx
                        )
                        
                        if is_feasible and delta_risk < -1e-9:
                            if first_improving:
                                evaluator.apply_move(student_id, new_subset_idx)
                                return True
                            elif delta_risk < best_delta:
                                best_delta = delta_risk
                                best_move = (delta_risk, student_id, new_subset_idx)
            
            # Try REMOVING interventions from current subset
            if len(current_subset) > 0:  # Only if subset is not empty
                for intervention_id in current_subset:
                    # Create new subset with this intervention removed
                    new_subset = current_subset - {intervention_id}
                    new_subset_idx = self._find_subset_by_interventions(student_id, new_subset)
                    
                    if new_subset_idx is not None:
                        delta_risk, _, is_feasible = evaluator.evaluate_delta(
                            student_id, new_subset_idx
                        )
                        
                        if is_feasible and delta_risk < -1e-9:
                            if first_improving:
                                evaluator.apply_move(student_id, new_subset_idx)
                                return True
                            elif delta_risk < best_delta:
                                best_delta = delta_risk
                                best_move = (delta_risk, student_id, new_subset_idx)
        
        if first_improving:
            return False
        else:
            if best_move:
                _, student_id, new_subset_idx = best_move
                evaluator.apply_move(student_id, new_subset_idx)
                return True
            return False
    
    def _vnd_local_search(self, solution):
        """
        Variable Neighborhood Descent: systematically explore neighborhoods.
        """
        evaluator = IncrementalEvaluator(self.instance, solution)
        
        # Three neighborhood structures (N3)
        neighborhoods = [
            self._neighborhood_intervention_add_remove,  # N3
            self._neighborhood_subset_swap,              # N1
            self._neighborhood_student_swap              # N2
        ]
        
        k = 0  # Current neighborhood index
        
        # VND main loop
        while k < len(neighborhoods):
            if self.local_search == 'first':
                improved = neighborhoods[k](solution, evaluator, first_improving=True)
            else:
                improved = neighborhoods[k](solution, evaluator, first_improving=False)
            
            if improved:
                k = 0  # Restart from first neighborhood
            else:
                k += 1  # Try next neighborhood
        
        solution.evaluate()
        return solution
    
    def _neighborhood_subset_swap(self, solution, evaluator, first_improving=True):
        """
        Subset swap neighborhood: change subset assignment for single student.
        """
        n_students_sample = min(self.instance.n_students, 200)
        students = random.sample(range(self.instance.n_students), n_students_sample)
        
        best_move = None
        best_delta = 0.0
        
        for student_id in students:
            current_subset_idx = solution.assignment[student_id]
            
            # Try subsets in random order
            subset_indices = list(range(len(self.instance.subsets[student_id])))
            random.shuffle(subset_indices)
            subset_indices = subset_indices[:min(20, len(subset_indices))]
            
            for new_subset_idx in subset_indices:
                if new_subset_idx == current_subset_idx:
                    continue
                
                delta_risk, delta_cost, is_feasible = evaluator.evaluate_delta(
                    student_id, new_subset_idx
                )
                
                if is_feasible and delta_risk < -1e-9:
                    if first_improving:
                        # Accept first improvement
                        evaluator.apply_move(student_id, new_subset_idx)
                        return True
                    else:
                        # Track best improvement
                        if delta_risk < best_delta:
                            best_delta = delta_risk
                            best_move = (delta_risk, student_id, new_subset_idx)
        
        if first_improving:
            return False
        else:
            if best_move:
                _, student_id, new_subset_idx = best_move
                evaluator.apply_move(student_id, new_subset_idx)
                return True
            return False
    
    def _neighborhood_student_swap(self, solution, evaluator, first_improving=True):
        """
        Student swap neighborhood: coordinated changes between two students.
        """
        n_pairs_sample = min(self.instance.n_students, 100)
        
        best_move = None
        best_delta = 0.0
        
        for _ in range(n_pairs_sample):
            # Pick two random students
            student1 = random.randint(0, self.instance.n_students - 1)
            student2 = random.randint(0, self.instance.n_students - 1)
            
            if student1 == student2:
                continue
            
            current_idx1 = solution.assignment[student1]
            current_idx2 = solution.assignment[student2]
            
            # Try a few random swaps
            for _ in range(3):
                eligible1 = list(range(len(self.instance.subsets[student1])))
                eligible2 = list(range(len(self.instance.subsets[student2])))
                
                if not eligible1 or not eligible2:
                    continue
                
                new_idx1 = random.choice(eligible1)
                new_idx2 = random.choice(eligible2)
                
                if new_idx1 == current_idx1 and new_idx2 == current_idx2:
                    continue
                
                # Evaluate double move
                delta1_risk, _, feas1 = evaluator.evaluate_delta(student1, new_idx1)
                if not feas1:
                    continue
                
                # Apply first move temporarily
                evaluator.apply_move(student1, new_idx1)
                delta2_risk, _, feas2 = evaluator.evaluate_delta(student2, new_idx2)
                # Revert first move
                evaluator.apply_move(student1, current_idx1)
                
                if not feas2:
                    continue
                
                total_delta = delta1_risk + delta2_risk
                
                if total_delta < -1e-9:
                    if first_improving:
                        # Apply both moves
                        evaluator.apply_move(student1, new_idx1)
                        evaluator.apply_move(student2, new_idx2)
                        return True
                    else:
                        if total_delta < best_delta:
                            best_delta = total_delta
                            best_move = (total_delta, student1, new_idx1, student2, new_idx2)
        
        if first_improving:
            return False
        else:
            if best_move:
                _, student1, new_idx1, student2, new_idx2 = best_move
                evaluator.apply_move(student1, new_idx1)
                evaluator.apply_move(student2, new_idx2)
                return True
            return False
    
    def _construction_phase(self, init_strategy='empty', alpha=0.1):
        """
        Construction phase with benefit-to-cost ratio and risk prioritization.
        """
        solution = Solution(self.instance)
        
        # Apply initialization strategy
        if init_strategy == 'random':
            self._initialize_random(solution)
        elif init_strategy == 'risk':
            self._initialize_risk_based(solution)
        
        evaluator = IncrementalEvaluator(self.instance, solution)
        solution.evaluate()
        
        # Build solution using benefit-to-cost ratio
        candidate_students = self.instance.students_by_risk.copy()
        
        max_iterations = len(candidate_students) * 2
        iterations_count = 0
        
        while candidate_students and iterations_count < max_iterations:
            iterations_count += 1
            
            # Build RCL with benefit-to-cost candidates
            rcl_candidates = self._build_rcl_benefit_cost(
                solution, evaluator, candidate_students, alpha
            )
            
            if not rcl_candidates:
                break
            
            # Select random candidate from RCL
            selected = random.choice(rcl_candidates)
            _, student_id, new_subset_idx = selected
            
            # Apply assignment
            evaluator.apply_move(student_id, new_subset_idx)
            
            # Remove if non-empty subset assigned
            if new_subset_idx != 0:
                candidate_students.remove(student_id)
        
        solution.evaluate()
        return solution
    
    def _initialize_random(self, solution):
        """Initialize solution with random assignments for some students."""
        n_random = min(10, self.instance.n_students // 10)
        students = random.sample(range(self.instance.n_students), n_random)
        
        for student_id in students:
            eligible = [i for i in range(len(self.instance.subsets[student_id])) if i != 0]
            if eligible:
                solution.assignment[student_id] = random.choice(eligible)
    
    def _initialize_risk_based(self, solution):
        """Initialize solution focusing on high-risk students."""
        n_high_risk = self.instance.n_students // 5
        high_risk_students = self.instance.students_by_risk[:n_high_risk]
        
        for student_id in high_risk_students[:len(high_risk_students)//2]:
            eligible = [i for i in range(len(self.instance.subsets[student_id])) if i != 0]
            if eligible:
                best_idx = 0
                best_ratio = -float('inf')
                
                for idx in eligible[:5]:
                    eff = self.instance.effectiveness[student_id][idx]
                    cost = self.instance.subset_costs[(student_id, idx)]
                    ratio = (eff * self.instance.risks[student_id]) / (cost + 1.0)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_idx = idx
                
                solution.assignment[student_id] = best_idx
    
    def _build_rcl_benefit_cost(self, solution, evaluator, candidate_students, alpha):
        """Build RCL using benefit-to-cost ratio."""
        candidates = []
        
        # Sample students to evaluate
        n_students_to_sample = min(len(candidate_students), 100)
        sampled_students = random.sample(candidate_students, n_students_to_sample)
        
        for student_id in sampled_students:
            current_subset_idx = solution.assignment[student_id]
            
            # Get eligible subsets
            eligible_subsets = []
            for idx in range(len(self.instance.subsets[student_id])):
                if idx != current_subset_idx:
                    eligible_subsets.append(idx)
            
            # Sort by effectiveness and take top K
            eligible_subsets.sort(
                key=lambda idx: self.instance.effectiveness[student_id][idx],
                reverse=True
            )
            eligible_subsets = eligible_subsets[:self.max_candidates_per_student]
            
            # Evaluate each eligible subset
            for new_subset_idx in eligible_subsets:
                delta_risk, delta_cost, is_feasible = evaluator.evaluate_delta(
                    student_id, new_subset_idx
                )
                
                if not is_feasible:
                    continue
                
                # Benefit-to-cost ratio
                benefit = -delta_risk
                cost = abs(delta_cost) + 1.0
                
                if benefit > 0:
                    ratio = benefit / cost
                    candidates.append((ratio, student_id, new_subset_idx))
                elif benefit > -0.2 and delta_cost < 0:
                    score = abs(benefit) / (abs(delta_cost) + 1.0) * 0.5
                    candidates.append((score, student_id, new_subset_idx))
        
        if not candidates:
            return []
        
        # Build RCL based on alpha
        candidates.sort(key=lambda x: x[0], reverse=True)
        
        best_ratio = candidates[0][0]
        worst_ratio = candidates[-1][0]
        
        threshold = best_ratio - alpha * (best_ratio - worst_ratio)
        rcl = [c for c in candidates if c[0] >= threshold]
        
        return rcl


def save_metrics(solution, elapsed_time, output_file):
    """Save solution metrics to file."""
    instance = solution.instance
    
    # Calculate metrics
    initial_risk = sum(instance.risks)
    final_risk = solution.objective_value
    risk_reduction = initial_risk - final_risk
    risk_reduction_pct = (risk_reduction / initial_risk) * 100
    
    cost_details = solution.get_cost_details()
    
    students_served = sum(1 for s in range(instance.n_students) 
                         if solution.assignment[s] != 0)
    students_served_pct = (students_served / instance.n_students) * 100
    
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("REACTIVE GRASP + VND SOLUTION METRICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OPTIMIZATION RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Status: FEASIBLE\n")
        f.write(f"Solve time: {elapsed_time:.2f} seconds\n\n")
        
        f.write("RISK REDUCTION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Initial total risk: {initial_risk:.6f}\n")
        f.write(f"Final total risk: {final_risk:.6f}\n")
        f.write(f"Risk reduction: {risk_reduction:.6f}\n")
        f.write(f"Risk reduction: {risk_reduction_pct:.2f}%\n\n")
        
        f.write("BUDGET AND COSTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Available budget: ${instance.budget:,.2f}\n")
        f.write(f"Total cost: ${cost_details['total_cost']:,.2f}\n")
        budget_used_pct = (cost_details['total_cost'] / instance.budget) * 100
        f.write(f"Budget used: {budget_used_pct:.2f}%\n")
        f.write(f"  Variable costs: ${cost_details['variable_cost']:,.2f}\n")
        f.write(f"  Fixed costs: ${cost_details['fixed_cost']:,.2f}\n\n")
        
        f.write("ALLOCATION SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Students served: {students_served:,} ({students_served_pct:.2f}%)\n")
        f.write(f"Interventions activated: {cost_details['activated_interventions']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\nMetrics saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Reactive GRASP + VND for School Dropout Problem",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--instance",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--iterations",
        type=int,
        default=100
    )
    
    parser.add_argument(
        "--local-search",
        type=str,
        choices=['first', 'best'],
        default='first'
    )
    
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=10
    )
    
    parser.add_argument(
        "--initialization",
        type=str,
        choices=['empty', 'random', 'risk', 'varied'],
        default='varied'
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )
    
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--max-time",
        type=float,
        default=None
    )
    
    parser.add_argument(
        "--max-iter-no-improvement",
        type=int,
        default=100
    )
    
    args = parser.parse_args()
    
    # Determine output filename
    instance_basename = Path(args.instance).stem
    if args.output_metrics is None:
        args.output_metrics = f"metrics_reactive_vnd_{instance_basename}.txt"
    
    print("=" * 70)
    print("REACTIVE GRASP + VND - SCHOOL DROPOUT PROBLEM")
    print("=" * 70)
    print(f"\nInstance: {args.instance}")
    print(f"Iterations: {args.iterations}")
    print(f"Local search: {args.local_search.upper()}")
    print(f"Max candidates per student: {args.max_candidates}")
    print(f"Initialization: {args.initialization.upper()}")
    print(f"Random seed: {args.seed}")
    
    # Load instance
    instance = SchoolDropoutInstance(args.instance)
    
    # Run Reactive GRASP + VND
    grasp_vnd = ReactiveGRASP_VND(
        instance=instance,
        iterations=args.iterations,
        local_search=args.local_search,
        max_candidates_per_student=args.max_candidates,
        initialization=args.initialization,
        seed=args.seed,
        max_time=args.max_time,
        max_iter_no_improvement=args.max_iter_no_improvement
    )
    
    solution, elapsed_time = grasp_vnd.solve()
    
    if solution is None or not solution.is_feasible:
        print("\nERROR: No feasible solution found")
        return 1
    
    # Save outputs
    save_metrics(solution, elapsed_time, args.output_metrics)
    
    print("\n" + "=" * 70)
    print("REACTIVE GRASP + VND COMPLETED SUCCESSFULLY")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())