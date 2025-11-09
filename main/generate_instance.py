#!/usr/bin/env python3
"""
Instance Generator for School Dropout Problem
"""

import argparse
import random
import numpy as np
from pathlib import Path
from itertools import combinations
import json


# Intervention categories based on literature
INTERVENTION_TYPES = [
    {"id": 0, "name": "financial_aid", "category": "economic"},
    {"id": 1, "name": "transportation", "category": "logistic"},
    {"id": 2, "name": "tutoring", "category": "academic"},
    {"id": 3, "name": "mentoring", "category": "psychosocial"},
    {"id": 4, "name": "nutrition", "category": "health"},
    {"id": 5, "name": "counseling", "category": "psychosocial"},
    {"id": 6, "name": "extended_day", "category": "academic"},
    {"id": 7, "name": "family_engagement", "category": "social"},
]


def load_risks(risk_file):
    """Load initial dropout risks from file."""
    risks = []
    with open(risk_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) == 2:
                    risks.append(float(parts[1]))
    
    return risks


def generate_fixed_costs(n_interventions, seed=None):
    """
    Generate fixed costs for interventions.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Different cost levels for different categories
    base_costs = {
        "economic": (5000, 15000),      # High setup (system development)
        "logistic": (20000, 50000),     # Very high (vehicle acquisition)
        "academic": (3000, 8000),       # Medium (materials, training)
        "psychosocial": (2000, 6000),   # Low-medium (professional training)
        "health": (10000, 25000),       # High (kitchen infrastructure)
        "social": (4000, 10000),        # Medium (program development)
    }
    
    fixed_costs = []
    for intervention in INTERVENTION_TYPES:
        category = intervention["category"]
        min_cost, max_cost = base_costs[category]
        # Use log-normal to create asymmetry
        cost = np.random.uniform(min_cost, max_cost)
        fixed_costs.append(cost)
    
    return fixed_costs


def generate_variable_costs(n_students, n_interventions, risks, seed=None):
    """
    Generate variable costs.
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Base variable costs per intervention per student
    base_var_costs = {
        "economic": (200, 500),         # Scholarship/aid per student
        "logistic": (50, 150),          # Transport per student
        "academic": (100, 300),         # Tutoring hours per student
        "psychosocial": (150, 400),     # Mentoring sessions per student
        "health": (80, 200),            # Meals per student
        "social": (100, 250),           # Family engagement activities
    }
    
    var_costs = np.zeros((n_interventions, n_students))
    
    for i, intervention in enumerate(INTERVENTION_TYPES):
        category = intervention["category"]
        min_cost, max_cost = base_var_costs[category]
        
        for student in range(n_students):
            # Base cost with some randomness
            base = np.random.uniform(min_cost, max_cost)
            
            risk_factor = 1.0 + (risks[student] * 0.5) 
            
            var_costs[i, student] = base * risk_factor
    
    return var_costs


def generate_capacities(n_interventions, n_students, seed=None):
    """
    Generate capacities
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Capacity as percentage of total students
    capacity_ranges = {
        "economic": (0.30, 0.50),       
        "logistic": (0.15, 0.25),       
        "academic": (0.20, 0.35),       
        "psychosocial": (0.10, 0.20),   
        "health": (0.40, 0.60),         
        "social": (0.25, 0.40),         
    }
    
    capacities = []
    for intervention in INTERVENTION_TYPES:
        category = intervention["category"]
        min_pct, max_pct = capacity_ranges[category]
        pct = np.random.uniform(min_pct, max_pct)
        capacity = int(n_students * pct)
        capacities.append(capacity)
    
    return capacities


def generate_intervention_subsets(n_students, n_interventions, risks, seed=None):
    """
    Generate intervention subsets S_a for each student.

    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    subsets = []
    
    for student in range(n_students):
        risk = risks[student]
        
        # Determine eligible interventions based on risk and random factors
        eligible = []
        for i in range(n_interventions):
            # Higher risk students are eligible for more interventions
            eligibility_prob = 0.5 + (risk * 0.4)  
            if random.random() < eligibility_prob:
                eligible.append(i)
        
        # Ensure at least 3 interventions are eligible
        while len(eligible) < 3:
            eligible.append(random.randint(0, n_interventions - 1))
        eligible = list(set(eligible))
        
        # Generate subsets: empty set + individual + pairs + some triples
        student_subsets = [tuple()] 
        
        # All individual interventions
        for i in eligible:
            student_subsets.append((i,))
        
        # Pairs
        if len(eligible) >= 2:
            all_pairs = list(combinations(eligible, 2))
            # Take 70% of possible pairs randomly
            n_pairs = max(1, int(len(all_pairs) * 0.7))
            pairs = random.sample(all_pairs, n_pairs)
            student_subsets.extend(pairs)
        
        # Triples
        if risk > 0.5 and len(eligible) >= 3:
            all_triples = list(combinations(eligible, 3))
            # Take only 30% of possible triples
            n_triples = max(1, int(len(all_triples) * 0.3))
            triples = random.sample(all_triples, min(n_triples, len(all_triples)))
            student_subsets.extend(triples)
        
        # Remove duplicates and sort
        student_subsets = sorted(list(set(student_subsets)))
        subsets.append(student_subsets)
    
    return subsets


def calculate_effectiveness(subset, student_risk, intervention_synergies, seed_base=0):
    """
    Calculate effectiveness with non-additive effects.
    """
    if len(subset) == 0:
        return 0.0
    
    # Base effectiveness: higher risk students have more room for improvement
    base_effectiveness = student_risk * 0.7  
    
    # Single intervention
    if len(subset) == 1:
        # Random variation per intervention type
        np.random.seed(seed_base + subset[0])
        variation = np.random.uniform(0.4, 0.7)
        return base_effectiveness * variation
    
    # Multiple interventions
    effectiveness = 0.0
    for idx, intervention in enumerate(sorted(subset)):
        np.random.seed(seed_base + intervention)
        variation = np.random.uniform(0.4, 0.7)
        
        # Diminishing returns: each additional intervention is less effective
        discount = 0.7 ** idx
        effectiveness += base_effectiveness * variation * discount
    
    # Apply synergies/antagonisms
    if len(subset) >= 2:
        for i, int1 in enumerate(subset):
            for int2 in list(subset)[i+1:]:
                key = tuple(sorted([int1, int2]))
                if key in intervention_synergies:
                    synergy = intervention_synergies[key]
                    effectiveness *= (1.0 + synergy)
    
    # Cap effectiveness at reducing risk by 95%
    return min(0.95 * student_risk, effectiveness)


def generate_synergies(n_interventions, seed=None):
    """
    Generate synergy/antagonism matrix for intervention pairs.
    Positive values = synergy, negative = antagonism.
    """
    if seed is not None:
        np.random.seed(seed)
    
    synergies = {}
    
    # Define synergies based on intervention categories
    for i in range(n_interventions):
        for j in range(i+1, n_interventions):
            cat_i = INTERVENTION_TYPES[i]["category"]
            cat_j = INTERVENTION_TYPES[j]["category"]
            
            # Same category: slight antagonism (overlap)
            if cat_i == cat_j:
                synergies[(i, j)] = np.random.uniform(-0.15, -0.05)
            
            # Complementary categories: synergy
            elif (cat_i == "academic" and cat_j == "psychosocial") or \
                 (cat_i == "economic" and cat_j in ["logistic", "health"]) or \
                 (cat_i == "social" and cat_j == "psychosocial"):
                synergies[(i, j)] = np.random.uniform(0.05, 0.20)
            
            # Default: small random effect
            else:
                synergies[(i, j)] = np.random.uniform(-0.05, 0.10)
    
    return synergies


def calculate_budget(var_costs, fixed_costs, capacities, risks, budget_factor):
    """
    Calculate budget that allows covering approximately budget_factor of high-risk students.
    """
    n_interventions, n_students = var_costs.shape
    
    # Identify high-risk students (top 40%)
    risk_threshold = np.percentile(risks, 60)
    high_risk_students = [i for i, r in enumerate(risks) if r >= risk_threshold]
    n_high_risk = len(high_risk_students)
    
    # Calculate target students to cover
    target_students = int(n_high_risk * budget_factor)
    
    # Estimate cost: assume each target student gets 1-2 interventions on average
    avg_interventions_per_student = 1.5
    
    # Calculate average cost per intervention for high-risk students
    avg_var_cost = np.mean([var_costs[:, s].mean() for s in high_risk_students])
    
    # Total variable cost
    total_var_cost = target_students * avg_interventions_per_student * avg_var_cost
    
    # Fixed costs: assume we use about 60% of available interventions
    total_fixed_cost = sum(fixed_costs) * 0.6
    
    # Total budget with some random variation
    budget = total_var_cost + total_fixed_cost
    budget *= np.random.uniform(0.9, 1.1)  # ±10% variation
    
    return budget


def save_instance(output_file, n_students, n_interventions, risks, fixed_costs, 
                  var_costs, capacities, subsets, effectiveness_params, budget, synergies):
    """Save complete instance to file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        # Header
        f.write("# School Dropout Problem - Complete Instance\n")
        f.write(f"# Students: {n_students}\n")
        f.write(f"# Interventions: {n_interventions}\n")
        f.write(f"# Budget: {budget:.2f}\n")
        f.write("#\n")
        
        # Intervention information
        f.write("# INTERVENTIONS\n")
        f.write("# Format: id name category fixed_cost capacity\n")
        for i, intervention in enumerate(INTERVENTION_TYPES):
            f.write(f"INTERVENTION {i} {intervention['name']} {intervention['category']} "
                   f"{fixed_costs[i]:.2f} {capacities[i]}\n")
        f.write("#\n")
        
        # Budget
        f.write("# BUDGET\n")
        f.write(f"BUDGET {budget:.2f}\n")
        f.write("#\n")
        
        # Student risks
        f.write("# STUDENT RISKS\n")
        f.write("# Format: student_id risk\n")
        for s, risk in enumerate(risks):
            f.write(f"RISK {s} {risk:.6f}\n")
        f.write("#\n")
        
        # Variable costs
        f.write("# VARIABLE COSTS\n")
        f.write("# Format: intervention_id student_id cost\n")
        for i in range(n_interventions):
            for s in range(n_students):
                f.write(f"VAR_COST {i} {s} {var_costs[i, s]:.2f}\n")
        f.write("#\n")
        
        # Intervention subsets per student
        f.write("# INTERVENTION SUBSETS\n")
        f.write("# Format: student_id num_subsets [subset_interventions]\n")
        for s, student_subsets in enumerate(subsets):
            f.write(f"SUBSETS {s} {len(student_subsets)}")
            for subset in student_subsets:
                f.write(f" {len(subset)}")
                for intervention in subset:
                    f.write(f" {intervention}")
            f.write("\n")
        f.write("#\n")
        
        # Effectiveness values
        f.write("# EFFECTIVENESS\n")
        f.write("# Format: student_id subset_index effectiveness\n")
        for s, student_subsets in enumerate(subsets):
            for idx, subset in enumerate(student_subsets):
                eff = calculate_effectiveness(subset, risks[s], synergies, seed_base=s*1000)
                f.write(f"EFFECTIVENESS {s} {idx} {eff:.6f}\n")
        f.write("#\n")
        
        # Synergies
        f.write("# SYNERGIES\n")
        f.write("# Format: intervention_i intervention_j synergy_value\n")
        for (i, j), synergy in synergies.items():
            f.write(f"SYNERGY {i} {j} {synergy:.6f}\n")
    
    print(f"Instance saved to: {output_file}")


def print_statistics(n_students, n_interventions, risks, fixed_costs, var_costs, 
                     capacities, subsets, budget):
    """Print instance statistics."""
    print(f"\n{'='*60}")
    print(f"INSTANCE STATISTICS")
    print(f"{'='*60}")
    print(f"Students: {n_students:,}")
    print(f"Interventions: {n_interventions}")
    print(f"\nBudget: ${budget:,.2f}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Generate complete School Dropout Problem instance",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--risks",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--budget-factor",
        type=float,
        default=0.3
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None
    )
    
    args = parser.parse_args()
    
    # Set random seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Determine output filename
    if args.output is None:
        risk_basename = Path(args.risks).stem
        args.output = f"instance_{risk_basename.replace('risks_', '')}.txt"
    
    risks = load_risks(args.risks)
    n_students = len(risks)
    n_interventions = len(INTERVENTION_TYPES)
    
    # Generate instance components
    fixed_costs = generate_fixed_costs(n_interventions, seed=args.seed)

    var_costs = generate_variable_costs(n_students, n_interventions, risks, seed=args.seed)
    
    capacities = generate_capacities(n_interventions, n_students, seed=args.seed)

    subsets = generate_intervention_subsets(n_students, n_interventions, risks, seed=args.seed)
    
    synergies = generate_synergies(n_interventions, seed=args.seed)

    budget = calculate_budget(var_costs, fixed_costs, capacities, risks, args.budget_factor)
    
    # Print statistics
    print_statistics(n_students, n_interventions, risks, fixed_costs, var_costs, 
                    capacities, subsets, budget)
    
    # Save instance
    print(f"Saving instance to: {args.output}")
    effectiveness_params = {"synergies": synergies}
    save_instance(args.output, n_students, n_interventions, risks, fixed_costs,
                  var_costs, capacities, subsets, effectiveness_params, budget, synergies)


if __name__ == "__main__":
    main()
