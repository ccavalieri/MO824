#!/usr/bin/env python3
"""
Integer Linear Programming Solver for School Dropout Problem
"""

import argparse
import time
import sys
from pathlib import Path
import gurobipy as gp
from gurobipy import GRB

# Gurobi WLS credentials
WLSACCESSID = "98f532ed-4ce0-43ae-ad8b-338b6bf81b0d"
WLSSECRET = "bee31482-7e08-4d1d-8f14-8a8455182d1e"
LICENSEID = 2697514


def parse_instance(filename):
    """
    Parse instance file and return all problem data.
    Returns: (n_students, n_interventions, risks, fixed_costs, var_costs, 
              capacities, subsets, effectiveness, budget)
    """
    
    n_students = 0
    n_interventions = 0
    budget = 0.0
    
    fixed_costs = []
    capacities = []
    risks = []
    var_costs = {}
    subsets = {}
    effectiveness = {}
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Parse header
    for line in lines[:10]:
        if line.startswith("# Students:"):
            n_students = int(line.split(":")[1].strip())
        elif line.startswith("# Interventions:"):
            n_interventions = int(line.split(":")[1].strip())
        elif line.startswith("# Budget:"):
            budget = float(line.split(":")[1].strip())
    
    print(f"  Students: {n_students:,}")
    print(f"  Interventions: {n_interventions}")
    print(f"  Budget: ${budget:,.2f}")
    
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
            fixed_costs.append(fixed_cost)
            capacities.append(capacity)
        
        elif parts[0] == "BUDGET":
            budget = float(parts[1])
        
        elif parts[0] == "RISK":
            student_id = int(parts[1])
            risk = float(parts[2])
            risks.append(risk)
        
        elif parts[0] == "VAR_COST":
            intervention_id = int(parts[1])
            student_id = int(parts[2])
            cost = float(parts[3])
            var_costs[(intervention_id, student_id)] = cost
        
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
            
            subsets[student_id] = student_subsets
        
        elif parts[0] == "EFFECTIVENESS":
            student_id = int(parts[1])
            subset_idx = int(parts[2])
            eff = float(parts[3])
            
            if student_id not in effectiveness:
                effectiveness[student_id] = {}
            effectiveness[student_id][subset_idx] = eff
    
    print(f"  Parsed {len(risks)} student risks")
    print(f"  Parsed {len(var_costs)} variable costs")
    print(f"  Parsed {sum(len(s) for s in subsets.values())} total subsets")
    
    return (n_students, n_interventions, risks, fixed_costs, var_costs, 
            capacities, subsets, effectiveness, budget)


def build_and_solve_ilp(n_students, n_interventions, risks, fixed_costs, var_costs,
                        capacities, subsets, effectiveness, budget, 
                        time_limit=600, mip_gap=0.001):
    """
    Build and solve the ILP formulation using Gurobi.
    """
    
    print("\n" + "="*70)
    print("BUILDING ILP MODEL")
    print("="*70)
    
    # Try to create Gurobi environment
    env = None
    model = None
    
    try:
        # Try WLS credentials
        print("\nAttempting to connect to Gurobi WLS...")
        env = gp.Env(params={
            'WLSACCESSID': WLSACCESSID,
            'WLSSECRET': WLSSECRET,
            'LICENSEID': LICENSEID,
            'OutputFlag': 1
        })
        model = gp.Model("SchoolDropoutProblem", env=env)
        print("Successfully connected to Gurobi WLS")
    except gp.GurobiError as e:
        print(f"WLS connection failed: {e}")
    
    if model is None:
        print("\nERROR: Failed to create Gurobi model")
        return None
    
    # Set parameters
    model.setParam('TimeLimit', time_limit)
    model.setParam('MIPGap', mip_gap)
    model.setParam('LogToConsole', 0)
    
    # ========== VARIABLES ==========
    
    # x_{i,a}: student a receives intervention i
    x = {}
    for i in range(n_interventions):
        for a in range(n_students):
            x[i, a] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{a}")
    
    # y_{s,a}: student a receives intervention subset s
    y = {}
    for a in range(n_students):
        for s_idx in range(len(subsets[a])):
            y[s_idx, a] = model.addVar(vtype=GRB.BINARY, name=f"y_{s_idx}_{a}")
    
    # z_i: intervention i is activated
    z = {}
    for i in range(n_interventions):
        z[i] = model.addVar(vtype=GRB.BINARY, name=f"z_{i}")
    
    # ========== OBJECTIVE FUNCTION ==========
    
    # Objective: min Σ_a Σ_{s∈S_a} r_a * (1 - e_{s,a}) * y_{s,a}
    obj = gp.QuadExpr()
    for a in range(n_students):
        for s_idx in range(len(subsets[a])):
            e_sa = effectiveness[a][s_idx]
            coeff = risks[a] * (1.0 - e_sa)
            obj += coeff * y[s_idx, a]
    
    model.setObjective(obj, GRB.MINIMIZE)
    
    # ========== CONSTRAINTS ==========
    
    # (2) Budget constraint
    budget_expr = gp.LinExpr()
    for i in range(n_interventions):
        for a in range(n_students):
            if (i, a) in var_costs:
                budget_expr += var_costs[i, a] * x[i, a]
    for i in range(n_interventions):
        budget_expr += fixed_costs[i] * z[i]
    
    model.addConstr(budget_expr <= budget, name="budget")
    
    # (3) Capacity constraints
    capacity_constrs = 0
    for i in range(n_interventions):
        capacity_expr = gp.LinExpr()
        for a in range(n_students):
            capacity_expr += x[i, a]
        model.addConstr(capacity_expr <= capacities[i], name=f"capacity_{i}")
        capacity_constrs += 1
    
    # (4) Subset selection constraints
    subset_constrs = 0
    for a in range(n_students):
        subset_expr = gp.LinExpr()
        for s_idx in range(len(subsets[a])):
            subset_expr += y[s_idx, a]
        model.addConstr(subset_expr == 1, name=f"subset_select_{a}")
        subset_constrs += 1
    
    # (5) Consistency constraints
    consistency_constrs = 0
    for a in range(n_students):
        for i in range(n_interventions):
            # x_{i,a} = Σ_{s∈S_a: i∈s} y_{s,a}
            rhs = gp.LinExpr()
            for s_idx, subset in enumerate(subsets[a]):
                if i in subset:
                    rhs += y[s_idx, a]
            model.addConstr(x[i, a] == rhs, name=f"consistency_{i}_{a}")
            consistency_constrs += 1
    
    # (6) Activation constraints
    activation_constrs = 0
    for i in range(n_interventions):
        for a in range(n_students):
            model.addConstr(x[i, a] <= z[i], name=f"activation_{i}_{a}")
            activation_constrs += 1
    
    total_constrs = (1 + capacity_constrs + subset_constrs + 
                     consistency_constrs + activation_constrs)
    
    # ========== SOLVE ==========
    
    start_time = time.time()
    model.optimize()
    solve_time = time.time() - start_time
    
    # ========== EXTRACT SOLUTION ==========
    
    status = model.status
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INFEASIBLE_OR_UNBOUNDED",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED"
    }
    status_str = status_map.get(status, f"UNKNOWN({status})")
    
    print(f"\nStatus: {status_str}")
    print(f"Solve time: {solve_time:.2f}s")
    
    if status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        print(f"\nERROR: Model did not find a solution (status: {status_str})")
        return None
    
    # Extract solution
    solution = {
        'status': status_str,
        'solve_time': solve_time,
        'objective_value': model.objVal if model.SolCount > 0 else None,
        'mip_gap': model.MIPGap if model.SolCount > 0 else None,
        'x_values': {},
        'y_values': {},
        'z_values': {}
    }
    
    if model.SolCount > 0:
        for (i, a), var in x.items():
            if var.X > 0.5:  # Binary variable is 1
                solution['x_values'][(i, a)] = 1
        
        for (s_idx, a), var in y.items():
            if var.X > 0.5:
                solution['y_values'][(s_idx, a)] = 1
        
        for i, var in z.items():
            if var.X > 0.5:
                solution['z_values'][i] = 1
        
        print(f"Objective value: {solution['objective_value']:.6f}")
        print(f"MIP gap: {solution['mip_gap']*100:.4f}%")
    
    return solution, model, (n_students, n_interventions, risks, fixed_costs, 
                             var_costs, capacities, subsets, effectiveness, budget)


def calculate_metrics(solution, n_students, risks, fixed_costs, var_costs, 
                     capacities, subsets, effectiveness, budget):
    """Calculate solution metrics."""
    
    if solution['objective_value'] is None:
        return None
    
    # Initial total risk
    initial_risk = sum(risks)
    
    # Final total risk (objective value)
    final_risk = solution['objective_value']
    
    # Risk reduction
    risk_reduction = initial_risk - final_risk
    risk_reduction_pct = (risk_reduction / initial_risk) * 100
    
    # Total cost
    x_vals = solution['x_values']
    z_vals = solution['z_values']
    
    total_var_cost = sum(var_costs.get((i, a), 0) for (i, a) in x_vals)
    total_fixed_cost = sum(fixed_costs[i] for i in z_vals)
    total_cost = total_var_cost + total_fixed_cost
    
    # Students served
    students_served = len(set(a for (i, a) in x_vals))
    
    # Interventions activated
    interventions_activated = len(z_vals)
    
    metrics = {
        'initial_risk': initial_risk,
        'final_risk': final_risk,
        'risk_reduction': risk_reduction,
        'risk_reduction_pct': risk_reduction_pct,
        'budget': budget,
        'total_cost': total_cost,
        'budget_used_pct': (total_cost / budget) * 100,
        'variable_cost': total_var_cost,
        'fixed_cost': total_fixed_cost,
        'mip_gap': solution['mip_gap'],
        'solve_time': solution['solve_time'],
        'status': solution['status'],
        'students_served': students_served,
        'students_served_pct': (students_served / n_students) * 100,
        'interventions_activated': interventions_activated
    }
    
    return metrics


def save_metrics(metrics, output_file):
    """Save metrics to file."""
    with open(output_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("ILP SOLUTION METRICS\n")
        f.write("=" * 70 + "\n\n")
        
        f.write("OPTIMIZATION RESULTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Status: {metrics['status']}\n")
        f.write(f"Solve time: {metrics['solve_time']:.2f} seconds\n")
        f.write(f"MIP gap: {metrics['mip_gap']*100:.4f}%\n\n")
        
        f.write("RISK REDUCTION\n")
        f.write("-" * 70 + "\n")
        f.write(f"Initial total risk: {metrics['initial_risk']:.6f}\n")
        f.write(f"Final total risk: {metrics['final_risk']:.6f}\n")
        f.write(f"Risk reduction: {metrics['risk_reduction']:.6f}\n")
        f.write(f"Risk reduction: {metrics['risk_reduction_pct']:.2f}%\n\n")
        
        f.write("BUDGET AND COSTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Available budget: ${metrics['budget']:,.2f}\n")
        f.write(f"Total cost: ${metrics['total_cost']:,.2f}\n")
        f.write(f"Budget used: {metrics['budget_used_pct']:.2f}%\n")
        f.write(f"  Variable costs: ${metrics['variable_cost']:,.2f}\n")
        f.write(f"  Fixed costs: ${metrics['fixed_cost']:,.2f}\n\n")
        
        f.write("ALLOCATION SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Students served: {metrics['students_served']:,} ({metrics['students_served_pct']:.2f}%)\n")
        f.write(f"Interventions activated: {metrics['interventions_activated']}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    print(f"\nMetrics saved to: {output_file}")


def print_summary(metrics):
    """Print summary to console."""
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\nStatus: {metrics['status']}")
    print(f"Solve time: {metrics['solve_time']:.2f}s")
    print(f"MIP gap: {metrics['mip_gap']*100:.4f}%")
    print(f"\nRisk reduction: {metrics['risk_reduction_pct']:.2f}%")
    print(f"Budget used: ${metrics['total_cost']:,.2f} / ${metrics['budget']:,.2f} ({metrics['budget_used_pct']:.2f}%)")
    print(f"Students served: {metrics['students_served']:,} ({metrics['students_served_pct']:.2f}%)")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Solve School Dropout Problem using Integer Linear Programming (Gurobi)",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--instance",
        type=str,
        required=True
    )
    
    parser.add_argument(
        "--time-limit",
        type=int,
        default=600
    )
    
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.001
    )
    
    parser.add_argument(
        "--output-metrics",
        type=str,
        default=None
    )
    
    parser.add_argument(
        "--output-solution",
        type=str,
        default=None
    )
    
    args = parser.parse_args()
    
    # Determine output filenames
    instance_basename = Path(args.instance).stem
    if args.output_metrics is None:
        args.output_metrics = f"metrics_{instance_basename}.txt"
    if args.output_solution is None:
        args.output_solution = f"solution_{instance_basename}.txt"
    
    print("="*70)
    print("ILP SOLVER FOR SCHOOL DROPOUT PROBLEM")
    print("="*70)
    print(f"\nInstance: {args.instance}")
    print(f"Time limit: {args.time_limit}s")
    print(f"MIP gap: {args.mip_gap*100:.2f}%")
    
    # Parse instance
    instance_data = parse_instance(args.instance)
    (n_students, n_interventions, risks, fixed_costs, var_costs, 
     capacities, subsets, effectiveness, budget) = instance_data
    
    # Solve
    result = build_and_solve_ilp(
        n_students, n_interventions, risks, fixed_costs, var_costs,
        capacities, subsets, effectiveness, budget,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap
    )
    
    if result is None:
        print("\nERROR: Failed to solve instance")
        return 1
    
    solution, model, _ = result
    
    # Save solution
    if solution['objective_value'] is not None:
        
        # Calculate metrics
        metrics = calculate_metrics(solution, n_students, risks, fixed_costs,
                                    var_costs, capacities, subsets, 
                                    effectiveness, budget)
        
        # Print summary
        print_summary(metrics)
        
        # Save outputs
        save_metrics(metrics, args.output_metrics)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
