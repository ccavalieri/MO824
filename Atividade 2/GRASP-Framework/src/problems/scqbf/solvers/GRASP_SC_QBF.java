package problems.scqbf.solvers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.Set;

import metaheuristics.grasp.AbstractGRASP;
import problems.scqbf.SC_QBF_Inverse;
import solutions.Solution;

/**
 * GRASP implementation for the Set Cover QBF problem
 */
public class GRASP_SC_QBF extends AbstractGRASP<Integer> {
    
    private SC_QBF_Inverse scqbf;
    
    /**
     * Constructor for GRASP_SC_QBF
     * @param alpha The GRASP greediness-randomness parameter
     * @param iterations Number of iterations
     * @param filename Instance file name
     */
    public GRASP_SC_QBF(Double alpha, Integer iterations, String filename) throws IOException {
        super(new SC_QBF_Inverse(filename), alpha, iterations);
        this.scqbf = (SC_QBF_Inverse) this.ObjFunction;
    }
    
    /**
     * Creates the Candidate List with all subsets
     */
    @Override
    public ArrayList<Integer> makeCL() {
        ArrayList<Integer> _CL = new ArrayList<Integer>();
        for (int i = 0; i < ObjFunction.getDomainSize(); i++) {
            _CL.add(i);
        }
        return _CL;
    }
    
    /**
     * Creates the Restricted Candidate List
     */
    @Override
    public ArrayList<Integer> makeRCL() {
        return new ArrayList<Integer>();
    }
    
    /**
     * Updates CL removing elements already in solution
     */
    @Override
    public void updateCL() {
        ArrayList<Integer> toRemove = new ArrayList<>();
        for (Integer elem : sol) {
            if (CL.contains(elem)) {
                toRemove.add(elem);
            }
        }
        CL.removeAll(toRemove);
    }
    
    /**
     * Creates an empty solution
     */
    @Override
    public Solution<Integer> createEmptySol() {
        Solution<Integer> sol = new Solution<Integer>();
        sol.cost = 0.0;
        return sol;
    }
    
    /**
     * Modified constructive heuristic that ensures valid cover
     */
    @Override
    public Solution<Integer> constructiveHeuristic() {
        CL = makeCL();
        RCL = makeRCL();
        sol = createEmptySol();
        cost = Double.POSITIVE_INFINITY;
        
        // First phase: ensure valid cover
        Set<Integer> uncovered = new HashSet<>();
        for (int i = 1; i <= scqbf.getDomainSize(); i++) {
            uncovered.add(i);
        }
        
        // Greedily add subsets to cover all elements
        while (!uncovered.isEmpty() && !CL.isEmpty()) {
            Integer bestSubset = null;
            int maxNewCovered = 0;
            
            // Find subset that covers most uncovered elements
            for (Integer subset : CL) {
                Set<Integer> covered = new HashSet<>(scqbf.getSubsets().get(subset));
                covered.retainAll(uncovered);
                if (covered.size() > maxNewCovered) {
                    maxNewCovered = covered.size();
                    bestSubset = subset;
                }
            }
            
            if (bestSubset != null) {
                sol.add(bestSubset);
                CL.remove(bestSubset);
                uncovered.removeAll(scqbf.getSubsets().get(bestSubset));
            } else {
                break;
            }
        }
        
        // Second phase: GRASP construction for remaining elements
        while (!constructiveStopCriteria() && !CL.isEmpty()) {
            double maxCost = Double.NEGATIVE_INFINITY;
            double minCost = Double.POSITIVE_INFINITY;
            
            cost = ObjFunction.evaluate(sol);
            updateCL();
            
            // Evaluate all candidates
            for (Integer c : CL) {
                Double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
                if (deltaCost < minCost) minCost = deltaCost;
                if (deltaCost > maxCost) maxCost = deltaCost;
            }
            
            // Build RCL with alpha threshold
            for (Integer c : CL) {
                Double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
                if (deltaCost <= minCost + alpha * (maxCost - minCost)) {
                    RCL.add(c);
                }
            }
            
            if (!RCL.isEmpty()) {
                // Select random element from RCL
                int rndIndex = rng.nextInt(RCL.size());
                Integer inCand = RCL.get(rndIndex);
                CL.remove(inCand);
                sol.add(inCand);
                ObjFunction.evaluate(sol);
            }
            
            RCL.clear();
        }
        
        return sol;
    }
    
    /**
     * Local search with set cover constraints
     */
    @Override
    public Solution<Integer> localSearch() {
        Double minDeltaCost;
        Integer bestCandIn = null, bestCandOut = null;
        
        do {
            minDeltaCost = Double.POSITIVE_INFINITY;
            updateCL();
            
            // Evaluate insertions
            for (Integer candIn : CL) {
                double deltaCost = ObjFunction.evaluateInsertionCost(candIn, sol);
                if (deltaCost < minDeltaCost) {
                    minDeltaCost = deltaCost;
                    bestCandIn = candIn;
                    bestCandOut = null;
                }
            }
            
            // Evaluate removals
            for (Integer candOut : sol) {
                double deltaCost = ObjFunction.evaluateRemovalCost(candOut, sol);
                if (deltaCost < minDeltaCost && deltaCost < Double.POSITIVE_INFINITY) {
                    minDeltaCost = deltaCost;
                    bestCandIn = null;
                    bestCandOut = candOut;
                }
            }
            
            // Evaluate exchanges
            for (Integer candIn : CL) {
                for (Integer candOut : sol) {
                    double deltaCost = ObjFunction.evaluateExchangeCost(candIn, candOut, sol);
                    if (deltaCost < minDeltaCost && deltaCost < Double.POSITIVE_INFINITY) {
                        minDeltaCost = deltaCost;
                        bestCandIn = candIn;
                        bestCandOut = candOut;
                    }
                }
            }
            
            // Apply best move if it improves solution
            if (minDeltaCost < -Double.MIN_VALUE) {
                if (bestCandOut != null) {
                    sol.remove(bestCandOut);
                    CL.add(bestCandOut);
                }
                if (bestCandIn != null) {
                    sol.add(bestCandIn);
                    CL.remove(bestCandIn);
                }
                ObjFunction.evaluate(sol);
            }
            
        } while (minDeltaCost < -Double.MIN_VALUE);
        
        return sol;
    }
    
    /**
     * Main method for testing
     */
    public static void main(String[] args) throws IOException {
        long startTime = System.currentTimeMillis();
        
        GRASP_SC_QBF grasp = new GRASP_SC_QBF(0.05, 100, "instances/scqbf/n200p1.txt");
        Solution<Integer> bestSol = grasp.solve();
        
        System.out.println("Best solution found: " + bestSol);
        System.out.println("Solution covers all elements: " + 
            ((SC_QBF_Inverse)grasp.ObjFunction).isCoverValid(bestSol));
        
        long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        System.out.println("Time = " + (double)totalTime/1000.0 + " sec");
    }
}