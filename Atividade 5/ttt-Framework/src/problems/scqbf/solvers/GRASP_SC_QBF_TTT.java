package problems.scqbf.solvers;

import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import metaheuristics.grasp.AbstractGRASP;
import problems.scqbf.SC_QBF_Inverse;
import solutions.Solution;

/**
 * Advanced GRASP implementation for the Set Cover QBF problem
 * with Reactive GRASP, Random Plus Greedy, and First/Best Improving options
 */
public class GRASP_SC_QBF_TTT extends AbstractGRASP<Integer> {
    
    private SC_QBF_Inverse scqbf;
    
    // Configuration parameters
    private boolean useFirstImproving;
    private boolean useReactiveGRASP;
    private boolean useRandomPlusGreedy;
    
    // Random Plus Greedy parameters
    private int randomSteps; // for Random Plus Greedy
    
    // Reactive GRASP parameters
    private double[] alphaValues = {0.01, 0.05, 0.10, 0.20, 0.50, 0.90};
    private Map<Double, Double> alphaAverages;
    private Map<Double, Integer> alphaCounts;
    private Map<Double, Double> alphaProbabilities;
    private Double currentAlpha;
    
    // --- Early-stop by target ---
    private Double targetValue;          
    private long   timeToTargetMs = -1;  
    private long maxTimeMillis = 30 * 60 * 1000; // 30 minutes
    
    /**
     * Constructor for GRASP_SC_QBF_TTT
     * @param alpha The GRASP greediness-randomness parameter (ignored if reactive)
     * @param iterations Number of iterations
     * @param filename Instance file name
     * @param useFirstImproving Use first-improving instead of best-improving
     * @param useReactiveGRASP Use Reactive GRASP for alpha selection
     * @param useRandomPlusGreedy Use Random Plus Greedy construction
     * @param randomSteps Number of random steps for Random Plus Greedy
     */
    public GRASP_SC_QBF_TTT(Double alpha, Integer iterations, String filename,
                                 boolean useFirstImproving, boolean useReactiveGRASP,
                                 boolean useRandomPlusGreedy, int randomSteps, int maxTimeMillis,
                                 Double targetValue) throws IOException {
        super(new SC_QBF_Inverse(filename), alpha, iterations);
        this.scqbf = (SC_QBF_Inverse) this.ObjFunction;
        this.useFirstImproving = useFirstImproving;
        this.useReactiveGRASP = useReactiveGRASP;
        this.useRandomPlusGreedy = useRandomPlusGreedy;
        this.randomSteps = randomSteps;
        this.maxTimeMillis = maxTimeMillis;
        this.targetValue = targetValue;
        
        // Initialize Reactive GRASP structures
        if (useReactiveGRASP) {
            initializeReactiveGRASP();
        }
    }
    
    /**
     * Initialize Reactive GRASP data structures
     */
    private void initializeReactiveGRASP() {
        alphaAverages = new HashMap<>();
        alphaCounts = new HashMap<>();
        alphaProbabilities = new HashMap<>();
        
        // Initialize with equal probabilities
        double initialProb = 1.0 / alphaValues.length;
        for (double alpha : alphaValues) {
            alphaAverages.put(alpha, 0.0);
            alphaCounts.put(alpha, 0);
            alphaProbabilities.put(alpha, initialProb);
        }
    }
    
    /**
     * Select alpha value for Reactive GRASP
     */
    private double selectAlpha() {
        if (!useReactiveGRASP) {
            return alpha;
        }
        
        // Select alpha based on probabilities
        double rand = rng.nextDouble();
        double cumProb = 0.0;
        
        for (double alphaVal : alphaValues) {
            cumProb += alphaProbabilities.get(alphaVal);
            if (rand <= cumProb) {
                currentAlpha = alphaVal;
                return alphaVal;
            }
        }
        
        // Fallback 
        currentAlpha = alphaValues[alphaValues.length - 1];
        return currentAlpha;
    }
    
    /**
     * Update Reactive GRASP probabilities
     */
    private void updateReactiveGRASP(double solutionCost) {
        if (!useReactiveGRASP || currentAlpha == null) {
            return;
        }
        
        // Update average for current alpha
        int count = alphaCounts.get(currentAlpha);
        double avg = alphaAverages.get(currentAlpha);
        avg = (avg * count + solutionCost) / (count + 1);
        alphaAverages.put(currentAlpha, avg);
        alphaCounts.put(currentAlpha, count + 1);
        
        // Update probabilities every 50 iterations
        int totalCount = 0;
        for (int c : alphaCounts.values()) {
            totalCount += c;
        }
        
        if (totalCount % 50 == 0 && totalCount > 0) {
            double bestCost = bestSol != null ? -bestSol.cost : Double.NEGATIVE_INFINITY;
            double sumQ = 0.0;
            
            // Calculate q values
            Map<Double, Double> qValues = new HashMap<>();
            for (double alphaVal : alphaValues) {
                if (alphaCounts.get(alphaVal) > 0) {
                    double q = bestCost / alphaAverages.get(alphaVal);
                    qValues.put(alphaVal, Math.max(q, 0.001)); // Avoid zero probabilities
                    sumQ += qValues.get(alphaVal);
                } else {
                    qValues.put(alphaVal, 0.001);
                    sumQ += 0.001;
                }
            }
            
            // Update probabilities
            for (double alphaVal : alphaValues) {
                alphaProbabilities.put(alphaVal, qValues.get(alphaVal) / sumQ);
            }
        }
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
     * Modified constructive heuristic with Random Plus Greedy option
     */
    @Override
    public Solution<Integer> constructiveHeuristic() {
        if (useRandomPlusGreedy) {
            return randomPlusGreedyConstruction();
        } else {
            return standardConstruction();
        }
    }
    
    /**
     * Random Plus Greedy construction
     */
    private Solution<Integer> randomPlusGreedyConstruction() {
        CL = makeCL();
        sol = createEmptySol();
        
        // Phase 1: Random steps
        Set<Integer> uncovered = new HashSet<>();
        for (int i = 1; i <= scqbf.getDomainSize(); i++) {
            uncovered.add(i);
        }
        
        // Add random elements for the first randomSteps iterations
        int stepsCount = 0;
        while (stepsCount < randomSteps && !CL.isEmpty() && !uncovered.isEmpty()) {
            int rndIndex = rng.nextInt(CL.size());
            Integer selected = CL.get(rndIndex);
            sol.add(selected);
            CL.remove(selected);
            uncovered.removeAll(scqbf.getSubsets().get(selected));
            stepsCount++;
        }
        
        // Phase 2: Greedy completion to ensure valid cover
        while (!uncovered.isEmpty() && !CL.isEmpty()) {
            Integer bestSubset = null;
            int maxNewCovered = 0;
            
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
        
        // Phase 3: Greedy improvement with original objective
        updateCL();
        while (!CL.isEmpty()) {
            Integer bestCandidate = null;
            double bestCost = Double.POSITIVE_INFINITY;
            
            for (Integer c : CL) {
                double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
                if (deltaCost < bestCost) {
                    bestCost = deltaCost;
                    bestCandidate = c;
                }
            }
            
            if (bestCandidate != null && bestCost < 0) {
                sol.add(bestCandidate);
                CL.remove(bestCandidate);
            } else {
                break;
            }
        }
        
        ObjFunction.evaluate(sol);
        return sol;
    }
    
    /**
     * Standard GRASP construction with Reactive GRASP support
     */
    private Solution<Integer> standardConstruction() {
        // Select alpha for this iteration
        double iterAlpha = selectAlpha();
        
        CL = makeCL();
        RCL = makeRCL();
        sol = createEmptySol();
        cost = Double.POSITIVE_INFINITY;
        
        // First phase: ensure valid cover
        Set<Integer> uncovered = new HashSet<>();
        for (int i = 1; i <= scqbf.getDomainSize(); i++) {
            uncovered.add(i);
        }
        
        while (!uncovered.isEmpty() && !CL.isEmpty()) {
            Integer bestSubset = null;
            int maxNewCovered = 0;
            
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
        
        // Second phase: GRASP construction with selected alpha
        while (!constructiveStopCriteria() && !CL.isEmpty()) {
            double maxCost = Double.NEGATIVE_INFINITY;
            double minCost = Double.POSITIVE_INFINITY;
            
            cost = ObjFunction.evaluate(sol);
            updateCL();
            
            for (Integer c : CL) {
                Double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
                if (deltaCost < minCost) minCost = deltaCost;
                if (deltaCost > maxCost) maxCost = deltaCost;
            }
            
            for (Integer c : CL) {
                Double deltaCost = ObjFunction.evaluateInsertionCost(c, sol);
                if (deltaCost <= minCost + iterAlpha * (maxCost - minCost)) {
                    RCL.add(c);
                }
            }
            
            if (!RCL.isEmpty()) {
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
     * Local search with first-improving or best-improving option
     */
    @Override
    public Solution<Integer> localSearch() {
        if (useFirstImproving) {
            return localSearchFirstImproving();
        } else {
            return localSearchBestImproving();
        }
    }
    
    /**
     * Local search with first-improving strategy
     */
    private Solution<Integer> localSearchFirstImproving() {
        boolean improved;
        
        do {
            improved = false;
            updateCL();
            
            // Try insertions
            for (Integer candIn : CL) {
                double deltaCost = ObjFunction.evaluateInsertionCost(candIn, sol);
                if (deltaCost < -Double.MIN_VALUE) {
                    sol.add(candIn);
                    CL.remove(candIn);
                    ObjFunction.evaluate(sol);
                    improved = true;
                    break;
                }
            }
            
            if (!improved) {
                // Try removals
                for (Integer candOut : sol) {
                    double deltaCost = ObjFunction.evaluateRemovalCost(candOut, sol);
                    if (deltaCost < -Double.MIN_VALUE && deltaCost < Double.POSITIVE_INFINITY) {
                        sol.remove(candOut);
                        CL.add(candOut);
                        ObjFunction.evaluate(sol);
                        improved = true;
                        break;
                    }
                }
            }
            
            if (!improved) {
                // Try exchanges
                outerLoop:
                for (Integer candIn : CL) {
                    for (Integer candOut : sol) {
                        double deltaCost = ObjFunction.evaluateExchangeCost(candIn, candOut, sol);
                        if (deltaCost < -Double.MIN_VALUE && deltaCost < Double.POSITIVE_INFINITY) {
                            sol.remove(candOut);
                            sol.add(candIn);
                            CL.remove(candIn);
                            CL.add(candOut);
                            ObjFunction.evaluate(sol);
                            improved = true;
                            break outerLoop;
                        }
                    }
                }
            }
            
        } while (improved);
        
        return sol;
    }
    
    /**
     * Local search with best-improving strategy
     */
    private Solution<Integer> localSearchBestImproving() {
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
     * Modified solve method to handle Reactive GRASP
     */
    @Override
    public Solution<Integer> solve() {
    	long solveStart = System.currentTimeMillis();
        bestSol = createEmptySol();
        
        for (int i = 0; i < iterations; i++) {
            constructiveHeuristic();
            localSearch();
            
            if (System.currentTimeMillis() - solveStart > maxTimeMillis) {                
                break;
            }
            
            // Update Reactive GRASP statistics
            if (useReactiveGRASP) {
                updateReactiveGRASP(-sol.cost);
            }
            
            if (bestSol.cost > sol.cost) {
                bestSol = new Solution<Integer>(sol);
                if (verbose) {
                    System.out.println("(Iter. " + i + ") BestSol = " + bestSol);
                    if (useReactiveGRASP && currentAlpha != null) {
                        System.out.println("  -> Alpha used: " + currentAlpha);
                    }
                }
            }
            
            // --- EARLY STOP by target ---
            if (targetValue != null) {
                double valorAtual = -bestSol.cost;
                if (valorAtual >= targetValue) {
                    if (timeToTargetMs < 0) {
                        timeToTargetMs = System.currentTimeMillis() - solveStart; 
                    }
                    break; 
                }
            }
        }
        
        return bestSol;
    }
    
    public long getTimeToTargetMs() {
        return timeToTargetMs;
    }
    
    public void setSeed(long seed) {
        this.rng = new java.util.Random(seed);
    }
    
    private static void writeCsvHeaderIfNeeded(String csvPath) {
        java.io.File f = new java.io.File(csvPath);
        if (!f.exists()) {
            try (java.io.FileWriter fw = new java.io.FileWriter(f, true)) {
                fw.write("instancia,target,seed,tempo_para_target_ms\n");
            } catch (java.io.IOException e) {
                throw new RuntimeException(e);
            }
        }
    }

    private static void appendCsvLine(String csvPath, String instancia, Double target, long seed, long tempoMs) {
        try (java.io.FileWriter fw = new java.io.FileWriter(csvPath, true)) {
            fw.write(String.format("%s,%s,%d,%d%n",
                    instancia,
                    (target == null ? "" : target.toString()),
                    seed,
                    tempoMs
            ));
        } catch (java.io.IOException e) {
            throw new RuntimeException(e);
        }
    }

    
    /**
     * Main method for testing
     */
    public static void main(String[] args) throws IOException {

        String instance   = "instances/scqbf/n400p2.txt";
        int    iterations = 100000;
        double alpha      = 0.05;
        boolean firstImproving   = true;
        boolean reactive         = false;
        boolean randomPlusGreedy = true;
        int     randomSteps      = 5;
        int     maxTimeMillis      = 10*60*1000;

        // Target
        Double target = 10486.78; 

        int runs    = 100;      
        long baseSeed = 2025L; 

        // Output
        String csvPath = "resultados_grasp_scqbf.csv";
        writeCsvHeaderIfNeeded(csvPath);

        for (int run = 0; run < runs; run++) {
            long seed = baseSeed + run;

            GRASP_SC_QBF_TTT grasp = new GRASP_SC_QBF_TTT(
                alpha, iterations, instance,
                firstImproving, reactive, randomPlusGreedy, randomSteps, maxTimeMillis,
                target 
            );
            grasp.setSeed(seed);  

            long start = System.currentTimeMillis();
            Solution<Integer> sol = grasp.solve();
            long total = System.currentTimeMillis() - start;

            long ttt = grasp.getTimeToTargetMs(); 

            
            appendCsvLine(csvPath, instance, target, seed, ttt);

            System.out.printf("Run %d | seed=%d | best=%.4f | t_total=%.3fs | t_target=%s%n",
                    run, seed, -sol.cost, total/1000.0, (ttt >= 0 ? (ttt/1000.0 + "s") : "NA"));
        }
    }

}