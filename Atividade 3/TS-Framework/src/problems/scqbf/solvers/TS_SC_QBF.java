package problems.scqbf.solvers;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;

import metaheuristics.tabusearch.AbstractTS;
import problems.scqbf.SC_QBF_Inverse;
import solutions.Solution;

public class TS_SC_QBF extends AbstractTS<Integer> {
    
    // Search strategies
    public enum SearchStrategy {
        STANDARD,
        DIVERSIFICATION_RESTART,
        INTENSIFICATION_RESTART
    }
    
    // Search methods
    public enum SearchMethod {
        FIRST_IMPROVING,
        BEST_IMPROVING
    }
    
    // Configuration parameters
    private SearchStrategy strategy;
    private SearchMethod searchMethod;
    private int maxIterationsWithoutImprovement = 100;
    private int maxIterations = 1000;
    private long maxTimeMillis = 30 * 60 * 1000; // 30 minutes
    private long startTime;
    
    // Strategy-specific parameters
    private int diversificationInterval = 50;
    private int intensificationInterval = 30;
    private Solution<Integer> bestLocalSolution;
    private int iterationsWithoutImprovement = 0;
    
    // Tracking for experiments
    private String stoppingCriteria;
    private int totalIterations;
    
    public TS_SC_QBF(SC_QBF_Inverse objFunction, Integer tenure, Integer iterations, 
                     SearchStrategy strategy, SearchMethod searchMethod) {
        super(objFunction, tenure, iterations);
        this.strategy = strategy;
        this.searchMethod = searchMethod;
        this.bestLocalSolution = createEmptySol();
    }
    
    @Override
    public ArrayList<Integer> makeCL() {
        ArrayList<Integer> CL = new ArrayList<>();
        int n = ObjFunction.getDomainSize();
        for (int i = 0; i < n; i++) {
            CL.add(i);
        }
        return CL;
    }
    
    @Override
    public ArrayList<Integer> makeRCL() {
        return new ArrayList<>();
    }
    
    @Override
    public ArrayDeque<Integer> makeTL() {
        return new ArrayDeque<>(tenure);
    }
    
    @Override
    public void updateCL() {
        // Not needed for this implementation
    }
    
    @Override
    public Solution<Integer> createEmptySol() {
        return new Solution<>();
    }
    
    // Create a valid initial solution that covers all elements
    private Solution<Integer> createValidInitialSolution() {
        Solution<Integer> initialSol = new Solution<>();
        SC_QBF_Inverse scqbf = (SC_QBF_Inverse) ObjFunction;
        
        // Greedy approach: add subsets until all elements are covered
        Set<Integer> uncovered = new HashSet<>();
        for (int i = 1; i <= ObjFunction.getDomainSize(); i++) {
            uncovered.add(i);
        }
        
        ArrayList<Integer> candidates = makeCL();
        Collections.shuffle(candidates, rng);
        
        for (Integer subset : candidates) {
            if (uncovered.isEmpty()) break;
            
            // Check if this subset covers any uncovered element
            Set<Integer> covered = scqbf.getSubsets().get(subset);
            boolean useful = false;
            for (Integer elem : covered) {
                if (uncovered.contains(elem)) {
                    useful = true;
                    break;
                }
            }
            
            if (useful) {
                initialSol.add(subset);
                uncovered.removeAll(covered);
            }
        }
        
        // Ensure solution is valid
        if (!uncovered.isEmpty()) {
            // Add all subsets if necessary (shouldn't happen with valid instances)
            for (Integer subset : candidates) {
                if (!initialSol.contains(subset)) {
                    initialSol.add(subset);
                }
            }
        }
        
        ObjFunction.evaluate(initialSol);
        return initialSol;
    }
    
    @Override
    public Solution<Integer> neighborhoodMove() {
        ArrayList<Integer> CL = makeCL();
        Solution<Integer> bestNeighbor = null;
        double bestCost = sol.cost; // Start with current cost
        
        // Variables for move tracking
        Integer bestElemIn = null;
        Integer bestElemOut = null;
        
        SC_QBF_Inverse scqbf = (SC_QBF_Inverse) ObjFunction;
        
        if (searchMethod == SearchMethod.FIRST_IMPROVING) {
            // First improving
            for (Integer elem : CL) {
                if (sol.contains(elem)) {
                    // Try removal only if it maintains coverage
                    if (!TL.contains(elem)) {
                        Solution<Integer> neighbor = new Solution<>(sol);
                        neighbor.remove(elem);
                        
                        if (scqbf.isCoverValid(neighbor)) {
                            double cost = ObjFunction.evaluate(neighbor);
                            if (cost < bestCost) {
                                bestNeighbor = neighbor;
                                bestCost = cost;
                                bestElemOut = elem;
                                bestElemIn = null;
                                if (cost < sol.cost) {
                                    break; // First improvement found
                                }
                            }
                        }
                    }
                } else {
                    // Try insertion
                    if (!TL.contains(elem)) {
                        Solution<Integer> neighbor = new Solution<>(sol);
                        neighbor.add(elem);
                        double cost = ObjFunction.evaluate(neighbor);
                        if (cost < bestCost) {
                            bestNeighbor = neighbor;
                            bestCost = cost;
                            bestElemIn = elem;
                            bestElemOut = null;
                            if (cost < sol.cost) {
                                break; // First improvement found
                            }
                        }
                    }
                }
                
                // Try swap moves
                if (sol.contains(elem)) {
                    for (Integer other : CL) {
                        if (!sol.contains(other) && !TL.contains(elem) && !TL.contains(other)) {
                            Solution<Integer> neighbor = new Solution<>(sol);
                            neighbor.remove(elem);
                            neighbor.add(other);
                            
                            if (scqbf.isCoverValid(neighbor)) {
                                double cost = ObjFunction.evaluate(neighbor);
                                if (cost < bestCost) {
                                    bestNeighbor = neighbor;
                                    bestCost = cost;
                                    bestElemIn = other;
                                    bestElemOut = elem;
                                    if (cost < sol.cost) {
                                        break; // First improvement found
                                    }
                                }
                            }
                        }
                    }
                    if (bestCost < sol.cost) {
                        break;
                    }
                }
            }
        } else {
            // Best improving
            for (Integer elem : CL) {
                if (sol.contains(elem)) {
                    // Try removal only if it maintains coverage
                    if (!TL.contains(elem)) {
                        Solution<Integer> neighbor = new Solution<>(sol);
                        neighbor.remove(elem);
                        
                        if (scqbf.isCoverValid(neighbor)) {
                            double cost = ObjFunction.evaluate(neighbor);
                            if (cost < bestCost) {
                                bestNeighbor = neighbor;
                                bestCost = cost;
                                bestElemOut = elem;
                                bestElemIn = null;
                            }
                        }
                    }
                } else {
                    // Try insertion
                    if (!TL.contains(elem)) {
                        Solution<Integer> neighbor = new Solution<>(sol);
                        neighbor.add(elem);
                        double cost = ObjFunction.evaluate(neighbor);
                        if (cost < bestCost) {
                            bestNeighbor = neighbor;
                            bestCost = cost;
                            bestElemIn = elem;
                            bestElemOut = null;
                        }
                    }
                }
                
                // Try swap moves
                if (sol.contains(elem)) {
                    for (Integer other : CL) {
                        if (!sol.contains(other) && !TL.contains(elem) && !TL.contains(other)) {
                            Solution<Integer> neighbor = new Solution<>(sol);
                            neighbor.remove(elem);
                            neighbor.add(other);
                            
                            if (scqbf.isCoverValid(neighbor)) {
                                double cost = ObjFunction.evaluate(neighbor);
                                if (cost < bestCost) {
                                    bestNeighbor = neighbor;
                                    bestCost = cost;
                                    bestElemIn = other;
                                    bestElemOut = elem;
                                }
                            }
                        }
                    }
                }
            }
        }
        
        // Update tabu list and solution
        if (bestNeighbor != null) {
            if (bestElemIn != null) {
                TL.add(bestElemIn);
                if (TL.size() > tenure) {
                    TL.poll();
                }
            }
            if (bestElemOut != null) {
                TL.add(bestElemOut);
                if (TL.size() > tenure) {
                    TL.poll();
                }
            }
            sol = bestNeighbor;
        }
        
        return sol;
    }
    
    private void diversificationRestart() {
        SC_QBF_Inverse scqbf = (SC_QBF_Inverse) ObjFunction;
        
        // Create frequency memory
        int[] frequency = new int[ObjFunction.getDomainSize()];
        for (Integer elem : bestSol) {
            frequency[elem]++;
        }
        
        // Find rarely used elements
        ArrayList<Integer> rarelyUsed = new ArrayList<>();
        for (int i = 0; i < frequency.length; i++) {
            if (frequency[i] == 0) {
                rarelyUsed.add(i);
            }
        }
        
        if (!rarelyUsed.isEmpty()) {
            // Start from best solution
            sol = new Solution<>(bestSol);
            Collections.shuffle(rarelyUsed, rng);
            
            // Try to add rarely used elements
            int added = 0;
            for (Integer elem : rarelyUsed) {
                if (added >= 3) break;
                if (!sol.contains(elem)) {
                    sol.add(elem);
                    added++;
                }
            }
            
            // Remove redundant elements while maintaining coverage
            ArrayList<Integer> toCheck = new ArrayList<>(sol);
            Collections.shuffle(toCheck, rng);
            
            for (Integer elem : toCheck) {
                Solution<Integer> test = new Solution<>(sol);
                test.remove(elem);
                if (scqbf.isCoverValid(test)) {
                    sol = test;
                }
            }
            
            ObjFunction.evaluate(sol);
        }
    }
    
    private void intensificationRestart() {
        // Restart from best solution
        sol = new Solution<>(bestSol);
        
        // Clear tabu list
        TL.clear();
        
        // Intensive local search
        for (int i = 0; i < 20; i++) {
            Solution<Integer> oldSol = new Solution<>(sol);
            neighborhoodMove();
            if (sol.cost >= oldSol.cost) {
                break;
            }
        }
    }
    
    @Override
    public Solution<Integer> solve() {
        startTime = System.currentTimeMillis();
        
        // Create valid initial solution
        sol = createValidInitialSolution();
        bestSol = new Solution<>(sol);
        bestLocalSolution = new Solution<>(sol);
        TL = makeTL();
        
        totalIterations = 0;
        iterationsWithoutImprovement = 0;
        
        for (int i = 0; i < iterations; i++) {
            totalIterations++;
            
            // Check stopping criteria
            if (System.currentTimeMillis() - startTime > maxTimeMillis) {
                stoppingCriteria = "TIME_LIMIT";
                break;
            }
            
            if (totalIterations >= maxIterations) {
                stoppingCriteria = "MAX_ITERATIONS";
                break;
            }
            
            if (iterationsWithoutImprovement >= maxIterationsWithoutImprovement) {
                stoppingCriteria = "NO_IMPROVEMENT";
                break;
            }
            
            neighborhoodMove();
            
            // Update best solution
            if (sol.cost < bestSol.cost) {
                bestSol = new Solution<>(sol);
                iterationsWithoutImprovement = 0;
                if (verbose) {
                    System.out.println("(Iter. " + i + ") BestSol = " + bestSol.cost);
                }
            } else {
                iterationsWithoutImprovement++;
            }
            
            // Apply strategy-specific operations
            if (strategy == SearchStrategy.DIVERSIFICATION_RESTART) {
                if (i > 0 && i % diversificationInterval == 0) {
                    diversificationRestart();
                }
            } else if (strategy == SearchStrategy.INTENSIFICATION_RESTART) {
                if (i > 0 && i % intensificationInterval == 0) {
                    intensificationRestart();
                }
            }
        }
        
        if (stoppingCriteria == null) {
            stoppingCriteria = "ITERATIONS_COMPLETE";
        }
        
        return bestSol;
    }
    
    // Experimental framework
    public static void runExperiments(List<String> instanceFiles) throws IOException {
        FileWriter csvWriter = new FileWriter("results.csv");
        csvWriter.append("Configuration,Instance,BestValue,ExecutionTime(s),Iterations,StoppingCriteria\n");
        
        // Configuration combinations
        int tenure1 = 10;
        int tenure2 = 20;
        
        for (String instanceFile : instanceFiles) {
            System.out.println("\nProcessing instance: " + instanceFile);
            
            // 1. STANDARD
            runConfiguration(csvWriter, instanceFile, "STANDARD", 
                           SearchStrategy.STANDARD, SearchMethod.FIRST_IMPROVING, tenure1);
            
            // 2. STANDARD+BEST
            runConfiguration(csvWriter, instanceFile, "STANDARD+BEST", 
                           SearchStrategy.STANDARD, SearchMethod.BEST_IMPROVING, tenure1);
            
            // 3. STANDARD+TENURE
            runConfiguration(csvWriter, instanceFile, "STANDARD+TENURE", 
                           SearchStrategy.STANDARD, SearchMethod.FIRST_IMPROVING, tenure2);
            
            // 4. STANDARD+METHOD1 (Diversification)
            runConfiguration(csvWriter, instanceFile, "STANDARD+DIVERSIFICATION", 
                           SearchStrategy.DIVERSIFICATION_RESTART, SearchMethod.FIRST_IMPROVING, tenure1);
            
            // 5. STANDARD+METHOD2 (Intensification)
            runConfiguration(csvWriter, instanceFile, "STANDARD+INTENSIFICATION", 
                           SearchStrategy.INTENSIFICATION_RESTART, SearchMethod.FIRST_IMPROVING, tenure1);
            
            // 6. STANDARD+METHOD1+BEST (Diversification)
            runConfiguration(csvWriter, instanceFile, "DIVERSIFICATION+BEST", 
                           SearchStrategy.DIVERSIFICATION_RESTART, SearchMethod.BEST_IMPROVING, tenure1);

            // 7. STANDARD+METHOD2+BEST (Intensification)
            runConfiguration(csvWriter, instanceFile, "INTENSIFICATION+BEST", 
                           SearchStrategy.INTENSIFICATION_RESTART, SearchMethod.BEST_IMPROVING, tenure1);
            
            // 8. STANDARD+METHOD1+TENURE (Diversification)
            runConfiguration(csvWriter, instanceFile, "DIVERSIFICATION+TENURE", 
                           SearchStrategy.DIVERSIFICATION_RESTART, SearchMethod.FIRST_IMPROVING, tenure2);

            // 9. STANDARD+METHOD2+TENURE (Intensification)
            runConfiguration(csvWriter, instanceFile, "INTENSIFICATION+TENURE", 
                           SearchStrategy.INTENSIFICATION_RESTART, SearchMethod.FIRST_IMPROVING, tenure2);
        }
        
        csvWriter.flush();
        csvWriter.close();
        System.out.println("\nExperiments completed. Results saved to results.csv");
    }
    
    private static void runConfiguration(FileWriter csvWriter, String instanceFile, String configName,
                                        SearchStrategy strategy, SearchMethod method, int tenure) 
                                        throws IOException {
        System.out.println("  Running configuration: " + configName);
        
        SC_QBF_Inverse problem = new SC_QBF_Inverse(instanceFile);
        TS_SC_QBF ts = new TS_SC_QBF(problem, tenure, 1000, strategy, method);
        ts.verbose = false;
        
        long startTime = System.currentTimeMillis();
        Solution<Integer> solution = ts.solve();
        long executionTime = System.currentTimeMillis() - startTime;
        
        // Write results
        csvWriter.append(String.format("%s,%s,%.2f,%.2f,%d,%s\n",
            configName,
            instanceFile.substring(instanceFile.lastIndexOf('/') + 1),
            -solution.cost, // Convert back to maximization value
            executionTime / 1000.0,
            ts.totalIterations,
            ts.stoppingCriteria
        ));
        csvWriter.flush();
    }
    
    public static void main(String[] args) throws IOException {
        List<String> instances = Arrays.asList(
            "instances/scqbf/n25p1.txt",
            "instances/scqbf/n25p2.txt",
            "instances/scqbf/n25p3.txt",
            "instances/scqbf/n50p1.txt",
            "instances/scqbf/n50p2.txt",
            "instances/scqbf/n50p3.txt",
            "instances/scqbf/n100p1.txt",
            "instances/scqbf/n100p2.txt",
            "instances/scqbf/n100p3.txt",
            "instances/scqbf/n200p1.txt",
            "instances/scqbf/n200p2.txt",
            "instances/scqbf/n200p3.txt",
            "instances/scqbf/n400p1.txt",
            "instances/scqbf/n400p2.txt",
            "instances/scqbf/n400p3.txt"
        );
        
        runExperiments(instances);
        
        // FileWriter csvWriter = new FileWriter("results.csv");
        // int tenure1 = 10;

        // runConfiguration(csvWriter, "instances/scqbf/n200p1.txt", "STANDARD", 
        //                   SearchStrategy.STANDARD, SearchMethod.FIRST_IMPROVING, tenure1);
    }
}