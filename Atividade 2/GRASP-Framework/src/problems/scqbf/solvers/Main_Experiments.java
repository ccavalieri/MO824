package problems.scqbf.solvers;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;

import metaheuristics.grasp.AbstractGRASP;
import solutions.Solution;

/**
 * Main class for running experiments with different GRASP configurations
 */
public class Main_Experiments {
    
    // Experiment parameters
    private static final long MAX_TIME_MILLIS = 30 * 60 * 1000; // 30 minutes
    private static final int MAX_ITERATIONS_WITHOUT_IMPROVEMENT = 100;
    private static final int MAX_TOTAL_ITERATIONS = 1000;
    private static final double[] ALPHA_VALUES = {0.05, 0.90}; 
    private static final int RANDOM_STEPS = 5;
    
    // Instance files to test
    private static final String[] INSTANCE_FILES = {
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
    };
    
    /**
     * Wrapper class to add stopping criteria to GRASP_SC_QBF_Advanced
     */
    static class GRASP_SC_QBF_WithStoppingCriteria extends GRASP_SC_QBF_Advanced {
        
        private long startTime;
        private long maxTimeMillis;
        private int maxIterWithoutImprovement;
        private int iterWithoutImprovement = 0;
        private int currentIteration = 0;
        private double lastBestCost = Double.POSITIVE_INFINITY;
        
        // Statistics for results
        public int totalIterations = 0;
        public long executionTime = 0;
        public int iterationsToOptimal = 0;
        public boolean stoppedByTime = false;
        public boolean stoppedByNoImprovement = false;
        
        public GRASP_SC_QBF_WithStoppingCriteria(Double alpha, Integer iterations, String filename,
                                                 boolean useFirstImproving, boolean useReactiveGRASP,
                                                 boolean useRandomPlusGreedy, int randomSteps,
                                                 long maxTimeMillis, int maxIterWithoutImprovement) 
                                                 throws IOException {
            super(alpha, iterations, filename, useFirstImproving, useReactiveGRASP, 
                  useRandomPlusGreedy, randomSteps);
            this.maxTimeMillis = maxTimeMillis;
            this.maxIterWithoutImprovement = maxIterWithoutImprovement;
            AbstractGRASP.verbose = false;
        }
        
        @Override
        public Solution<Integer> solve() {
            bestSol = createEmptySol();
            startTime = System.currentTimeMillis();
            iterWithoutImprovement = 0;
            currentIteration = 0;
            
            while (currentIteration < iterations) {
                // Check time limit
                if (System.currentTimeMillis() - startTime > maxTimeMillis) {
                    stoppedByTime = true;
                    break;
                }
                
                // Check iterations without improvement
                if (iterWithoutImprovement >= maxIterWithoutImprovement) {
                    stoppedByNoImprovement = true;
                    break;
                }
                
                constructiveHeuristic();
                localSearch();
                
                // Update best solution and check improvement
                if (bestSol.cost > sol.cost) {
                    bestSol = new Solution<Integer>(sol);
                    iterationsToOptimal = currentIteration;
                    iterWithoutImprovement = 0;
                } else {
                    iterWithoutImprovement++;
                }
                
                currentIteration++;
            }
            
            totalIterations = currentIteration;
            executionTime = System.currentTimeMillis() - startTime;
            
            return bestSol;
        }
    }
    
    /**
     * Experiment configuration
     */
    static class ExperimentConfig {
        String name;
        double alpha;
        boolean useFirstImproving;
        boolean useReactiveGRASP;
        boolean useRandomPlusGreedy;
        int randomSteps;
        
        public ExperimentConfig(String name, double alpha, boolean useFirstImproving,
                               boolean useReactiveGRASP, boolean useRandomPlusGreedy,
                               int randomSteps) {
            this.name = name;
            this.alpha = alpha;
            this.useFirstImproving = useFirstImproving;
            this.useReactiveGRASP = useReactiveGRASP;
            this.useRandomPlusGreedy = useRandomPlusGreedy;
            this.randomSteps = randomSteps;
        }
    }
    
    /**
     * Experiment result
     */
    static class ExperimentResult {
        String configName;
        String instanceFile;
        double bestCost;
        int totalIterations;
        int iterationsToOptimal;
        long executionTimeMs;
        boolean stoppedByTime;
        boolean stoppedByNoImprovement;
        int solutionSize;
        
        public String toCSV() {
            return String.format("%s,%s,%.2f,%d,%d,%d,%s,%s,%d",
                configName,
                instanceFile.substring(instanceFile.lastIndexOf('/') + 1),
                -bestCost,
                totalIterations,
                iterationsToOptimal,
                executionTimeMs,
                stoppedByTime ? "TIME" : (stoppedByNoImprovement ? "NO_IMPROV" : "ITERATIONS"),
                stoppedByTime,
                solutionSize
            );
        }
    }
    
    /**
     * Generate all experiment configurations
     */
    private static List<ExperimentConfig> generateConfigurations() {
        List<ExperimentConfig> configs = new ArrayList<>();
        
        // === Individual configurations ===
        
        // 1. Standard GRASP with different alphas
        for (double alpha : ALPHA_VALUES) {
            configs.add(new ExperimentConfig(
                "GRASP_alpha=" + alpha,
                alpha, false, false, false, 0
            ));
        }
        
        // 2. First-improving with fixed alpha
        configs.add(new ExperimentConfig(
            "GRASP_FirstImproving",
            ALPHA_VALUES[0], true, false, false, 0
        ));
        
        // 3. Reactive GRASP
        configs.add(new ExperimentConfig(
            "GRASP_Reactive",
            0.0, false, true, false, 0
        ));
        
        // 4. Random Plus Greedy
        configs.add(new ExperimentConfig(
            "GRASP_RandomPlusGreedy",
            ALPHA_VALUES[0], false, false, true, RANDOM_STEPS
        ));
        
        // === Combinations ===
        
        // 5. First-improving + Different alphas (except reactive)
        for (double alpha : ALPHA_VALUES) {
            configs.add(new ExperimentConfig(
                "GRASP_FirstImp_alpha=" + alpha,
                alpha, true, false, false, 0
            ));
        }
        
        // 6. First-improving + Reactive GRASP
        configs.add(new ExperimentConfig(
            "GRASP_FirstImp_Reactive",
            0.0, true, true, false, 0
        ));
        
        // 7. First-improving + Random Plus Greedy
        configs.add(new ExperimentConfig(
            "GRASP_FirstImp_RandomPlusGreedy",
            ALPHA_VALUES[0], true, false, true, RANDOM_STEPS
        ));
        
        // 8. Reactive + Random Plus Greedy
        configs.add(new ExperimentConfig(
            "GRASP_Reactive_RandomPlusGreedy",
            0.0, false, true, true, RANDOM_STEPS
        ));
        
        // 9. Random Plus Greedy with different alphas
        for (double alpha : ALPHA_VALUES) {
            if (alpha != ALPHA_VALUES[0]) { 
                configs.add(new ExperimentConfig(
                    "GRASP_RandomPlusGreedy_alpha=" + alpha,
                    alpha, false, false, true, RANDOM_STEPS
                ));
            }
        }
        
        // 10. First-improving + Reactive + Random Plus Greedy
        configs.add(new ExperimentConfig(
            "GRASP_FirstImp_Reactive_RandomPlusGreedy",
            0.0, true, true, true, RANDOM_STEPS
        ));
        
        return configs;
    }
    
    /**
     * Run a single experiment
     */
    private static ExperimentResult runExperiment(ExperimentConfig config, String instanceFile) {
        System.out.println("  Running: " + config.name + " on " + 
                          instanceFile.substring(instanceFile.lastIndexOf('/') + 1));
        
        ExperimentResult result = new ExperimentResult();
        result.configName = config.name;
        result.instanceFile = instanceFile;
        
        try {
            GRASP_SC_QBF_WithStoppingCriteria grasp = new GRASP_SC_QBF_WithStoppingCriteria(
                config.alpha,
                MAX_TOTAL_ITERATIONS,
                instanceFile,
                config.useFirstImproving,
                config.useReactiveGRASP,
                config.useRandomPlusGreedy,
                config.randomSteps,
                MAX_TIME_MILLIS,
                MAX_ITERATIONS_WITHOUT_IMPROVEMENT
            );
            
            Solution<Integer> solution = grasp.solve();
            
            result.bestCost = solution.cost;
            result.totalIterations = grasp.totalIterations;
            result.iterationsToOptimal = grasp.iterationsToOptimal;
            result.executionTimeMs = grasp.executionTime;
            result.stoppedByTime = grasp.stoppedByTime;
            result.stoppedByNoImprovement = grasp.stoppedByNoImprovement;
            result.solutionSize = solution.size();
            
            System.out.println("    -> Cost: " + (-solution.cost) + 
                             ", Iterations: " + grasp.totalIterations +
                             ", Time: " + (grasp.executionTime/1000.0) + "s");
            
        } catch (IOException e) {
            System.err.println("    -> Error: " + e.getMessage());
            result.bestCost = Double.POSITIVE_INFINITY;
        }
        
        return result;
    }
    
    /**
     * Save results to CSV file
     */
    private static void saveResultsToCSV(List<ExperimentResult> results, String filename) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(filename))) {
            // Write header
            writer.println("Configuration,Instance,BestCost,TotalIterations,IterationsToOptimal," +
                          "ExecutionTimeMs,StoppingCriterion,StoppedByTime,SolutionSize");
            
            // Write results
            for (ExperimentResult result : results) {
                writer.println(result.toCSV());
            }
            
            System.out.println("\nResults saved to: " + filename);
            
        } catch (IOException e) {
            System.err.println("Error saving results: " + e.getMessage());
        }
    }
    
    /**
     * Main method to run all experiments
     */
    public static void main(String[] args) {
        SimpleDateFormat dateFormat = new SimpleDateFormat("yyyyMMdd_HHmmss");
        String timestamp = dateFormat.format(new Date());
        String outputFile = "src/problems/scqbf/solvers/results/grasp_experiments_" + timestamp + ".csv";
        
        System.out.println("==============================================");
        System.out.println("GRASP SC-QBF EXPERIMENTS");
        System.out.println("==============================================");
        System.out.println("Max time per run: " + (MAX_TIME_MILLIS/1000) + " seconds");
        System.out.println("Max iterations without improvement: " + MAX_ITERATIONS_WITHOUT_IMPROVEMENT);
        System.out.println("Output file: " + outputFile);
        System.out.println("==============================================\n");
        
        List<ExperimentConfig> configs = generateConfigurations();
        List<ExperimentResult> allResults = new ArrayList<>();
        
        System.out.println("Total configurations: " + configs.size());
        System.out.println("Total instances: " + INSTANCE_FILES.length);
        System.out.println("Total experiments: " + (configs.size() * INSTANCE_FILES.length));
        System.out.println("\nStarting experiments...\n");
        
        long totalStartTime = System.currentTimeMillis();
        int experimentCount = 0;
        
        for (String instanceFile : INSTANCE_FILES) {
            System.out.println("\n--- Instance: " + 
                             instanceFile.substring(instanceFile.lastIndexOf('/') + 1) + " ---");
            
            for (ExperimentConfig config : configs) {
                experimentCount++;
                System.out.println("\n[Experiment " + experimentCount + "/" + 
                                 (configs.size() * INSTANCE_FILES.length) + "]");
                
                ExperimentResult result = runExperiment(config, instanceFile);
                allResults.add(result);
                
                // Save intermediate results every 10 experiments
                if (experimentCount % 10 == 0) {
                    saveResultsToCSV(allResults, outputFile);
                }
            }
        }
        
        // Save final results
        saveResultsToCSV(allResults, outputFile);
        
        long totalTime = System.currentTimeMillis() - totalStartTime;
        System.out.println("\n==============================================");
        System.out.println("EXPERIMENTS COMPLETED");
        System.out.println("Total time: " + (totalTime/1000.0/60.0) + " minutes");
        System.out.println("Results saved to: " + outputFile);
        System.out.println("==============================================");
        
        // Print summary statistics
        printSummaryStatistics(allResults);
    }
    
    /**
     * Print summary statistics of the experiments
     */
    private static void printSummaryStatistics(List<ExperimentResult> results) {
        System.out.println("\n=== SUMMARY STATISTICS ===\n");
        
        // Group results by instance
        for (String instanceFile : INSTANCE_FILES) {
            String instanceName = instanceFile.substring(instanceFile.lastIndexOf('/') + 1);
            System.out.println("Instance: " + instanceName);
            
            ExperimentResult bestResult = null;
            double bestCost = Double.POSITIVE_INFINITY;
            
            for (ExperimentResult result : results) {
                if (result.instanceFile.equals(instanceFile) && result.bestCost < bestCost) {
                    bestCost = result.bestCost;
                    bestResult = result;
                }
            }
            
            if (bestResult != null) {
                System.out.println("  Best configuration: " + bestResult.configName);
                System.out.println("  Best cost: " + (-bestResult.bestCost));
                System.out.println("  Iterations: " + bestResult.totalIterations);
                System.out.println("  Time: " + (bestResult.executionTimeMs/1000.0) + " seconds");
            }
            System.out.println();
        }
    }
}