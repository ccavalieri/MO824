package problems.scqbf.solvers;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import metaheuristics.ga.AbstractGA;
import problems.Evaluator;
import problems.scqbf.SC_QBF;
import solutions.Solution;

/**
 * Genetic Algorithm implementation for the MAX-SC-QBF problem
 */
public class GA_SC_QBF_TTT extends AbstractGA<Integer, Integer> {

    private static final double uniformCrossoverP = 0.5;

    // Configuration parameters
    private boolean usePopulationFunction; // For population variation
    private boolean useMutationFunction; // For Mutation variation
    private String crossoverStrategy; // For Crossover variation
    private String mutationStrategy; // For Mutation variation
    private String parentSelectStrategy; // For Parent selection variation
    private long maxTimeMillis = 30 * 60 * 1000; // 30 minutes
    
    // Stopping criteria constants
    private static final long MAX_TIME_MS = 30 * 60 * 1000; // 30 minutes
    private static final int MAX_GENERATIONS = 2000;
    private static final int MAX_GENERATIONS_NO_IMPROVEMENT = 200;
    
    // Statistics tracking
    private long startTime;
    private int generationsNoImprovement;
    private int actualGenerations;
    private String stopReason;
    
    // --- Early-stop by target ---
    private Double targetValue = null; 
    private long timeToTargetMs = -1;

    public GA_SC_QBF_TTT(Evaluator<Integer> objFunction, Integer generations, Integer popSize, 
                     Double mutationRate, boolean usePopFunction, boolean useMutFunction,
                     String crossoverStrategy, String mutationStrategy, String parentSelectStrategy,
                     int maxTimeMillis, Double targetValue) {
        super(objFunction, generations, popSize, mutationRate);
        this.usePopulationFunction = usePopFunction;
        this.useMutationFunction = useMutFunction;
        this.crossoverStrategy = crossoverStrategy;
        this.mutationStrategy = mutationStrategy;
        this.parentSelectStrategy = parentSelectStrategy;
        this.maxTimeMillis = maxTimeMillis;
        this.targetValue = targetValue;

        if (usePopulationFunction) {
            this.popSize = calculatePopulationSize(chromosomeSize);
        }
        if (useMutationFunction) {
            this.mutationRate = calculateMutationRate(chromosomeSize);
        }
    }
    
    private int calculatePopulationSize(int n) {
        return Math.max(50, 2 * n);
    }
    
    private double calculateMutationRate(int n) {
        return 1.0 / n;
    }
    
    @Override
    public Solution<Integer> createEmptySol() {
        return new Solution<Integer>();
    }
    
    @Override
    protected Solution<Integer> decode(Chromosome chromosome) {
        Solution<Integer> solution = createEmptySol();
        for (int i = 0; i < chromosome.size(); i++) {
            if (chromosome.get(i) == 1) {
                solution.add(i);
            }
        }
        solution.cost = ObjFunction.evaluate(solution);
        return solution;
    }
    
    @Override
    protected Chromosome generateRandomChromosome() {
        Chromosome chromosome = new Chromosome();
        
        // Generate random chromosome with higher probability of 1s
        for (int i = 0; i < chromosomeSize; i++) {
            chromosome.add(rng.nextDouble() < 0.5 ? 1 : 0);
        }
        
        // Ensure at least one feasible solution
        Solution<Integer> testSol = decode(chromosome);
        if (testSol.cost.equals(Double.POSITIVE_INFINITY)) {
            // Add all subsets to guarantee feasible solution
            for (int i = 0; i < chromosomeSize; i++) {
                chromosome.set(i, 1);
            }
        }
        
        return chromosome;
    }
    
    @Override
    protected Double fitness(Chromosome chromosome) {
        Solution<Integer> solution = decode(chromosome);
        
        return solution.cost;
    }
    
    @Override
    protected void mutateGene(Chromosome chromosome, Integer locus) {
        chromosome.set(locus, 1 - chromosome.get(locus));
    }

    @Override
    protected Population crossover(Population parents) {
        if (this.crossoverStrategy.equals("uniform_crossover")) {
            Population offsprings = new Population();
            for (int i = 0; i < popSize; i += 2) {
                Chromosome parent1 = parents.get(i);
                Chromosome parent2 = parents.get(i + 1);

                Chromosome offspring1 = new Chromosome();
                Chromosome offspring2 = new Chromosome();

                for (int j = 0; j < chromosomeSize; j++) {
                    if (rng.nextDouble() < uniformCrossoverP) {
                        // no exchange
                        offspring1.add(parent1.get(j));
                        offspring2.add(parent2.get(j));
                    } else {
                        // exchange
                        offspring1.add(parent2.get(j));
                        offspring2.add(parent1.get(j));
                    }
                }

                offsprings.add(offspring1);
                offsprings.add(offspring2);
            }

            return offsprings;
        } else {
            return super.crossover(parents);
        }
    }
    
    @Override
    protected Population mutate(Population offsprings) {
        if (this.mutationStrategy.equals("adaptative_mutation")) {
            double meanFitness = 0.0;
            for (Chromosome c : offsprings) {
                meanFitness += fitness(c);
            }
            meanFitness /= offsprings.size();

            double stdDev = 0.0;
            for (Chromosome c : offsprings) {
                stdDev += Math.pow(fitness(c) - meanFitness, 2);
            }
            stdDev = Math.sqrt(stdDev / offsprings.size());

            double coefficientOfVariation = stdDev / meanFitness;
            double adjustedMutationRate = 0.5 * (1 - coefficientOfVariation);
            adjustedMutationRate = Math.max(0.0, Math.min(1.0, adjustedMutationRate));

            for (Chromosome c : offsprings) {
                for (int locus = 0; locus < chromosomeSize; locus++) {
                    if (rng.nextDouble() < adjustedMutationRate) {
                        mutateGene(c, locus);
                    }
                }
            }

            return offsprings;
        } else {
            return super.mutate(offsprings);
        }
    }

    @Override
    protected Population initializePopulation() {
        Population population = new Population();
        
        // Ensure first chromosome is all 1s (always feasible for first population)
        Chromosome allOnes = new Chromosome();
        for (int i = 0; i < chromosomeSize; i++) {
            allOnes.add(1);
        }
        population.add(allOnes);
        
        // Generate rest of population
        while (population.size() < popSize) {
            population.add(generateRandomChromosome());
        }
        
        return population;
    }

    @Override
    protected Population selectParents(Population pop) {
        if (this.parentSelectStrategy.equals("stochastic_universal_selection")) {
            int P = popSize, N = pop.size();
            Population parents = new Population();

            double[] w = new double[N];
            double minF = Double.POSITIVE_INFINITY, total = 0.0;
            for (int i = 0; i < N; i++) minF = Math.min(minF, fitness(pop.get(i)));
            double eps = 1e-9;
            for (int i = 0; i < N; i++) {
                w[i] = Math.max(0.0, fitness(pop.get(i)) - minF + eps);
                total += w[i];
            }

            if (total <= 0.0) {
                for (int k = 0; k < P; k++) parents.add(pop.get(rng.nextInt(N)));
                return parents;
            }

            double step = total / P;
            double start = rng.nextDouble() * step;

            double acc = 0.0;
            int k = 0;
            for (int i = 0; i < N && k < P; i++) {
                acc += w[i];
                while (k < P && acc >= start + k * step) {
                    parents.add(pop.get(i));
                    k++;
                }
            }

            while (parents.size() < P) parents.add(pop.get(rng.nextInt(N)));
            return parents;
        } else {
            return super.selectParents(pop);
        }
    }
    
    @Override
    public Solution<Integer> solve() {
        startTime = System.currentTimeMillis();
        generationsNoImprovement = 0;
        actualGenerations = 0;
        stopReason = "";
        
        Population population = initializePopulation();
        
        bestChromosome = getBestChromosome(population);
        bestSol = decode(bestChromosome);
        System.out.println("(Gen. 0) BestSol = " + bestSol);
        
        if (targetValue != null) {
            double valorInicial = bestSol.cost;
            if (valorInicial >= targetValue && timeToTargetMs < 0) {
                timeToTargetMs = System.currentTimeMillis() - startTime;
                return bestSol; 
            }
        }
        
        for (int g = 1; g <= generations; g++) {
            actualGenerations = g;
            
            if (System.currentTimeMillis() - startTime > maxTimeMillis) {                
                break;
            }
            
            if (checkStoppingCriteria(g)) {
                break;
            }
            
            Population parents = selectParents(population);
            Population offsprings = crossover(parents);
            Population mutants = mutate(offsprings);
            Population newPopulation = selectPopulation(mutants);
            
            population = newPopulation;
            
            Chromosome currentBest = getBestChromosome(population);
            double currentFitness = fitness(currentBest);
            
            if (currentFitness > fitness(bestChromosome)) {
                bestChromosome = currentBest;
                bestSol = decode(bestChromosome);
                
                // --- EARLY-STOP by target ---
                if (targetValue != null) {
                    double valorAtual = bestSol.cost;
                    if (valorAtual >= targetValue) {
                        if (timeToTargetMs < 0) {
                            timeToTargetMs = System.currentTimeMillis() - startTime;
                        }
                        break; 
                    }
                }
                
                generationsNoImprovement = 0;
                if (verbose) {
                    System.out.println("(Gen. " + g + ") BestSol = " + bestSol);
                }
            } else {
                generationsNoImprovement++;
            }
        }
        
        if (stopReason.isEmpty()) {
            stopReason = "MAX_GENERATIONS";
        }
        
        return bestSol;
    }
    
    private boolean checkStoppingCriteria(int generation) {
        long elapsedTime = System.currentTimeMillis() - startTime;
        if (elapsedTime >= MAX_TIME_MS) {
            stopReason = "TIME_LIMIT";
            return true;
        }
        
        if (generation >= MAX_GENERATIONS) {
            stopReason = "MAX_GENERATIONS";
            return true;
        }
        
        if (generationsNoImprovement >= MAX_GENERATIONS_NO_IMPROVEMENT) {
            stopReason = "NO_IMPROVEMENT";
            return true;
        }
        
        return false;
    }
    
    public String getStopReason() {
        return stopReason;
    }
    
    public long getElapsedTime() {
        return System.currentTimeMillis() - startTime;
    }
    
    public int getActualGenerations() {
        return actualGenerations;
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

    
    public static void main(String[] args) throws IOException {
        String instance = "instances/scqbf/n400p2.txt";
        int populationSize = 100;
        int maxGenerations = 100000;
        double crossoverRate = 0.9;
        double mutationRate  = 0.02;
        int     maxTimeMillis      = 10*60*1000;

        // target
        Double target = 10486.78; 

        int runs = 100;          
        long baseSeed = 2025L; 

        // Output 
        String csvPath = "resultados_ga_scqbf.csv";
        writeCsvHeaderIfNeeded(csvPath);
        
        SC_QBF scqbf = new SC_QBF(instance);        

        for (int run = 0; run < runs; run++) {
            long seed = baseSeed + run;
            
            
            GA_SC_QBF_TTT ga = new GA_SC_QBF_TTT(scqbf, maxGenerations, populationSize, mutationRate,
                    true, true, "uniform_crossover", "standard", "standard", maxTimeMillis, target);
            
           
            
            ga.setSeed(seed);

            long start = System.currentTimeMillis();
            Solution<Integer> best = ga.solve();
            long total = System.currentTimeMillis() - start;

            long ttt = ga.getTimeToTargetMs(); 

            appendCsvLine(csvPath, instance, target, seed, ttt);

            System.out.printf(
                    "Run %d | seed=%d | best=%.4f | t_total=%.3fs | t_target=%s%n",
                    run, seed, -best.cost, total/1000.0, (ttt >= 0 ? (ttt/1000.0 + "s") : "NA")
            );
        }
    }

}