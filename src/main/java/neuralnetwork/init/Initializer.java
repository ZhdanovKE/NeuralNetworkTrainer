package neuralnetwork.init;

/**
 * {@link NeuralNetwork} weights and biases initializer. Its methods are used
 * to supply initial values for a network's weights and biases.
 * @author Konstantin Zhdanov
 */
public interface Initializer extends WeightsSupplier, BiasesSupplier{
    
    /**
     * Create instance of {@link Initializer} based on instances of
     * {@link WeightsSupplier} and {@link BiasesSupplier}.
     * @param weightsSupplier instance of {@link WeightsSupplier}.
     * @param biasesSupplier instance of {@link BiasesSupplier}.
     * @return Instance that delegates to {@code weightsSupplier} and 
     * {@code biasesSupplier}.
     */
    static Initializer of(WeightsSupplier weightsSupplier, BiasesSupplier biasesSupplier) {
        return new DelegatingInitializer(weightsSupplier, biasesSupplier);
    }
    
    /**
     * Create instance of {@link Initializer} that initializes all weights with
     * the value {@code weightValue} and all biases with the value of 
     * {@code biasesValue}.
     * @param weightsValue Value to be set to all weights.
     * @param biasesValue Value to be set to all biases.
     * @return instance of {@link Initializer} that initializes all weights with
     * the value {@code weightValue} and all biases with the value of 
     * {@code biasesValue}.
     */
    static Initializer of(double weightsValue, double biasesValue) {
        return new ConstValueInitializer(weightsValue, biasesValue);
    }
    
    /**
     * Create instance of {@link Initializer} that initializes all weights
     * and all biases with random {@code double} numbers within range {@code [0,1]}.
     * @return Instance of {@link Initializer} that initializes all weights
     * and all biases with random {@code double} numbers within range {@code [0,1]}.
     */
    static Initializer ofStdRandomRange() {
        return new RandomRangeInitializer(0.0, 1.0, 0.0, 1.0);
    }
    
    /**
     * Create instance of {@link Initializer} that initializes all weights
     * with random {@code double} numbers within range 
     * {@code [minWeightsValue,maxWeightsValue]} and
     * all biases with random {@code double} numbers within range 
     * {@code [minBiasesValue,maxBiasesValue]}.
     * @param minWeightsValue Lower bound on random numbers used to initialize weights.
     * @param maxWeightsValue Upper bound on random numbers used to initialize weights.
     * @param minBiasesValue Lower bound on random numbers used to initialize biases.
     * @param maxBiasesValue Upper bound on random numbers used to initialize biases.
     * @return Instance of {@link Initializer} that initializes all weights
     * with random {@code double} numbers within range 
     * {@code [minWeightsValue,maxWeightsValue]} and
     * all biases with random {@code double} numbers within range 
     * {@code [minBiasesValue,maxBiasesValue]}.
     */
    static Initializer ofCustomRandomRange(
            double minWeightsValue, double maxWeightsValue,
            double minBiasesValue, double maxBiasesValue) {
        return new RandomRangeInitializer(
                minWeightsValue, maxWeightsValue, 
                minBiasesValue, maxBiasesValue);
    }
}


