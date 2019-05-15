package neuralnetwork.init;

import java.util.Random;

/**
 * Implementation of {@link Initializer} that initializes all weights and
 * biases with random {@code double} numbers within provided range.
 * @author Konstantin Zhdanov
 */
class RandomRangeInitializer implements Initializer {
    final double minWeightsValue;
    final double weightsRange;
    
    final double minBiasesValue;
    final double biasesRange;
    
    final Random weightsRandom;
    final Random biasesRandom;

    public RandomRangeInitializer(double minWeightsValue, double maxWeightsValue, double minBiasesValue, double maxBiasesValue) {
        if (    minWeightsValue >= maxWeightsValue ||
                minBiasesValue >= maxBiasesValue     ) {
            throw new IllegalArgumentException("Min values must be less than max values");
        }
        this.minWeightsValue = minWeightsValue;
        this.minBiasesValue = minBiasesValue;
        
        weightsRange = maxWeightsValue - this.minWeightsValue;
        biasesRange = maxBiasesValue - this.minBiasesValue;
        
        weightsRandom = new Random();
        biasesRandom = new Random();
    }
    
    @Override
    public double supplyWeight(int layerNum, int prevLayerNeuronNum, int layerNeuronNum) {
        return minWeightsValue + weightsRandom.nextDouble()*weightsRange;
    }

    @Override
    public double supplyBias(int layerNum, int layerNeuronNum) {
        return minBiasesValue + weightsRandom.nextDouble()*biasesRange;
    }
}
