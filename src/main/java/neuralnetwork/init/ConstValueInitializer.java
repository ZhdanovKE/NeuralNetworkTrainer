package neuralnetwork.init;

/**
 * Implementation of {@link Initializer} that initializes all weights with
 * one constant value and all biases with another constant value.
 * @author Konstantin Zhdanov
 */
class ConstValueInitializer implements Initializer {
    final double weightsValue;
    final double biasesValue;
    
    public ConstValueInitializer(double weightsValue, double biasesValue) {
        this.weightsValue = weightsValue;
        this.biasesValue = biasesValue;
    }

    @Override
    public double supplyWeight(int layerNum, int prevLayerNeuronNum, int layerNeuronNum) {
        return weightsValue;
    }

    @Override
    public double supplyBias(int layerNum, int layerNeuronNum) {
        return biasesValue;
    }
}
