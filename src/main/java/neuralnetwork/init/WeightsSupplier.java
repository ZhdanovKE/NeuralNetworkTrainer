package neuralnetwork.init;

/**
 * {@link NeuralNetwork} weights supplier functional interface. 
 * Used to supply initial values for a network's weights.
 * @author Konstantin Zhdanov
 */
@FunctionalInterface
public interface WeightsSupplier {
    /**
     * Supply the weight between the neuron with index {@code prevLayerNeuronNum}
     * at layer {@code layerNum - 1} and the neuron with index 
     * {@code layerNeuronNum} at layer {@code layerNum}.
     * @param layerNum Index of the one of the two layers that is closer to the 
     * output layer. Starts with 0. Layer 0 means the first hidden layer.
     * @param prevLayerNeuronNum Index of the neuron at the level {@code layerNum - 1}.
     *  Starts with 0.
     * @param layerNeuronNum Index of the neuron at the level {@code layerNum}.
     *  Starts with 0.
     * @return {@code double} value to be used as initial weight between the two 
     * neurons.
     */
    double supplyWeight(int layerNum, int prevLayerNeuronNum, int layerNeuronNum);
    
}
