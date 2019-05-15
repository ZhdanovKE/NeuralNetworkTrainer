package neuralnetwork.init;

/**
 * {@link NeuralNetwork} biases supplier functional interface. 
 * Used to supply initial values for a network's biases.
 * @author Konstantin Zhdanov
 */
@FunctionalInterface
public interface BiasesSupplier {
    /**
     * Supply the bias for the neuron with index {@code layerNeuronNum}
     * at layer {@code layerNum}.
     * @param layerNum Index of the layer one of which neurons biases is to be 
     * initialized. Starts with 0. Layer 0 means the first hidden layer.
     * @param layerNeuronNum Index of the neuron at the level {@code layerNum}.
     *  Starts with 0.
     * @return {@code double} value to be used as initial bias for the neuron.
     */
    double supplyBias(int layerNum, int layerNeuronNum);
}
