package neuralnetwork.init;

/**
 * Implementation of {@link Initializer} interface that simply delegates
 * to the instances of {@link WeightsSupplier} and {@link BiasesSupplier}.
 * @author Konstantin Zhdanov
 */
class DelegatingInitializer implements Initializer {
    final WeightsSupplier weightsSupplier;
    final BiasesSupplier biasesSupplier;
    
    public DelegatingInitializer(WeightsSupplier weightsSupplier, BiasesSupplier biasesSupplier) {
        if (weightsSupplier == null || biasesSupplier == null) {
            throw new NullPointerException("Arguments cannot be null");
        }
        this.weightsSupplier = weightsSupplier;
        this.biasesSupplier = biasesSupplier;
    }

    @Override
    public double supplyWeight(int layerNum, int prevLayerNeuronNum, int layerNeuronNum) {
        return weightsSupplier.supplyWeight(layerNum, prevLayerNeuronNum, layerNeuronNum);
    }

    @Override
    public double supplyBias(int layerNum, int layerNeuronNum) {
        return biasesSupplier.supplyBias(layerNum, layerNeuronNum);
    }
    
    
}
