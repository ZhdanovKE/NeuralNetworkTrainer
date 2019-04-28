package neuralnetwork.samples;

/**
 * Symmetric normalizer mapping every sample into [-1; 1]
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkSamplesNormalizerSym extends AbstractNeuralNetworkSamplesNormalizer {
    
    /**
     * <p>Create a normalizer for neural network's inputs</p>
     */
    public NeuralNetworkSamplesNormalizerSym() {
    }
    
    public NeuralNetworkSamplesNormalizerSym(double[] minSamplesValues,
                                             double[] maxSamplesValues) {
        
        super(minSamplesValues, maxSamplesValues);
    }
    
    public NeuralNetworkSamplesNormalizerSym(double[][] samples) {
        super(samples);
    }
            
    @Override
    protected void doNormalization(double[][] samples) {
        for (int varNum = 0; varNum < minSamplesValues.length; varNum++) {
            if (maxSamplesValues[varNum] != minSamplesValues[varNum]) {
                double difference = maxSamplesValues[varNum] - minSamplesValues[varNum];
                for (int sampleNum = 0; sampleNum < samples.length; sampleNum++) {
                    samples[sampleNum][varNum] = 2*(samples[sampleNum][varNum] - 
                        minSamplesValues[varNum])/difference - 1;
                }
            }
        }
    }
    
    @Override
    protected void doNormalization(double[] sample) {
        for (int varNum = 0; varNum < minSamplesValues.length; varNum++) {
            if (maxSamplesValues[varNum] != minSamplesValues[varNum]) {
                double difference = maxSamplesValues[varNum] - minSamplesValues[varNum];
                
                sample[varNum] = 2*(sample[varNum] - 
                        minSamplesValues[varNum])/difference - 1;
            }
        }
    }
}
