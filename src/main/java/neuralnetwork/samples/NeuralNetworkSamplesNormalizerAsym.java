package neuralnetwork.samples;

/**
 * Non-symmetric normalizer mapping every sample into [0; 1]
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkSamplesNormalizerAsym extends AbstractNeuralNetworkSamplesNormalizer {
    
    /**
     * <p>Create a normalizer for neural network's inputs</p>
     */
    public NeuralNetworkSamplesNormalizerAsym() {
    }
    
    public NeuralNetworkSamplesNormalizerAsym(double[] minSamplesValues,
                                                double[] maxSamplesValues) {
        
        super(minSamplesValues, maxSamplesValues);
    }
    
    public NeuralNetworkSamplesNormalizerAsym(double[][] samples) {
        super(samples);
    }
            
    @Override
    protected void doNormalization(double[][] samples) {
        for (int varNum = 0; varNum < minSamplesValues.length; varNum++) {
            if (maxSamplesValues[varNum] != minSamplesValues[varNum]) {
                double difference = maxSamplesValues[varNum] - minSamplesValues[varNum];
                for (double[] sample : samples) {
                    sample[varNum] = (sample[varNum] - minSamplesValues[varNum]) / difference;
                }
            }
        }
    }
    
    @Override
    protected void doNormalization(double[] sample) {
        for (int varNum = 0; varNum < minSamplesValues.length; varNum++) {
            if (maxSamplesValues[varNum] != minSamplesValues[varNum]) {
                double difference = maxSamplesValues[varNum] - minSamplesValues[varNum];
                
                sample[varNum] = (sample[varNum] - 
                        minSamplesValues[varNum])/difference;
            }
        }
    }
   
}
