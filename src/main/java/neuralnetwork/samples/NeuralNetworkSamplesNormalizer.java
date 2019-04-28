package neuralnetwork.samples;

/**
 * Interface for a normalizer of a neural network's samples (inputs and outputs), 
 * which is used for mapping the samples to the range supported by a training
 * algorithm. 
 * A class implementing this interface defines the range of normalization.
 * @author Konstantin Zhdanov
 */
public interface NeuralNetworkSamplesNormalizer {

    /**
     * Normalize the {@link samples} array of samples by mapping them in a 
     * predefined range.
     * The {@link samples} array will contain the normalized values.
     * @param samples a 2-D {@code double} array of samples to be normalized (in-place).
     */
    void normalize(double[][] samples);

    /**
     * Normalize one {@link sample} by mapping it in a predefined range.
     * The {@link sample} array will contain the normalized values.
     * @param sample a {@code double} array of the sample's values to
     * be normalized (in-place).
     */
    void normalize(double[] sample);
}
