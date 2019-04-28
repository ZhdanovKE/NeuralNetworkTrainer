package neuralnetwork.samples;

import java.util.Arrays;

/**
 * Abstract base implementation of the common operation for a normalizer that maps
 * each sample's value into the same range. The class defines two abstract methods
 * {@code doNormalization(double[][] samples)} and {@code doNormalization(double[] sample)} 
 * that should perform the real mapping to a predefined range. Concrete classes must 
 * implement these methods.
 * @author Konstantin Zhdanov
 */
abstract class AbstractNeuralNetworkSamplesNormalizer implements NeuralNetworkSamplesNormalizer {
    /** Min values for every variable across all samples */
    double[] minSamplesValues;
    
    /** Max values for every variable across all samples */
    double[] maxSamplesValues;
    
    /** Whether the normalizer is initialized with samples' data */
    private boolean initialized = false;
    
    public AbstractNeuralNetworkSamplesNormalizer() {
        
    }
    
    public AbstractNeuralNetworkSamplesNormalizer(double[] minSamplesValues,
                                                    double[] maxSamplesValues) {
        if (minSamplesValues == null || maxSamplesValues == null) {
            throw new NullPointerException("Arguments cannot be null");
        }
        if (minSamplesValues.length == 0 || maxSamplesValues.length == 0) {
            throw new IllegalArgumentException("Max and min values of samples cannot be empty");
        }
        if (minSamplesValues.length != maxSamplesValues.length) {
            throw new IllegalArgumentException("There must be equal number of max and min values");
        }
        
        this.minSamplesValues = minSamplesValues;
        this.maxSamplesValues = maxSamplesValues;
        
        try {
            boolean correct = minsLessOrEqualMaxs();
            if (!correct) {
                this.minSamplesValues = null;
                this.maxSamplesValues = null;
                throw new IllegalArgumentException("Max values must be greater or equal to min values");
            }
        }
        catch (ArrayIndexOutOfBoundsException e) {
            this.minSamplesValues = null;
            this.maxSamplesValues = null;
            throw new IllegalArgumentException("Min and max values are inconsistent");
        }
        
        initialized = true;
    }
    
    public AbstractNeuralNetworkSamplesNormalizer(double[][] samples) {
        if (samples == null) {
            throw new NullPointerException("Samples cannot be null");
        }
        if (samples.length == 0 || samples[0].length == 0) {
            throw new IllegalArgumentException("Samples cannot be empty");
        }
        
        try {
            init(samples);
        }
        catch(ArrayIndexOutOfBoundsException e) {
            throw new IllegalArgumentException("Samples are corrupted");
        }
    }
    
    boolean initialized() {
        return initialized;
    }
    
    private void init(double[][] samples) {
        final int numVars = samples[0].length;
        checkSamplesOfSameSize(samples, numVars);
        minSamplesValues = new double[numVars];
        maxSamplesValues = new double[numVars];
        Arrays.fill(minSamplesValues, Double.POSITIVE_INFINITY);
        Arrays.fill(maxSamplesValues, Double.NEGATIVE_INFINITY);
        
        for (int varNum = 0; varNum < numVars; varNum++) {
            for (double[] sample : samples) {
                if (minSamplesValues[varNum] > sample[varNum]) {
                    minSamplesValues[varNum] = sample[varNum];
                }
                if (maxSamplesValues[varNum] < sample[varNum]) {
                    maxSamplesValues[varNum] = sample[varNum];
                }
            } 
        }
        
        initialized = true;
    }
    
    private void checkSamplesOfSameSize(double[][] samples, int size) {
        for (double[] sample : samples) {
            if (sample.length != size) {
                throw new IllegalArgumentException("Samples have different sizes");
            }
        }
    }
    
    private boolean minsLessOrEqualMaxs() {
        for (int varNum = 0; varNum < minSamplesValues.length; varNum++) {
            if (minSamplesValues[varNum] > maxSamplesValues[varNum]) {
                return false;
            }
        }
        return true;
    }
    
    private void checkSamplesCorrect(double[][] samples) {
        if (samples == null) {
            throw new NullPointerException("Samples cannot be null");
        }
        if (samples.length == 0 || samples[0].length == 0) {
            throw new IllegalArgumentException("Samples cannot be empty");
        }
        if (initialized) {
            checkSamplesOfSameSize(samples, minSamplesValues.length);
        }
    }
    
    private void initialize(double[][] samples) {
        if (!initialized) {
            try {
                init(samples);
            }
            catch(ArrayIndexOutOfBoundsException e) {
                throw new IllegalArgumentException("Samples are corrupt");
            }
        }
    }

    @Override
    public void normalize(double[] sample) {
        if (sample == null) {
            throw new NullPointerException("Sample cannot be null");
        }
        if (sample.length == 0) {
            throw new IllegalArgumentException("Sample cannot be empty");
        }
        if (!initialized) {
            throw new IllegalStateException("Normalizer has not been initialized");
        }
        if (sample.length != minSamplesValues.length) {
            throw new IllegalArgumentException("Sample contains wrong number of values");
        }
        doNormalization(sample);
    }
    
    protected abstract void doNormalization(double[] sample);

    @Override
    public void normalize(double[][] samples) {
        checkSamplesCorrect(samples);
        initialize(samples);
        doNormalization(samples);
    }
    
    protected abstract void doNormalization(double[][] samples);
}
