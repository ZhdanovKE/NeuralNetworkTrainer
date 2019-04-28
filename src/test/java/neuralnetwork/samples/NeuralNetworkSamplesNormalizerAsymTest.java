package neuralnetwork.samples;

import neuralnetwork.TestUtils;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkSamplesNormalizerAsymTest {
    
    public NeuralNetworkSamplesNormalizerAsymTest() {
    }

    /**
     * Test of normalize method, of class NeuralNetworkSamplesNormalizerAsym.
     */
    @Test(expected = NullPointerException.class)
    public void testNormalizeArrSamples_NullArgument_Throw() {
        System.out.println("testNormalizeArrSamples_NullArgument_Throw");
        double[][] samples = null;
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym();
        instance.normalize(samples);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeArrSamples_EmptyArgument_Throw() {
        System.out.println("testNormalizeArrSamples_NullArgument_Throw");
        double[][] samples = {};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym();
        instance.normalize(samples);
        
        fail("The test case must throw");
    }
    
    @Test(expected = NullPointerException.class)
    public void testNormalizeArrSamples_OneOfSamplesNull_Throw() {
        System.out.println("testNormalizeArrSamples_OneOfSamplesNull_Throw");
        double[][] samples = {{1, 2}, null, {3, 2}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym();
        instance.normalize(samples);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeArrSamples_SamplesHaveDifferentLength_Throw() {
        System.out.println("testNormalizeArrSamples_SamplesHaveDifferentLength_Thro");
        double[][] samples = {{1, 2}, {3, 2, 5}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym();
        instance.normalize(samples);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testNormalizeArrSamples_ValidSamples_SamplesChangedToBeInAsymmetricRange() {
        System.out.println("testNormalizeArrSamples_ValidSamples_SamplesChangedToBeInAsymmetricRange");
        double[][] samples = {{1, 2}, {3, 2}, {-5, 6}, {0, 100000}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym();
        instance.normalize(samples);
        
        for (double[] sample : samples) {
            for (double value : sample) {
                if (value < 0 || value > 1) {
                    fail("Not in [0;1]");
                }
            }
        }
    }
    
    @Test
    public void testNormalizeArrSamples_SameDifferenceSamples_SamplesChangedToBeInAsymmetricRange() {
        System.out.println("testNormalizeArrSamples_SameDifferenceSamples_SamplesChangedToBeInAsymmetricRange");
        double[][] samples = {{0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}};
        double[][] expectedSamples = {{0, 0}, {0.25, 0.25}, {0.5, 0.5}, {0.75, 0.75}, {1, 1}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym();
        instance.normalize(samples);
        
        for (int sampleNum = 0; sampleNum < samples.length; sampleNum++) {
            for (int sampleVar = 0; sampleVar < samples[sampleNum].length; sampleVar++) {
                assertEquals(expectedSamples[sampleNum][sampleVar], samples[sampleNum][sampleVar], TestUtils.DELTA);
            }
        }
    }
    
    @Test
    public void testNormalizeArrSamples_SamplesAlreadyInRange_SamplesNotChanged() {
        System.out.println("testNormalizeArrSamples_SamplesAlreadyInRange_SamplesNotChanged");
        double[][] samples = {{0, 1}, {0.5, 0}, {1, 0.9}, {0.2, 0.356}};
        double[][] samplesExpected = samples.clone();
        for (int i = 0; i < samplesExpected.length; i++) {
            samplesExpected[i] = samples[i].clone();
        }
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym();
        instance.normalize(samples);
        
        for (int i = 0; i < samples.length; i++) {
            assertArrayEquals(samplesExpected[i], samples[i], TestUtils.DELTA);
        }
    }
    
    @Test
    public void testNormalizeArrSamples_InitWithSamplesArray_ChangesSamplesToAsymmetricRange() {
        System.out.println("testNormalizeArrSamples_InitWithSamplesArray_ChangesSamplesToAsymmetricRange");
        double[][] samples = {{1, 2}, {3, 2}, {-5, 6}, {0, 100000}};
        double[][] samplesToNormalize = {{-3, 4}, {0, 5643}, {3, 5.4}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym(samples);
        instance.normalize(samplesToNormalize);
        
        for (double[] sample : samplesToNormalize) {
            for (double value : sample) {
                if (value < 0 || value > 1) {
                    fail("Not in [0;1]");
                }
            }
        }
    }
    
    @Test
    public void testNormalizeArrSamples_InitWithSamplesArrayAlreadyInRange_SamplesNotChanged() {
        System.out.println("testNormalizeArrSamples_InitWithSamplesArrayAlreadyInRange_SamplesNotChanged");
        double[][] samples = {{0, 1}, {0.5, 0}, {0.1, 0.9}, {1, 0.356}};
        double[][] samplesToNormalize = {{0.4, 1}, {1, 1}, {0, 0}};
        double[][] samplesExpected = samplesToNormalize.clone();
        for (int i = 0; i < samplesExpected.length; i++) {
            samplesExpected[i] = samplesToNormalize[i].clone();
        }
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym(samples);
        instance.normalize(samplesToNormalize);
        
        for (int i = 0; i < samplesToNormalize.length; i++) {
            assertArrayEquals(samplesExpected[i], samplesToNormalize[i], TestUtils.DELTA);
        }
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeArrSamples_InitWithSamplesArrayOneOfSamplesDifferentSize_Throw() {
        System.out.println("testNormalizeArrSamples_InitWithSamplesArrayOneOfSamplesDifferentSize_Throw");
        double[][] samples = {{1, 2}, {3, 2}, {-5, 6}, {0, 100000}};
        double[][] samplesToNormalize = {{-3, 4}, {0}, {3, 5.4}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym(samples);
        instance.normalize(samplesToNormalize);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testNormalizeArrSamples_InitWithMinsMaxs_ChangesSamplesToAsymmetricRange() {
        System.out.println("testNormalizeArrSamples_InitWithMinsMaxs_ChangesSamplesToAsymmetricRange");
        double[] sampleMins = {-5, 2};
        double[] sampleMaxs = {3, 10000};
        double[][] samplesToNormalize = {{-3, 4}, {0, 5643}, {3, 5.4}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym(sampleMins, sampleMaxs);
        instance.normalize(samplesToNormalize);
        
        for (double[] sample : samplesToNormalize) {
            for (double value : sample) {
                if (value < 0 || value > 1) {
                    fail("Not in [0;1]");
                }
            }
        }
    }
    
    @Test
    public void testNormalizeArrSamples_InitWithMinsMaxsZeroOne_SamplesNotChanged() {
        System.out.println("testNormalizeArrSamples_InitWithMinsMaxsZeroOne_SamplesNotChanged");
        double[] sampleMins = {0, 0};
        double[] sampleMaxs = {1, 1};
        double[][] samplesToNormalize = {{0.4, 1}, {1, 1}, {0, 0}};
        double[][] samplesExpected = samplesToNormalize.clone();
        for (int i = 0; i < samplesExpected.length; i++) {
            samplesExpected[i] = samplesToNormalize[i].clone();
        }
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym(sampleMins, sampleMaxs);
        instance.normalize(samplesToNormalize);
        
        for (int i = 0; i < samplesToNormalize.length; i++) {
            assertArrayEquals(samplesExpected[i], samplesToNormalize[i], TestUtils.DELTA);
        }
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeArrSamples_InitWithMinsMaxsOneOfSamplesDifferentSize_Throw() {
        System.out.println("testNormalizeArrSamples_InitWithMinsMaxsOneOfSamplesDifferentSize_Throw");
        double[] sampleMins = {-5, 2};
        double[] sampleMaxs = {3, 10000};
        double[][] samplesToNormalize = {{-3, 4}, {0, 5643, 5}, {3, 5.4}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym(sampleMins, sampleMaxs);
        instance.normalize(samplesToNormalize);
        
        fail("The test case must throw");
    }

    /**
     * Test of normalize method, of class NeuralNetworkSamplesNormalizerAsym.
     */
    @Test(expected = NullPointerException.class)
    public void testNormalizeSample_NullArgument_Throw() {
        System.out.println("testNormalizeSample_NullArgument_Throw");
        double[] sample = null;
        NeuralNetworkSamplesNormalizerAsym instance = new NeuralNetworkSamplesNormalizerAsym();
        instance.normalize(sample);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeSample_EmptyArgument_Throw() {
        System.out.println("testNormalizeSample_EmptyArgument_Throw");
        double[] sample = {};
        NeuralNetworkSamplesNormalizerAsym instance = new NeuralNetworkSamplesNormalizerAsym();
        instance.normalize(sample);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalStateException.class)
    public void testNormalizeSample_NotInitialized_Throw() {
        System.out.println("testNormalizeSample_NotInitialized_Throw");
        double[] sample = {0, 4, 3, 5};
        NeuralNetworkSamplesNormalizerAsym instance = new NeuralNetworkSamplesNormalizerAsym();
        instance.normalize(sample);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testNormalizeSample_InitializedWithSamplesArray_ChangeSampleToBeInAsymmetricRange() {
        System.out.println("testNormalizeSample_InitializedWithSamplesArray_ChangeSampleToBeInAsymmetricRange");
        double[][] samples = {{1, 2}, {3, 2}, {-5, 6}, {0, 100000}};
        double[] sample = {0, 4};
        NeuralNetworkSamplesNormalizerAsym instance = new NeuralNetworkSamplesNormalizerAsym(samples);
        instance.normalize(sample);
        
        for (double value : sample) {
            if (value < 0 || value > 1) {
                fail("Sample's value is out of [0;1]");
            }
        }
    }
    
    @Test
    public void testNormalizeSample_InitializedWithSamplesArrayAlreadyInRange_SampleNotChanged() {
        System.out.println("testNormalizeSample_InitializedWithSamplesArrayAlreadyInRange_SampleNotChanged");
        double[][] samples = {{0.4, 1}, {0, 0}, {1, 0.9}, {0.2, 0.5452}};
        double[] sampleToNormalize = {0, 0.4};
        double[] sampleExpected = sampleToNormalize.clone();
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym(samples);
        instance.normalize(sampleToNormalize);
        
        assertArrayEquals(sampleExpected, sampleToNormalize, TestUtils.DELTA);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeSample_InitializedWithSamplesArraySampleDifferentSize_Throw() {
        System.out.println("testNormalizeSample_InitializedWithSamplesArraySampleDifferentSize_Throw");
        double[][] samples = {{1, 2}, {3, 2}, {-5, 6}, {0, 100000}};
        double[] sample = {0};
        NeuralNetworkSamplesNormalizerAsym instance = new NeuralNetworkSamplesNormalizerAsym(samples);
        instance.normalize(sample);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testNormalizeSample_InitializedWithMinsMaxs_ChangeSampleToBeInAsymmetricRange() {
        System.out.println("testNormalizeSample_InitializedWithMinsMaxs_ChangeSampleToBeInAsymmetricRange");
        double[] sampleMins = {-5, 2};
        double[] sampleMaxs = {3, 10000};
        double[] sample = {0, 4};
        NeuralNetworkSamplesNormalizerAsym instance = new NeuralNetworkSamplesNormalizerAsym(sampleMins, sampleMaxs);
        instance.normalize(sample);
        
        for (double value : sample) {
            if (value < 0 || value > 1) {
                fail("Sample's value is out of [0;1]");
            }
        }
    }
    
    @Test
    public void testNormalizeSample_InitWithMinsMaxsZeroOne_SampleNotChanged() {
        System.out.println("testNormalizeSample_InitWithMinsMaxsZeroOne_SampleNotChanged");
        double[] sampleMins = {0, 0};
        double[] sampleMaxs = {1, 1};
        double[] sampleToNormalize = {0, 0.4};
        double[] sampleExpected = sampleToNormalize.clone();
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerAsym(sampleMins, sampleMaxs);
        instance.normalize(sampleToNormalize);
        
        assertArrayEquals(sampleExpected, sampleToNormalize, TestUtils.DELTA);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeSample_InitializedWithMinsMaxsSampleDifferentSize_Throw() {
        System.out.println("testNormalizeSample_InitializedWithMinsMaxsSampleDifferentSize_Throw");
        double[] sampleMins = {-5, 2};
        double[] sampleMaxs = {3, 10000};
        double[] sample = {0, 4, 4};
        NeuralNetworkSamplesNormalizerAsym instance = new NeuralNetworkSamplesNormalizerAsym(sampleMins, sampleMaxs);
        instance.normalize(sample);
        
        fail("The test case must throw");
    }
}
