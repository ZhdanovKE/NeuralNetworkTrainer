package neuralnetwork.samples;

import neuralnetwork.TestUtils;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkSamplesNormalizerSymTest {
    
    public NeuralNetworkSamplesNormalizerSymTest() {
    }

    /**
     * Test of normalize method, of class NeuralNetworkSamplesNormalizerSym.
     */
    @Test(expected = NullPointerException.class)
    public void testNormalizeArrSamples_NullArgument_Throw() {
        System.out.println("testNormalizeArrSamples_NullArgument_Throw");
        double[][] samples = null;
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym();
        instance.normalize(samples);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeArrSamples_EmptyArgument_Throw() {
        System.out.println("testNormalizeArrSamples_NullArgument_Throw");
        double[][] samples = {};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym();
        instance.normalize(samples);
        
        fail("The test case must throw");
    }
    
    @Test(expected = NullPointerException.class)
    public void testNormalizeArrSamples_OneOfSamplesNull_Throw() {
        System.out.println("testNormalizeArrSamples_OneOfSamplesNull_Throw");
        double[][] samples = {{-1, 2}, null, {3, -5}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym();
        instance.normalize(samples);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeArrSamples_SamplesHaveDifferentLength_Throw() {
        System.out.println("testNormalizeArrSamples_SamplesHaveDifferentLength_Thro");
        double[][] samples = {{-1, 2}, {3, -2, 5}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym();
        instance.normalize(samples);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testNormalizeArrSamples_ValidSamples_SamplesChangedToBeInSymmetricRange() {
        System.out.println("testNormalizeArrSamples_ValidSamples_SamplesChangedToBeInSymmetricRange");
        double[][] samples = {{1, 2}, {3, 2}, {-5, 6}, {0, 100000}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym();
        instance.normalize(samples);
        
        for (double[] sample : samples) {
            for (double value : sample) {
                if (value < -1 || value > 1) {
                    fail("Not in [-1;1]");
                }
            }
        }
    }
    
    @Test
    public void testNormalizeArrSamples_SameDifferenceSamples_SamplesChangedToBeInSymmetricRange() {
        System.out.println("testNormalizeArrSamples_SameDifferenceSamples_SamplesChangedToBeInSymmetricRange");
        double[][] samples = {{0, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}};
        double[][] expectedSamples = {{-1, -1}, {-0.5, -0.5}, {0, 0}, {0.5, 0.5}, {1, 1}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym();
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
        double[][] samples = {{-0.6, 1}, {0.5, -0.3}, {-1, 0.9}, {1, -1}};
        double[][] samplesExpected = samples.clone();
        for (int i = 0; i < samplesExpected.length; i++) {
            samplesExpected[i] = samples[i].clone();
        }
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym();
        instance.normalize(samples);
        
        for (int i = 0; i < samples.length; i++) {
            assertArrayEquals(samplesExpected[i], samples[i], TestUtils.DELTA);
        }
    }
    
    @Test
    public void testNormalizeArrSamples_InitWithSamplesArray_ChangesSamplesToSymmetricRange() {
        System.out.println("testNormalizeArrSamples_InitWithSamplesArray_ChangesSamplesToSymmetricRange");
        double[][] samples = {{1, 2}, {3, 2}, {-5, 6}, {0, 100000}};
        double[][] samplesToNormalize = {{-3, 4}, {0, 5643}, {3, 5.4}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym(samples);
        instance.normalize(samplesToNormalize);
        
        for (double[] sample : samplesToNormalize) {
            for (double value : sample) {
                if (value < -1 || value > 1) {
                    fail("Not in [-1;1]");
                }
            }
        }
    }
    
    @Test
    public void testNormalizeArrSamples_InitWithSamplesArrayAlreadyInRange_SamplesNotChanged() {
        System.out.println("testNormalizeArrSamples_InitWithSamplesArrayAlreadyInRange_SamplesNotChanged");
        double[][] samples = {{-1, 1}, {0.5, -0.3}, {-0.1, -1}, {1, 0.356}};
        double[][] samplesToNormalize = {{0.4, -1}, {-1, 1}, {0, 0}};
        double[][] samplesExpected = samplesToNormalize.clone();
        for (int i = 0; i < samplesExpected.length; i++) {
            samplesExpected[i] = samplesToNormalize[i].clone();
        }
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym(samples);
        instance.normalize(samplesToNormalize);
        
        for (int i = 0; i < samplesToNormalize.length; i++) {
            assertArrayEquals(samplesExpected[i], samplesToNormalize[i], TestUtils.DELTA);
        }
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeArrSamples_InitWithSamplesArrayOneOfSamplesDifferentSize_Throw() {
        System.out.println("testNormalizeArrSamples_InitWithSamplesArrayOneOfSamplesDifferentSize_Throw");
        double[][] samples = {{1, -2}, {3, 2}, {-5, 6}, {0, 100000}};
        double[][] samplesToNormalize = {{-3, 4}, {0}, {3, 5.4}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym(samples);
        instance.normalize(samplesToNormalize);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testNormalizeArrSamples_InitWithMinsMaxs_ChangesSamplesToSymmetricRange() {
        System.out.println("testNormalizeArrSamples_InitWithMinsMaxs_ChangesSamplesToSymmetricRange");
        double[] sampleMins = {-5, 2};
        double[] sampleMaxs = {3, 10000};
        double[][] samplesToNormalize = {{-3, 4}, {0, 5643}, {3, 5.4}};
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym(sampleMins, sampleMaxs);
        instance.normalize(samplesToNormalize);
        
        for (double[] sample : samplesToNormalize) {
            for (double value : sample) {
                if (value < -1 || value > 1) {
                    fail("Not in [-1;1]");
                }
            }
        }
    }
    
    @Test
    public void testNormalizeArrSamples_InitWithMinsMaxsZeroOne_SamplesNotChanged() {
        System.out.println("testNormalizeArrSamples_InitWithMinsMaxsZeroOne_SamplesNotChanged");
        double[] sampleMins = {-1, -1};
        double[] sampleMaxs = {1, 1};
        double[][] samplesToNormalize = {{-0.4, 1}, {1, -1}, {0, 0}};
        double[][] samplesExpected = samplesToNormalize.clone();
        for (int i = 0; i < samplesExpected.length; i++) {
            samplesExpected[i] = samplesToNormalize[i].clone();
        }
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym(sampleMins, sampleMaxs);
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
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym(sampleMins, sampleMaxs);
        instance.normalize(samplesToNormalize);
        
        fail("The test case must throw");
    }

    /**
     * Test of normalize method, of class NeuralNetworkSamplesNormalizerSym.
     */
    @Test(expected = NullPointerException.class)
    public void testNormalizeSample_NullArgument_Throw() {
        System.out.println("testNormalizeSample_NullArgument_Throw");
        double[] sample = null;
        NeuralNetworkSamplesNormalizerSym instance = new NeuralNetworkSamplesNormalizerSym();
        instance.normalize(sample);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeSample_EmptyArgument_Throw() {
        System.out.println("testNormalizeSample_EmptyArgument_Throw");
        double[] sample = {};
        NeuralNetworkSamplesNormalizerSym instance = new NeuralNetworkSamplesNormalizerSym();
        instance.normalize(sample);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalStateException.class)
    public void testNormalizeSample_NotInitialized_Throw() {
        System.out.println("testNormalizeSample_NotInitialized_Throw");
        double[] sample = {0, -4, 3, 5};
        NeuralNetworkSamplesNormalizerSym instance = new NeuralNetworkSamplesNormalizerSym();
        instance.normalize(sample);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testNormalizeSample_InitializedWithSamplesArray_ChangeSampleToBeInSymmetricRange() {
        System.out.println("testNormalizeSample_InitializedWithSamplesArray_ChangeSampleToBeInSymmetricRange");
        double[][] samples = {{1, 2}, {3, 2}, {-5, 6}, {0, 100000}};
        double[] sample = {0, 4};
        NeuralNetworkSamplesNormalizerSym instance = new NeuralNetworkSamplesNormalizerSym(samples);
        instance.normalize(sample);
        
        for (double value : sample) {
            if (value < -1 || value > 1) {
                fail("Sample's value is out of [-1;1]");
            }
        }
    }
    
    @Test
    public void testNormalizeSample_InitializedWithSamplesArrayAlreadyInRange_SampleNotChanged() {
        System.out.println("testNormalizeSample_InitializedWithSamplesArrayAlreadyInRange_SampleNotChanged");
        double[][] samples = {{0.4, -1}, {-1, 0}, {1, 0.9}, {0.2, 1}};
        double[] sampleToNormalize = {0, -0.4};
        double[] sampleExpected = sampleToNormalize.clone();
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym(samples);
        instance.normalize(sampleToNormalize);
        
        assertArrayEquals(sampleExpected, sampleToNormalize, TestUtils.DELTA);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeSample_InitializedWithSamplesArraySampleDifferentSize_Throw() {
        System.out.println("testNormalizeSample_InitializedWithSamplesArraySampleDifferentSize_Throw");
        double[][] samples = {{1, 2}, {3, 2}, {-5, 6}, {0, 100000}};
        double[] sample = {0};
        NeuralNetworkSamplesNormalizerSym instance = new NeuralNetworkSamplesNormalizerSym(samples);
        instance.normalize(sample);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testNormalizeSample_InitializedWithMinsMaxs_ChangeSampleToBeInSymmetricRange() {
        System.out.println("testNormalizeSample_InitializedWithMinsMaxs_ChangeSampleToBeInSymmetricRange");
        double[] sampleMins = {-5, 2};
        double[] sampleMaxs = {3, 10000};
        double[] sample = {0, 4};
        NeuralNetworkSamplesNormalizerSym instance = new NeuralNetworkSamplesNormalizerSym(sampleMins, sampleMaxs);
        instance.normalize(sample);
        
        for (double value : sample) {
            if (value < -1 || value > 1) {
                fail("Sample's value is out of [-1;1]");
            }
        }
    }
    
    @Test
    public void testNormalizeSample_InitWithMinsMaxsZeroOne_SampleNotChanged() {
        System.out.println("testNormalizeSample_InitWithMinsMaxsZeroOne_SampleNotChanged");
        double[] sampleMins = {-1, -1};
        double[] sampleMaxs = {1, 1};
        double[] sampleToNormalize = {0, -0.4};
        double[] sampleExpected = sampleToNormalize.clone();
        NeuralNetworkSamplesNormalizer instance = new NeuralNetworkSamplesNormalizerSym(sampleMins, sampleMaxs);
        instance.normalize(sampleToNormalize);
        
        assertArrayEquals(sampleExpected, sampleToNormalize, TestUtils.DELTA);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testNormalizeSample_InitializedWithMinsMaxsSampleDifferentSize_Throw() {
        System.out.println("testNormalizeSample_InitializedWithMinsMaxsSampleDifferentSize_Throw");
        double[] sampleMins = {-5, 2};
        double[] sampleMaxs = {3, 10000};
        double[] sample = {0, 4, 4};
        NeuralNetworkSamplesNormalizerSym instance = new NeuralNetworkSamplesNormalizerSym(sampleMins, sampleMaxs);
        instance.normalize(sample);
        
        fail("The test case must throw");
    }
}

