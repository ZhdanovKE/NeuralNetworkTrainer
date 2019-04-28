package neuralnetwork.train;

import neuralnetwork.ActivationFunction;
import neuralnetwork.ActivationFunctions;
import neuralnetwork.NeuralNetwork;
import org.junit.Test;
import static org.junit.Assert.*;
import org.mockito.Matchers;
import org.mockito.Mockito;

import static neuralnetwork.TestUtils.*;
import org.junit.After;
import org.junit.Before;

/**
 *
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkEvaluatorTest {
    
    private int nInputs;
    private int[] hiddenLayerSizes;
    private int nOutputs;
    private NeuralNetwork nn;
    
    public NeuralNetworkEvaluatorTest() {
    }

    @Before
    public void setUpTestCase() {
        nInputs = 3;
        hiddenLayerSizes = new int[]{2};
        nOutputs = 3;
        nn = Mockito.mock(NeuralNetwork.class);
        Mockito.when(nn.getBias(Matchers.anyInt(), Matchers.anyInt())).thenReturn(0.0);
        Mockito.when(nn.getWeight(Matchers.anyInt(), Matchers.anyInt(), Matchers.anyInt())).thenReturn(0.0);
        Mockito.when(nn.getActivationFunction()).thenReturn(ActivationFunctions.SIGMOID);
        Mockito.when(nn.getNumberInputs()).thenReturn(nInputs);
        Mockito.when(nn.getNumberOutputs()).thenReturn(nOutputs);
        Mockito.when(nn.getHiddenLayerSizes()).thenReturn(hiddenLayerSizes);
        Mockito.when(nn.getNumberHiddenLayers()).thenReturn(hiddenLayerSizes.length);
        Mockito.when(nn.getHiddenLayerSize(0)).thenReturn(hiddenLayerSizes[0]);
        Mockito.doThrow(new AssertionError()).when(nn).setBias(Matchers.anyInt(), Matchers.anyInt(), Matchers.anyDouble());
        Mockito.doThrow(new AssertionError()).when(nn).setWeight(Matchers.anyInt(), Matchers.anyInt(), Matchers.anyInt(), Matchers.anyDouble());
        Mockito.doThrow(new AssertionError()).when(nn).setActivationFunction(Matchers.any(ActivationFunction.class));
    }
    
    @After
    public void cleanUpTestCase() {
        nInputs = 0;
        hiddenLayerSizes = null;
        nOutputs = 0;
        nn = null;
    }
    
    /**
     * Test of constructor, of class NeuralNetworkEvaluator.
     */
    @Test(expected = NullPointerException.class)
    public void testConstructor_NullArgument_Throws() {
        new NeuralNetworkEvaluator(null);

        fail("The test case must throw.");
    }
    
    @Test
    public void testConstructor_ValidArgument_Ok() {
        NeuralNetwork nn = Mockito.mock(NeuralNetwork.class);
        
        new NeuralNetworkEvaluator(nn);
    }
    
    /**
     * Test of evaluate method, of class NeuralNetworkEvaluator.
     */
    @Test(expected = NullPointerException.class)
    public void testEvaluate_NullArgument_Throws() {
        double[] input = null;
        NeuralNetwork nn = Mockito.mock(NeuralNetwork.class);
        NeuralNetworkEvaluator instance = new NeuralNetworkEvaluator(nn);
        
        instance.evaluate(input);

        fail("The test case must throw.");
    }
    
    @Test
    public void testEvaluate_ZeroInputZeroNN_ReturnsZeroPointFive() {
        double[] input = {0, 0, 0};
        
        NeuralNetworkEvaluator instance = new NeuralNetworkEvaluator(nn);
        
        double[][] neuronInputSums = new double[hiddenLayerSizes.length + 1][];
        double[][] neuronOutputs = new double[hiddenLayerSizes.length + 1][];
        neuronInputSums[0] = new double[]{0.0, 0.0};
        neuronInputSums[1] = new double[]{0.0, 0.0, 0.0};
        neuronOutputs[0] = new double[]{0.5, 0.5};
        neuronOutputs[1] = new double[]{0.5, 0.5, 0.5};
        NeuralNetworkResponse expResult = new NeuralNetworkResponse(neuronInputSums, neuronOutputs);
        
        NeuralNetworkResponse result = instance.evaluate(input);
        assertArraysEqual(expResult.neuronsInputSums, result.neuronsInputSums);
        assertArraysEqual(expResult.neuronsOutputs, result.neuronsOutputs);
    }
    
    @Test
    public void testEvaluate_NonZeroInputZeroNN_ReturnsZeroPointFive() {
        double[] input = {1, 0.5, 0.3};
        
        NeuralNetworkEvaluator instance = new NeuralNetworkEvaluator(nn);
        
        double[][] neuronInputSums = new double[hiddenLayerSizes.length + 1][];
        double[][] neuronOutputs = new double[hiddenLayerSizes.length + 1][];
        neuronInputSums[0] = new double[]{0.0, 0.0};
        neuronInputSums[1] = new double[]{0.0, 0.0, 0.0};
        neuronOutputs[0] = new double[]{0.5, 0.5};
        neuronOutputs[1] = new double[]{0.5, 0.5, 0.5};
        NeuralNetworkResponse expResult = new NeuralNetworkResponse(neuronInputSums, neuronOutputs);
        
        NeuralNetworkResponse result = instance.evaluate(input);
        assertArraysEqual(expResult.neuronsInputSums, result.neuronsInputSums);
        assertArraysEqual(expResult.neuronsOutputs, result.neuronsOutputs);
    }

    /**
     * Test of evaluateWithWeights method, of class NeuralNetworkEvaluator.
     */
    
    @Test(expected = NullPointerException.class)
    public void testEvaluateWithWeights_NullWeights_Throws() {
        double[] input = {1, 0.5, 0.3};
         
        NeuralNetworkWeights newWeights = null;

        NeuralNetworkEvaluator instance = new NeuralNetworkEvaluator(nn);
        
        instance.evaluateWithWeights(input, newWeights);
        
        fail("The test case must throw");
    }
    
    @Test(expected = NullPointerException.class)
    public void testEvaluateWithWeights_NullInputs_Throws() {
        double[] input = null;
        
        NeuralNetworkWeights newWeights = Mockito.mock(NeuralNetworkWeights.class);
        Mockito.when(newWeights.getBias(Matchers.anyInt(), Matchers.anyInt())).thenReturn(0.0);
        Mockito.when(newWeights.getWeight(Matchers.anyInt(), Matchers.anyInt(), Matchers.anyInt())).thenReturn(0.0);
        Mockito.doReturn(0.5).when(newWeights).getBias(0, 1);
        Mockito.doReturn(5.9).when(newWeights).getBias(1, 1);
        Mockito.doReturn(2.1).when(newWeights).getWeight(0, 0, 0);
        Mockito.doReturn(4.0).when(newWeights).getWeight(1, 1, 1);

        NeuralNetworkEvaluator instance = new NeuralNetworkEvaluator(nn);
        
        instance.evaluateWithWeights(input, newWeights);

        fail("The test case must throw");
    }
    
    @Test
    public void testEvaluateWithWeights_ValidWeights_CorrectResult() {
        double[] input = {1, 0.5, 0.3};
        
        NeuralNetworkWeights newWeights = Mockito.mock(NeuralNetworkWeights.class);
        Mockito.when(newWeights.getBias(Matchers.anyInt(), Matchers.anyInt())).thenReturn(0.0);
        Mockito.when(newWeights.getWeight(Matchers.anyInt(), Matchers.anyInt(), Matchers.anyInt())).thenReturn(0.0);
        Mockito.doReturn(0.5).when(newWeights).getBias(0, 1);
        Mockito.doReturn(5.9).when(newWeights).getBias(1, 1);
        Mockito.doReturn(2.1).when(newWeights).getWeight(0, 0, 0);
        Mockito.doReturn(4.0).when(newWeights).getWeight(1, 1, 1);
        
        NeuralNetworkEvaluator instance = new NeuralNetworkEvaluator(nn);
        
        double[][] neuronInputSums = new double[hiddenLayerSizes.length + 1][];
        double[][] neuronOutputs = new double[hiddenLayerSizes.length + 1][];
        neuronInputSums[0] = new double[]{2.1*1, 0.5};
        neuronOutputs[0] = new double[]{ActivationFunctions.SIGMOID.valueAt(neuronInputSums[0][0]), ActivationFunctions.SIGMOID.valueAt(neuronInputSums[0][1])};
        neuronInputSums[1] = new double[]{0.0, neuronOutputs[0][1]*4.0 + 5.9, 0.0};
        neuronOutputs[1] = new double[]{0.5, ActivationFunctions.SIGMOID.valueAt(neuronInputSums[1][1]), 0.5};
        NeuralNetworkResponse expResult = new NeuralNetworkResponse(neuronInputSums, neuronOutputs);
        
        NeuralNetworkResponse result = instance.evaluateWithWeights(input, newWeights);
        
        assertArraysEqual(expResult.neuronsInputSums, result.neuronsInputSums);
        assertArraysEqual(expResult.neuronsOutputs, result.neuronsOutputs);
    }

    @Test(expected = NullPointerException.class)
    public void testGetOutput_NullInputs_Throws() {
        double[] input = null;
       
        NeuralNetworkEvaluator instance = new NeuralNetworkEvaluator(nn);
        
        instance.getOutput(input);

        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testGetOutput_WrongSizedInputs_Throws() {
        double[] input = {1, 3, 4, 5};
       
        NeuralNetworkEvaluator instance = new NeuralNetworkEvaluator(nn);
        
        instance.getOutput(input);

        fail("The test case must throw");
    }
    
    @Test
    public void testGetOutput_ValidWeights_CorrectResult() {
        double[] input = {1, 0.5, 0.3};
        
        Mockito.doReturn(0.5).when(nn).getBias(0, 1);
        Mockito.doReturn(5.9).when(nn).getBias(1, 1);
        Mockito.doReturn(2.1).when(nn).getWeight(0, 0, 0);
        Mockito.doReturn(4.0).when(nn).getWeight(1, 1, 1);
        
        double[][] neuronInputSums = new double[hiddenLayerSizes.length + 1][];
        double[][] neuronOutputs = new double[hiddenLayerSizes.length + 1][];
        neuronInputSums[0] = new double[]{2.1*1, 0.5};
        neuronOutputs[0] = new double[]{ActivationFunctions.SIGMOID.valueAt(neuronInputSums[0][0]), ActivationFunctions.SIGMOID.valueAt(neuronInputSums[0][1])};
        neuronInputSums[1] = new double[]{0.0, neuronOutputs[0][1]*4.0 + 5.9, 0.0};
        neuronOutputs[1] = new double[]{0.5, ActivationFunctions.SIGMOID.valueAt(neuronInputSums[1][1]), 0.5};
        
        NeuralNetworkEvaluator instance = new NeuralNetworkEvaluator(nn);
        
        double[] result = instance.getOutput(input);
        
        assertArrayEquals(neuronOutputs[1], result, DELTA);
    }
    
}
