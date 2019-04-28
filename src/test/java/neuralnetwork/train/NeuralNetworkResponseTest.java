package neuralnetwork.train;

import neuralnetwork.TestUtils;
import org.junit.After;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Before;

/**
 *
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkResponseTest {
    
    int nOutputs;
    int[] hiddenLayerSizes;
    double[][] neuronsInputSums;
    double[][] neuronsOutputs;
    
    @Before
    public void initTestCase() {
        nOutputs = 4;
        hiddenLayerSizes = new int[]{2, 5};
        neuronsInputSums = new double[][] {{3.1, 6}, {-4, 4.6, 13.1, 0, 5.4}, {0.5, 1, 34.2, -5}};
        neuronsOutputs = new double[][] {{45, 2}, {34, 4.5, 1.1, -6, 6.1}, {1.5, 1.7, -0.2, -5}};
    }
    
    @After
    public void cleanUpTestCase() {
        neuronsInputSums = null;
        neuronsOutputs = null;
    }
    
    public NeuralNetworkResponseTest() {
    }

    @Test(expected = NullPointerException.class)
    public void testConstructor_NullFirstArgument_Throw() {
        System.out.println("testConstructor_NullFirstArgument_Throw");

        new NeuralNetworkResponse(
                null, 
                neuronsOutputs
        );
    }
    
    @Test(expected = NullPointerException.class)
    public void testConstructor_NullSecondArgument_Throw() {
        System.out.println("testConstructor_NullSecondArgument_Throw");

        new NeuralNetworkResponse(
                neuronsInputSums, 
                null
        );
    }
    
    @Test
    public void testConstructor_NotNullSecondArgument_Ok() {
        System.out.println("testConstructor_NullSecondArgument_Throw");

        new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
    }
    
    /**
     * Test of getNeuronInputSum method, of class NeuralNetworkResponse.
     */
    @Test
    public void testGetNeuronInputSum_ValidArguments_ReturnCorrectValue() {
        System.out.println("testGetNeuronInputSum_ValidArguments_ReturnCorrectValue");

        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        
        );
        for (int i = 0; i < hiddenLayerSizes.length; i++) {
            for (int j = 0; j < hiddenLayerSizes[i]; j++) {
                double expected = neuronsInputSums[i][j];
                double result = instance.getNeuronInputSum(i, j);
                assertEquals(expected, result, TestUtils.DELTA);
            }
        }
        for (int i = 0; i < nOutputs; i++) {
            double expected = neuronsInputSums[hiddenLayerSizes.length][i];
            double result = instance.getNeuronInputSum(hiddenLayerSizes.length, i);
            assertEquals(expected, result, TestUtils.DELTA);
        }
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetNeuronInputSum_LayerNumOutOfRange_Throw() {
        System.out.println("testGetNeuronInputSum_LayerNumOutOfRange_Throw");

        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        
        instance.getNeuronInputSum(hiddenLayerSizes.length + 2, 0);
        
        fail("Test case must throw");
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetNeuronInputSum_LayerNumNegative_Throw() {
        System.out.println("testGetNeuronInputSum_LayerNumNegative_Throw");

        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        
        instance.getNeuronInputSum(-1, 0);
        
        fail("Test case must throw");
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetNeuronInputSum_NeuronNumOutOfRange_Throw() {
        System.out.println("testGetNeuronInputSum_NeuronNumOutOfRange_Throw");

        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        
        instance.getNeuronInputSum(hiddenLayerSizes.length, nOutputs);
        
        fail("Test case must throw");
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetNeuronInputSum_NeuronNumNegative_Throw() {
        System.out.println("testGetNeuronInputSum_NeuronNumNegative_Throw");

        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        
        instance.getNeuronInputSum(hiddenLayerSizes.length, -1);
        
        fail("Test case must throw");
    }

    /**
     * Test of getNeuronOutput method, of class NeuralNetworkResponse.
     */
    @Test
    public void testGetNeuronOutput_ValidArguments_ReturnCorrectValue() {
        System.out.println("testGetNeuronOutput_ValidArguments_ReturnCorrectValue");
         NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        for (int i = 0; i < hiddenLayerSizes.length; i++) {
            for (int j = 0; j < hiddenLayerSizes[i]; j++) {
                double expected = neuronsOutputs[i][j];
                double result = instance.getNeuronOutput(i, j);
                assertEquals(expected, result, TestUtils.DELTA);
            }
        }
        for (int i = 0; i < nOutputs; i++) {
            double expected = neuronsOutputs[hiddenLayerSizes.length][i];
            double result = instance.getNeuronOutput(hiddenLayerSizes.length, i);
            assertEquals(expected, result, TestUtils.DELTA);
        }
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetNeuronOutput_LayerNumOutOfRange_Throw() {
        System.out.println("testGetNeuronOutput_LayerNumOutOfRange_Throw");
        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        
        instance.getNeuronOutput(hiddenLayerSizes.length + 1, 0);
        
        fail("Test case must throw");
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetNeuronOutput_LayerNumNegative_Throw() {
        System.out.println("testGetNeuronOutput_LayerNumNegative_Throw");
        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        
        instance.getNeuronOutput(-1, 0);
        
        fail("Test case must throw");
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetNeuronOutput_NeuronNumOutOfRange_Throw() {
        System.out.println("testGetNeuronOutput_NeuronNumOutOfRange_Throw");
        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        
        instance.getNeuronOutput(hiddenLayerSizes.length, nOutputs);
        
        fail("Test case must throw");
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetNeuronOutput_NeuronNumNegative_Throw() {
        System.out.println("testGetNeuronOutput_NeuronNumNegative_Throw");
        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        
        instance.getNeuronOutput(0, -1);
        
        fail("Test case must throw");
    }

    /**
     * Test of getOutput method, of class NeuralNetworkResponse.
     */
    @Test
    public void testGetOutput_ValidArgument_ReturnCorrectValue() {
        System.out.println("testGetOutput_ValidArgument_ReturnCorrectValue");
        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        for (int i = 0; i < nOutputs; i++) {
            double expected = neuronsOutputs[hiddenLayerSizes.length][i];
            double result = instance.getOutput(i);
            assertEquals(expected, result, TestUtils.DELTA);
        }
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetOutput_ArgumentNegative_Throw() {
        System.out.println("testGetOutput_ArgumentNegative_Throw");
        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        
        instance.getOutput(-1);
        
        fail("Test case must throw");
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetOutput_ArgumentOutOfBound_Throw() {
        System.out.println("testGetOutput_ArgumentOutOfBound_Throw");
        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        
        instance.getOutput(nOutputs);
        
        fail("Test case must throw");
    }

    /**
     * Test of getOutputs method, of class NeuralNetworkResponse.
     */
    @Test
    public void testGetOutputs_Invoked_ReturnCorrectValues() {
        System.out.println("getOutputs");
        NeuralNetworkResponse instance = new NeuralNetworkResponse(
                neuronsInputSums, 
                neuronsOutputs
        );
        double[] expResult = neuronsOutputs[hiddenLayerSizes.length].clone();
        double[] result = instance.getOutputs();
        assertArrayEquals(expResult, result, TestUtils.DELTA);
    }
    
}
