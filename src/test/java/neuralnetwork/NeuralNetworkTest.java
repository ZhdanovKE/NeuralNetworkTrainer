package neuralnetwork;

import java.util.Arrays;
import org.junit.Assert;
import org.junit.Test;

import static neuralnetwork.TestUtils.*;
import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;


/**
 *
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkTest {
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructor_ZeroNumInputs_Throws() {
        int nInputs = 0;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = 4;
        
        new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructor_NegativeNumInputs_Throws() {
        int nInputs = -2;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = 4;
        
        new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructor_ZeroNumOutputs_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = 0;
        
        new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructor_NegativeNumOutputs_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = -4;
        
        new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        Assert.fail();
    }
    
    @Test(expected = NullPointerException.class)
    public void testConstructor_NullHiddenLayerSizes_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = null;
        int nOutputs = 4;
        
        new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructor_EmptyHiddenLayerSizes_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {};
        int nOutputs = 4;
        
        new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructor_HiddenLayerSizesZero_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 0};
        int nOutputs = 4;
        
        new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructor_HiddenLayerSizesNegative_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, -3};
        int nOutputs = 4;
        
        new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        Assert.fail();
    }
    
    @Test
    public void testConstructor_ValidArguments_CreatesValidObjectWithZeroWeights() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        Assert.assertEquals("Number of inputs changed", nInputs, nn.getNumberInputs());
        Assert.assertEquals("Number of hidden layers changed", hiddenLayerSizes.length, nn.getNumberHiddenLayers());
        Assert.assertArrayEquals("Hidden layer size changed", hiddenLayerSizes, nn.getHiddenLayerSizes());
        Assert.assertEquals("Number of outputs changed", nOutputs, nn.getNumberOutputs());
        
        double[][][] weights = extractNNWeights(nn);
        
        for (int i = 0; i < hiddenLayerSizes[0]; i++) {
            Assert.assertEquals("Weights are inconsistent (inputs)", nInputs, weights[0][i].length);
        }
        
        for (int i = 1; i < hiddenLayerSizes.length; i++) {
            for (int j = 0; j < hiddenLayerSizes[i]; j++) {
                Assert.assertEquals("Weights are inconsistent", hiddenLayerSizes[i - 1], weights[i][j].length);
            }
        }
        
        for (double[] weight : weights[weights.length - 1]) {
            Assert.assertEquals("Weights are inconsistent (outputs)", hiddenLayerSizes[hiddenLayerSizes.length - 1], weight.length);
        }
        
        for (double[][] twoDarray : weights) {
            for (double[] oneDarray : twoDarray) {
                for (double elem : oneDarray) {
                    Assert.assertEquals(elem, 0.0, DELTA);
                }
            }
        }
        double[][] biases = extractNNBiases(nn);
        for (double[] oneDarray : biases) {
            for (double elem : oneDarray) {
                Assert.assertEquals(elem, 0.0, DELTA);
            }
        }
    }
    
    @Test()
    public void testCopyConstructor_ValidArgument_CopyIsSameAsArgument() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        NeuralNetwork nnCopy = new NeuralNetwork(nn);
    
        Assert.assertEquals("Numbers of inputs are different", nnCopy.getNumberInputs(), nn.getNumberInputs());
        Assert.assertEquals("Numbers of outputs are different", nnCopy.getNumberOutputs(), nnCopy.getNumberOutputs());
        Assert.assertArrayEquals("Numbers of hidden layers are different", nnCopy.getHiddenLayerSizes(), nnCopy.getHiddenLayerSizes());
        Assert.assertEquals("Activation functions are different", nn.getActivationFunction(), nnCopy.getActivationFunction());
        double[][][] nnWeights = extractNNWeights(nn);
        double[][][] nnCopyWeights = extractNNWeights(nnCopy);
        for (int i = 0; i < nnWeights.length; i++) {
            for (int j = 0; j < nnWeights[i].length; j++) {
                Assert.assertArrayEquals(nnWeights[i][j], nnCopyWeights[i][j], DELTA);
            }
        }
        double[][] nnBiases = extractNNBiases(nn);
        double[][] nnCopyBiases = extractNNBiases(nnCopy);
        for (int i = 0; i < nnBiases.length; i++) {
            Assert.assertArrayEquals(nnBiases[i], nnCopyBiases[i], DELTA);
        }
    }
    
    @Test(expected = NullPointerException.class)
    public void testCopyConstructor_NullArgument_Throws() {
        new NeuralNetwork(null);
    
        Assert.fail();
    }
    
    @Test
    public void testGetNumberInputs_OneNumInputs_OK() {
        int nInputs = 1;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.assertEquals("Number of inputs", nInputs, nn.getNumberInputs());
    }
    
    @Test
    public void testGetNumberInputs_PositiveNumInputs_OK() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.assertEquals("Number of inputs", nInputs, nn.getNumberInputs());
    }
    
    @Test
    public void testGetNumberOutputs_OneNumOutputs_OK() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.assertEquals("Number of outputs", nOutputs, nn.getNumberOutputs());
    }
    
    @Test
    public void testGetNumberOutputs_PositiveNumOutputs_OK() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.assertEquals("Number of outputs", nOutputs, nn.getNumberOutputs());
    }
    
    @Test
    public void testGetNumberHiddenLayers_SingleHiddenLayer_OK() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.assertEquals("Number of hidden layers", hiddenLayerSizes.length, nn.getNumberHiddenLayers());
    }
    
    @Test
    public void testGetNumberHiddenLayers_PositiveNumberHiddenLayers_OK() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {3, 5};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.assertEquals("Number of hidden layers", hiddenLayerSizes.length, nn.getNumberHiddenLayers());
    }
    
    @Test
    public void testGetHiddenLayerSize_SingleHiddenLayerSizeOne_OK() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {1};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        for (int i = 0; i < hiddenLayerSizes.length; i++) {
            Assert.assertEquals("Hidden layer " + i, hiddenLayerSizes[i], nn.getHiddenLayerSize(i));
        }
    }
    
    @Test
    public void testGetHiddenLayerSize_SingleHiddenLayerSizePositive_OK() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {5};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        for (int i = 0; i < hiddenLayerSizes.length; i++) {
            Assert.assertEquals("Hidden layer " + i, hiddenLayerSizes[i], nn.getHiddenLayerSize(i));
        }
    }
    
    @Test
    public void testGetHiddenLayerSize_TwoHiddenLayersSizePositive_OK() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {5, 2};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        for (int i = 0; i < hiddenLayerSizes.length; i++) {
            Assert.assertEquals("Hidden layer " + i, hiddenLayerSizes[i], nn.getHiddenLayerSize(i));
        }
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetHiddenLayerSize_NegativeArgument_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.getHiddenLayerSize(-2);
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetHiddenLayerSize_ArgumentGreaterThanNumberHiddenLayers_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.getHiddenLayerSize(2);
    }
    
    @Test
    public void testGetHiddenLayerSizes_Invoked_ReturnsEqualArray() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        int[] hiddenLayerSizesActual = nn.getHiddenLayerSizes();
        
        Assert.assertArrayEquals("Hidden layer sizes are different", hiddenLayerSizes, hiddenLayerSizesActual);
    }
    
    @Test
    public void testGetHiddenLayerSizes_ReturnedArrayChanged_InternalArrayNotChanged() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        int[] hiddenLayerSizesReturned = nn.getHiddenLayerSizes();
        hiddenLayerSizesReturned[0] = hiddenLayerSizesReturned[0] + 2;
        
        int[] hiddenLayerSizesActual = nn.getHiddenLayerSizes();
        
        Assert.assertArrayEquals("Hidden layer size changed", hiddenLayerSizes, hiddenLayerSizesActual);
    }
    
    @Test
    public void testGetActivationFunction_NotSet_SigmoidFunction() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.assertEquals("Default activation function is not Sigmoid", nn.getActivationFunction(), ActivationFunctions.SIGMOID);
    }
    
    @Test
    public void testSetActivationFunction_SetTanFunction_ReturnsSameTanFunction() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        ActivationFunction fcn = ActivationFunctions.TAN;
        nn.setActivationFunction(fcn);
        
        Assert.assertSame("Activation function is not the same", fcn, nn.getActivationFunction());
    }
    
    @Test(expected = NullPointerException.class)
    public void testSetActivationFunction_NullArgument_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        nn.setActivationFunction(null);
        
        Assert.fail();
    }
    
    @Test
    public void testGetBias_NotSet_ReturnsZeroForEveryIndex() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        for (int i = 0; i < nn.getNumberHiddenLayers(); i++) {
            for (int j = 0; j < nn.getHiddenLayerSize(i); j++) {
                Assert.assertEquals(0.0, nn.getBias(i, j), DELTA);
            }
        }
        
        for (int j = 0; j < nn.getNumberOutputs(); j++) {
            Assert.assertEquals(0.0, nn.getBias(nn.getNumberHiddenLayers(), j), DELTA);
        }
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetBias_NegativeFirstArgument_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.getBias(-1, 1);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetBias_NegativeSecondArgument_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.getBias(0, -1);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetBias_FirstArgumentGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.getBias(3, 1);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetBias_SecondArgumentGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.getBias(0, 2);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetBias_FirstIndexNegative_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.4;
        int layerNum = -1;
        int neuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        nn.setBias(layerNum, neuronNum, toSet);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetBias_FirstIndexGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.4;
        int layerNum = 3;
        int neuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        nn.setBias(layerNum, neuronNum, toSet);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetBias_SecondIndexNegative_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.4;
        int layerNum = 1;
        int neuronNum = -1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        nn.setBias(layerNum, neuronNum, toSet);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetBias_SecondIndexGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.4;
        int layerNum = 1;
        int neuronNum = 3;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        nn.setBias(layerNum, neuronNum, toSet);
        
        Assert.fail();
    }
    
    @Test
    public void testSetBias_SetValue_ReturnsSetValue() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double expected = 45.4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        nn.setBias(0, 1, expected);
        
        Assert.assertEquals(expected, nn.getBias(0, 1), DELTA);
    }
    
    @Test
    public void testSetBias_SetValue_OtherValuesNotChanged() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.4;
        int layerNum = 0;
        int neuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        double[][] expectedBiases = extractNNBiases(nn);
        expectedBiases[layerNum][neuronNum] = toSet;
        double[][][] expectedWeights = extractNNWeights(nn);
        
        nn.setBias(layerNum, neuronNum, toSet);
        
        double[][] actualBiases = extractNNBiases(nn);
        double[][][] actualWeights = extractNNWeights(nn);
        if (!Arrays.deepEquals(actualBiases, expectedBiases)) {
            Assert.fail();
        }
        if (!Arrays.deepEquals(actualWeights, expectedWeights)) {
            Assert.fail();
        }
    }
    
    @Test
    public void testGetWeight_NotSet_ReturnsZeroForEveryIndex() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        double[][][] actualWeights = extractNNWeights(nn);
        
        for (double[][] actualWeight : actualWeights) {
            for (double[] weights : actualWeight) {
                for (double weight : weights) {
                    Assert.assertEquals(0.0, weight, DELTA);
                }
            }
        }
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetWeight_FirstIndexNegative_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        int layerNum = -1;
        int neuronNum = 2;
        int prevNeuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        nn.getWeight(layerNum, prevNeuronNum, neuronNum);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetWeight_FirstIndexGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        int layerNum = 3;
        int neuronNum = 2;
        int prevNeuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        nn.getWeight(layerNum, prevNeuronNum, neuronNum);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetWeight_SecondIndexNegative_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        int layerNum = 1;
        int neuronNum = -2;
        int prevNeuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        nn.getWeight(layerNum, prevNeuronNum, neuronNum);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetWeight_SecondIndexGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        int layerNum = 1;
        int neuronNum = 3;
        int prevNeuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        nn.getWeight(layerNum, prevNeuronNum, neuronNum);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetWeight_ThirdIndexNegative_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        int layerNum = 1;
        int neuronNum = 2;
        int prevNeuronNum = -1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        nn.getWeight(layerNum, prevNeuronNum, neuronNum);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetWeight_ThirdIndexGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        int layerNum = 1;
        int neuronNum = 2;
        int prevNeuronNum = 2;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        nn.getWeight(layerNum, prevNeuronNum, neuronNum);
        
        Assert.fail();
    }
    
    @Test
    public void testSetWeight_SetValue_ReturnsSetValue() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.7;
        int layerNum = 1;
        int neuronNum = 2;
        int prevNeuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
    
        nn.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
        Assert.assertEquals(toSet, nn.getWeight(layerNum, prevNeuronNum, neuronNum), DELTA);
    }
    
    @Test
    public void testSetWeight_SetValue_OtherValuesNotChanged() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.7;
        int layerNum = 1;
        int neuronNum = 2;
        int prevNeuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        double[][][] expectedWeights = extractNNWeights(nn);
        expectedWeights[layerNum][neuronNum][prevNeuronNum] = toSet;
        double[][] expectedBiases = extractNNBiases(nn);
        
        nn.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        double[][][] actualWeights = extractNNWeights(nn);
        double[][] actualBiases = extractNNBiases(nn);
        
        if (!Arrays.deepEquals(actualWeights, expectedWeights)) {
            Assert.fail("Weights are different");
        }
        
        if (!Arrays.deepEquals(actualBiases, expectedBiases)) {
            Assert.fail("Biases are different");
        }
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetWeight_FirstIndexNegative_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.7;
        int layerNum = -1;
        int neuronNum = 2;
        int prevNeuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetWeight_FirstIndexGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.7;
        int layerNum = 3;
        int neuronNum = 2;
        int prevNeuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetWeight_SecondIndexNegative_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.7;
        int layerNum = 1;
        int neuronNum = -2;
        int prevNeuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetWeight_SecondIndexGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.7;
        int layerNum = 1;
        int neuronNum = 3;
        int prevNeuronNum = 1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetWeight_ThirdIndexNegative_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.7;
        int layerNum = 1;
        int neuronNum = 2;
        int prevNeuronNum = -1;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testSetWeight_ThirdIndexGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.7;
        int layerNum = 1;
        int neuronNum = 2;
        int prevNeuronNum = 2;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
        Assert.fail();
    }
    
    @Test
    public void testSetWeights_ZeroValueProvidedForNonZeroNN_AllWeightsAreZero() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 0;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        nn.setWeight(0, 1, 0, -87);
        
        nn.setWeights(toSet);
        
        double[][][] weights = extractNNWeights(nn);
        for (double[][] weightsForLayer : weights) {
            for (double[] weightsForNeuron : weightsForLayer) {
                for (double weight : weightsForNeuron) {
                    Assert.assertEquals(toSet, weight, DELTA);
                }
            }
        }
    }
    
    @Test
    public void testSetWeights_NonZeroValueProvided_AllWeightsAreCorrectNonZero() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 43.1122;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.setWeights(toSet);
        
        double[][][] weights = extractNNWeights(nn);
        for (double[][] weightsForLayer : weights) {
            for (double[] weightsForNeuron : weightsForLayer) {
                for (double weight : weightsForNeuron) {
                    Assert.assertEquals(toSet, weight, DELTA);
                }
            }
        }
    }
    
    @Test
    public void testSetWeights_NonZeroValueProvided_BiasesNotChanged() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 43.1122;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.setWeights(toSet);
        
        double[][] biases = extractNNBiases(nn);
        for (double[] biasesForLayer : biases) {
            for (double bias : biasesForLayer) {
                Assert.assertEquals(0, bias, DELTA);
            }
        }
    }
    
    @Test
    public void testSetBiases_ZeroValueProvidedForNonZeroNN_AllBiasesAreZero() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 0;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        nn.setBias(0, 1, -87);
        
        nn.setBiases(toSet);
        
        double[][] biases = extractNNBiases(nn);
        for (double[] biasesForLayer : biases) {
            for (double bias : biasesForLayer) {
                Assert.assertEquals(toSet, bias, DELTA);
            }
        }
    }
    
    @Test
    public void testSetWeights_NonZeroValueProvided_AllBiasesAreCorrectNonZero() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 43.1122;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.setBiases(toSet);
        
        double[][] biases = extractNNBiases(nn);
        for (double[] biasesForLayer : biases) {
            for (double bias : biasesForLayer) {
                Assert.assertEquals(toSet, bias, DELTA);
            }
        }
    }
    
    @Test
    public void testSetWeights_NonZeroValueProvided_WeightsNotChanged() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 43.1122;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        nn.setBiases(toSet);
        
        double[][][] weights = extractNNWeights(nn);
        for (double[][] weightsForLayer : weights) {
            for (double[] weightsForNeuron : weightsForLayer) {
                for (double weight : weightsForNeuron) {
                    Assert.assertEquals(0, weight, DELTA);
                }
            }
        }
    }
    
    @Test
    public void testToString_Invoked_ReturnCorrectStructureInString() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        String expected = "(2, 2, 3, 4)";
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        String actual = nn.toString();
        
        Assert.assertEquals("String format is different", expected, actual);
    }
    
    @Test
    public void testSerialization_WriteToObjectStream_ReadEqualStructureAndWeightsObject() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, 
                nOutputs);
        
        ByteArrayOutputStream out = new ByteArrayOutputStream();
        try (ObjectOutputStream oos = new ObjectOutputStream(out)) {
            oos.writeObject(nn);
        }
        catch(IOException e) {
            Assert.fail("Exception caught: " + e.toString());
        }
        
        NeuralNetwork readNN = null;
        ByteArrayInputStream in = new ByteArrayInputStream(out.toByteArray());
        try (ObjectInputStream ois = new ObjectInputStream(in)) {
            readNN = (NeuralNetwork)ois.readObject();
        }
        catch(IOException e) {
            Assert.fail("Exception caught: " + e.toString());
        }
        catch(ClassNotFoundException e) {
            Assert.fail("Class not found while reading: " + e.toString());
        }
        
        Assert.assertNotSame(nn, readNN);
        assertNNEquals(nn, readNN);
        
        readNN.setWeight(0, 0, 1, 10.0);
        assertNNNotEquals(nn, readNN);
    }
    
}
