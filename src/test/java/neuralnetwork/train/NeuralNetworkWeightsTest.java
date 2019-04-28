package neuralnetwork.train;

import neuralnetwork.NeuralNetwork;
import java.util.Arrays;
import org.junit.Assert;
import org.junit.Test;

import static neuralnetwork.TestUtils.*;

/**
 *
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkWeightsTest {
    
    @Test
    public void testCopyConstructor_ValidArgument_CreatesEqualObject() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weightsSrc = new NeuralNetworkWeights(nInputs, 
            hiddenLayerSizes, nOutputs);
        
        NeuralNetworkWeights weightsCopy = new NeuralNetworkWeights(weightsSrc);
        
        assertArraysEqual(weightsSrc.weights, weightsCopy.weights);
        assertArraysEqual(weightsSrc.biases, weightsCopy.biases);
    }
    
    @Test(expected = NullPointerException.class)
    public void testCopyConstructor_NullArgument_Throws() {
        new NeuralNetworkWeights(null);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorThreeArgs_ZeroNumInputs_Throws() {
        int nInputs = 0;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = 4;
        
        new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
    
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorThreeArgs_NegativeNumInputs_Throws() {
        int nInputs = -2;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = 4;
        
        new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorThreeArgs_ZeroNumOutputs_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = 0;
        
        new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorThreeArgs_NegativeNumOutputs_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {3, 1};
        int nOutputs = -4;
        
        new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.fail();
    }
    
    @Test(expected = NullPointerException.class)
    public void testConstructorThreeArgs_NullHiddenLayerSizes_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = null;
        int nOutputs = 4;
        
        new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorThreeArgs_EmptyHiddenLayerSizes_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {};
        int nOutputs = 4;
        
        new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorThreeArgs_HiddenLayerSizesZero_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 0};
        int nOutputs = 4;
        
        new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testConstructorThreeArgs_HiddenLayerSizesNegative_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, -3};
        int nOutputs = 4;
        
        new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.fail();
    }
    
    @Test
    public void testConstructor_ValidArguments_CreatesValidObjectWithZeroWeights() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        Assert.assertEquals("Number of hidden layers is not saved", hiddenLayerSizes.length + 1, weights.weights.length);
        Assert.assertEquals("Number of hidden layers is not saved", hiddenLayerSizes.length + 1, weights.biases.length);
        
        for (int i = 0; i < hiddenLayerSizes.length; i++) {
            Assert.assertEquals("Hidden layer " + i + " size is not saved", hiddenLayerSizes[i], weights.weights[i].length);
            Assert.assertEquals("Hidden layer " + i + " size is not saved", hiddenLayerSizes[i], weights.biases[i].length);
        }
        
        for (double[] weightSubArray : weights.weights[0]) {
            Assert.assertEquals("Number of inputs is not saved", nInputs, weightSubArray.length);
        }
        
        for (int i = 1; i < hiddenLayerSizes.length; i++) {
            for (int j = 0; j < hiddenLayerSizes[i]; j++) {
                Assert.assertEquals("Weights are inconsistent", hiddenLayerSizes[i - 1], weights.weights[i][j].length);
            }
        }
        
        for (double[] weightSubArray : weights.weights[weights.weights.length - 1]) {
            Assert.assertEquals("Weights are inconsistent", hiddenLayerSizes[hiddenLayerSizes.length - 1], weightSubArray.length);
        }
        Assert.assertEquals("Number of outputs is not saved", nOutputs, weights.weights[weights.weights.length - 1].length);
        Assert.assertEquals("Number of outputs is not saved", nOutputs, weights.biases[weights.biases.length - 1].length);


        for (double[][] twoDarray : weights.weights) {
            for (double[] oneDarray : twoDarray) {
                for (double elem : oneDarray) {
                    Assert.assertEquals(elem, 0.0, DELTA);
                }
            }
        }
        for (double[] oneDarray : weights.biases) {
            for (double elem : oneDarray) {
                Assert.assertEquals(elem, 0.0, DELTA);
            }
        }
    }
    
    @Test(expected = NullPointerException.class)
    public void testNewOf_NullArgument_Throws() {
        NeuralNetworkWeights.newOf(null);
        
        Assert.fail();
    }
    
    @Test
    public void testNewOf_ValidArgument_WeightsAndBiasesEqualToArgumentWeightsAndBiases() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        double[][][] nnWeights = extractNNWeights(nn);
        double[][] nnBiases = extractNNBiases(nn);
        
        NeuralNetworkWeights weights = NeuralNetworkWeights.newOf(nn);
        
        assertArraysEqual(nnBiases, weights.biases);
        assertArraysEqual(nnWeights, weights.weights);
    }
    
    @Test
    public void testNewOf_ValidArgument_WeightsAndBiasesCannotChangeArgument() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        double[][][] nnWeightsExpected = extractNNWeights(nn);
        double[][] nnBiasesExpected = extractNNBiases(nn);
        
        NeuralNetworkWeights weights = NeuralNetworkWeights.newOf(nn);
        weights.biases[1][2] = weights.biases[1][2] + 2;
        weights.weights[0][1][2] = weights.weights[0][1][2] + 3;
        
        double[][][] nnWeightsActual = extractNNWeights(nn);
        double[][] nnBiasesActual = extractNNBiases(nn);
        
        assertArraysEqual(nnBiasesExpected, nnBiasesActual);
        assertArraysEqual(nnWeightsExpected, nnWeightsActual);
    }
    
    
    @Test(expected = NullPointerException.class)
    public void testApplyToNeuralNetwork_NullArgument_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
 
        weights.applyToNeuralNetwork(null);
        
        Assert.fail();
    }
    
    @Test
    public void testApplyToNeuralNetwork_ValidArgument_NNWeightsAndBiasesChangedAccordingly() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        weights.setWeight(0, 1, 1, 45.7);
        weights.setBias(2, 1, 34.9);
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.applyToNeuralNetwork(nn);
        
        assertArraysEqual(weights.weights, extractNNWeights(nn));
        assertArraysEqual(weights.biases, extractNNBiases(nn));
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testApplyToNeuralNetwork_NNOfGreaterInputSize_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        int nInputsNN = nInputs + 1;
        NeuralNetwork nn = new NeuralNetwork(nInputsNN, hiddenLayerSizes, nOutputs);
        
        weights.applyToNeuralNetwork(nn);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testApplyToNeuralNetwork_NNOfGreaterHiddenLayersNumber_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        int[] hiddenLayerSizesNN = {2, 3, 5};
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizesNN, nOutputs);
        
        weights.applyToNeuralNetwork(nn);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testApplyToNeuralNetwork_NNOfGreaterHiddenLayerSize_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        int[] hiddenLayerSizesNN = {2, 4};
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizesNN, nOutputs);
        
        weights.applyToNeuralNetwork(nn);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testApplyToNeuralNetwork_NNOfGreaterOutputNumber_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        int nOutputsNN = 5;
        NeuralNetwork nn = new NeuralNetwork(nInputs, hiddenLayerSizes, nOutputsNN);
        
        weights.applyToNeuralNetwork(nn);
        
        Assert.fail();
    }
    
    @Test(expected = NullPointerException.class)
    public void testAdd_NullArgument_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.add(null);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testAdd_ArgumentOfWrongInputSizes_Throws() {
        int nInputsLhs = 3;
        int nInputsRhs = 4;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputsLhs, hiddenLayerSizes, nOutputs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputsRhs, hiddenLayerSizes, nOutputs);
        
        weightsLhs.add(weightsRhs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testAdd_ArgumentOfWrongHiddenLayersNumber_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizesLhs = {2, 3};
        int[] hiddenLayerSizesRhs = {2, 3, 5};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesLhs, nOutputs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesRhs, nOutputs);
        
        weightsLhs.add(weightsRhs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testAdd_ArgumentOfWrongHiddenLayersSizes_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizesLhs = {2, 3};
        int[] hiddenLayerSizesRhs = {2, 4};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesLhs, nOutputs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesRhs, nOutputs);
        
        weightsLhs.add(weightsRhs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testAdd_ArgumentOfWrongOutputSizes_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputsLhs = 4;
        int nOutputsRhs = 5;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputsLhs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputsRhs);
        
        weightsLhs.add(weightsRhs);
        
        Assert.fail();
    }
    
    @Test
    public void testAdd_ArgumentOfSameSizes_LhsIsSum() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        weightsLhs.setWeight(0, 0, 0, 12.5);
        weightsLhs.setWeight(0, 1, 1, -45.7);
        weightsLhs.setWeight(2, 1, 2, 3.9);
        weightsLhs.setBias(0, 0, -4.4);
        weightsLhs.setBias(2, 2, 34444.2);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        weightsRhs.setWeight(0, 0, 0, 45.3);
        weightsRhs.setWeight(0, 1, 1, 11);
        weightsRhs.setWeight(1, 1, 1, 33.2);
        weightsRhs.setBias(0, 0, -3.4);
        weightsRhs.setBias(1, 2, 45454);
        
        double[][][] lhsWeights = copyWeights(weightsLhs.weights);
        double[][] lhsBiases = copyBiases(weightsLhs.biases);
        
        weightsLhs.add(weightsRhs);
        
        Assert.assertEquals("The size of the result is different", lhsWeights.length, weightsLhs.weights.length);
        for (int i = 0; i < lhsWeights.length; i++) {
            Assert.assertEquals("The size of the result is different", lhsWeights[i].length, weightsLhs.weights[i].length);
            for (int j = 0; j < lhsWeights[i].length; j++) {
                Assert.assertEquals("The size of the result is different", lhsWeights[i][j].length, weightsLhs.weights[i][j].length);
                for (int k = 0; k < lhsWeights[i][j].length; k++) {
                    Assert.assertEquals(lhsWeights[i][j][k] + weightsRhs.weights[i][j][k], weightsLhs.weights[i][j][k], DELTA);
                }
            }
        }
        Assert.assertEquals("The size of the result is different", lhsBiases.length, weightsLhs.biases.length);
        for (int i = 0; i < lhsBiases.length; i++) {
            Assert.assertEquals("The size of the result is different", lhsBiases[i].length, weightsLhs.biases[i].length);
            for (int j = 0; j < lhsWeights[i].length; j++) {
                Assert.assertEquals(lhsBiases[i][j] + weightsRhs.biases[i][j], weightsLhs.biases[i][j], DELTA);
            }
        }
    }
    
    @Test(expected = NullPointerException.class)
    public void testSubtract_NullArgument_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.subtract(null);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testSubtract_ArgumentOfWrongInputSizes_Throws() {
        int nInputsLhs = 3;
        int nInputsRhs = 4;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputsLhs, hiddenLayerSizes, nOutputs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputsRhs, hiddenLayerSizes, nOutputs);
        
        weightsLhs.subtract(weightsRhs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testSubtract_ArgumentOfWrongHiddenLayersNumber_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizesLhs = {2, 3};
        int[] hiddenLayerSizesRhs = {2, 3, 5};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesLhs, nOutputs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesRhs, nOutputs);
        
        weightsLhs.subtract(weightsRhs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testSubtract_ArgumentOfWrongHiddenLayersSizes_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizesLhs = {2, 3};
        int[] hiddenLayerSizesRhs = {2, 4};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesLhs, nOutputs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesRhs, nOutputs);
        
        weightsLhs.subtract(weightsRhs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testSubtract_ArgumentOfWrongOutputSizes_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputsLhs = 4;
        int nOutputsRhs = 5;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputsLhs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputsRhs);
        
        weightsLhs.subtract(weightsRhs);
        
        Assert.fail();
    }
    
    @Test
    public void testSubtract_ArgumentOfSameSizes_LhsIsDifference() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        weightsLhs.setWeight(0, 0, 0, 12.5);
        weightsLhs.setWeight(0, 1, 1, -45.7);
        weightsLhs.setWeight(2, 1, 2, 3.9);
        weightsLhs.setBias(0, 0, -4.4);
        weightsLhs.setBias(2, 2, 34444.2);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        weightsRhs.setWeight(0, 0, 0, 45.3);
        weightsRhs.setWeight(0, 1, 1, 11);
        weightsRhs.setWeight(1, 1, 1, 33.2);
        weightsRhs.setBias(0, 0, -3.4);
        weightsRhs.setBias(1, 2, 45454);
        
        double[][][] lhsWeights = copyWeights(weightsLhs.weights);
        double[][] lhsBiases = copyBiases(weightsLhs.biases);
        
        weightsLhs.subtract(weightsRhs);
        
        Assert.assertEquals("The size of the result is different", lhsWeights.length, weightsLhs.weights.length);
        for (int i = 0; i < lhsWeights.length; i++) {
            Assert.assertEquals("The size of the result is different", lhsWeights[i].length, weightsLhs.weights[i].length);
            for (int j = 0; j < lhsWeights[i].length; j++) {
                Assert.assertEquals("The size of the result is different", lhsWeights[i][j].length, weightsLhs.weights[i][j].length);
                for (int k = 0; k < lhsWeights[i][j].length; k++) {
                    Assert.assertEquals(lhsWeights[i][j][k] - weightsRhs.weights[i][j][k], weightsLhs.weights[i][j][k], DELTA);
                }
            }
        }
        Assert.assertEquals("The size of the result is different", lhsBiases.length, weightsLhs.biases.length);
        for (int i = 0; i < lhsBiases.length; i++) {
            Assert.assertEquals("The size of the result is different", lhsBiases[i].length, weightsLhs.biases[i].length);
            for (int j = 0; j < lhsWeights[i].length; j++) {
                Assert.assertEquals(lhsBiases[i][j] - weightsRhs.biases[i][j], weightsLhs.biases[i][j], DELTA);
            }
        }
    }
    
    @Test
    public void testSetTo_InvokeWithValue_AllWeightsAndBiasesAreSetToValue() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        int[][] weightInsertIdxs = {{0, 0, 0}, {0, 1, 1}, {2, 1, 2}};
        double[] weightInsertVals = {12.5, -45.7, 3.9};
        int[][] biasInsertIdxs = {{0, 0}, {2, 2}};
        double[] biasInsertVals = {-4.4, 34442.2};
        for (int i = 0; i < weightInsertIdxs.length; i++) {
            weights.setWeight(weightInsertIdxs[i][0], weightInsertIdxs[i][1], weightInsertIdxs[i][2], weightInsertVals[i]);
        }
        for (int i = 0; i < biasInsertIdxs.length; i++) {
            weights.setBias(biasInsertIdxs[i][0], biasInsertIdxs[i][1], biasInsertVals[i]);
        }
        double toSet = 45.6;
        
        double[][][] weightsExpected = copyWeights(weights.weights);
        double[][] biasesExpected = copyBiases(weights.biases);
        for (double[][] weightsExpectedForLayer : weightsExpected) {
            for (double[] weightsExpectedForNeuron : weightsExpectedForLayer) {
                for (int k = 0; k < weightsExpectedForNeuron.length; k++) {
                    weightsExpectedForNeuron[k] = toSet;
                }
            }
        }
        
        for (double[] biasesExpectedForLayer : biasesExpected) {
            for (int j = 0; j < biasesExpectedForLayer.length; j++) {
                biasesExpectedForLayer[j] = toSet;
            }
        }
        
        weights.setTo(toSet);
        
        assertArraysEqual(weightsExpected, weights.weights);
        assertArraysEqual(biasesExpected, weights.biases);
    }
    
    @Test
    public void testSetTo_InvokeWithValue_SizeStaysSame() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        int[][] weightInsertIdxs = {{0, 0, 0}, {0, 1, 1}, {2, 1, 2}};
        double[] weightInsertVals = {12.5, -45.7, 3.9};
        int[][] biasInsertIdxs = {{0, 0}, {2, 2}};
        double[] biasInsertVals = {-4.4, 34442.2};
        for (int i = 0; i < weightInsertIdxs.length; i++) {
            weights.setWeight(weightInsertIdxs[i][0], weightInsertIdxs[i][1], weightInsertIdxs[i][2], weightInsertVals[i]);
            
        }
        for (int i = 0; i < biasInsertIdxs.length; i++) {
            weights.setBias(biasInsertIdxs[i][0], biasInsertIdxs[i][1], biasInsertVals[i]);
        }
        
        double toSet = 45.6;
        
        double[][][] weightsExpected = copyWeights(weights.weights);
        double[][] biasesExpected = copyBiases(weights.biases);
        
        for (double[][] weightsExpectedForLayer : weightsExpected) {
            for (double[] weightsExpectedForNeuron : weightsExpectedForLayer) {
                for (int k = 0; k < weightsExpectedForNeuron.length; k++) {
                    weightsExpectedForNeuron[k] = toSet;
                }
            }
        }
        for (double[] biasesExpectedForLayer : biasesExpected) {
            for (int j = 0; j < biasesExpectedForLayer.length; j++) {
                biasesExpectedForLayer[j] = toSet;
            }
        }
        
        weights.setTo(toSet);
        
        if (!weightsSameSize(weightsExpected, weights.weights)) {
            Assert.fail("Weights are wrong size");
        }
        if (!biasesSameSize(biasesExpected, weights.biases)) {
            Assert.fail("Biases are wrong size");
        }
    }
            
    @Test
    public void testMultiply_Invoke_AllWeightsAndBiasesAreMultiplied() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        int[][] weightInsertIdxs = {{0, 0, 0}, {0, 1, 1}, {2, 1, 2}};
        double[] weightInsertVals = {12.5, -45.7, 3.9};
        int[][] biasInsertIdxs = {{0, 0}, {2, 2}};
        double[] biasInsertVals = {-4.4, 34442.2};
        for (int i = 0; i < weightInsertIdxs.length; i++) {
            weights.setWeight(weightInsertIdxs[i][0], weightInsertIdxs[i][1], weightInsertIdxs[i][2], weightInsertVals[i]);
        }
        for (int i = 0; i < biasInsertIdxs.length; i++) {
            weights.setBias(biasInsertIdxs[i][0], biasInsertIdxs[i][1], biasInsertVals[i]);
        }
        
        double factor = 45.6;
        
        double[][][] weightsExpected = copyWeights(weights.weights);
        double[][] biasesExpected = copyBiases(weights.biases);
        
        for (double[][] weightsExpectedForLayer : weightsExpected) {
            for (double[] weightsExpectedForNeuron : weightsExpectedForLayer) {
                for (int k = 0; k < weightsExpectedForNeuron.length; k++) {
                    weightsExpectedForNeuron[k] *= factor;
                }
            }
        }
        for (double[] biasesExpectedForLayer : biasesExpected) {
            for (int j = 0; j < biasesExpectedForLayer.length; j++) {
                biasesExpectedForLayer[j] *= factor;
            }
        }
        
        weights.multiply(factor);
        
        assertArraysEqual(weightsExpected, weights.weights);
        assertArraysEqual(biasesExpected, weights.biases);
    }
    
    @Test
    public void testMultiply_Invoke_SizeStaysSame() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        int[][] weightInsertIdxs = {{0, 0, 0}, {0, 1, 1}, {2, 1, 2}};
        double[] weightInsertVals = {12.5, -45.7, 3.9};
        int[][] biasInsertIdxs = {{0, 0}, {2, 2}};
        double[] biasInsertVals = {-4.4, 34442.2};
        for (int i = 0; i < weightInsertIdxs.length; i++) {
            weights.setWeight(weightInsertIdxs[i][0], weightInsertIdxs[i][1], weightInsertIdxs[i][2], weightInsertVals[i]);
            
        }
        for (int i = 0; i < biasInsertIdxs.length; i++) {
            weights.setBias(biasInsertIdxs[i][0], biasInsertIdxs[i][1], biasInsertVals[i]);
        }
        
        double factor = 45.6;
        
        double[][][] weightsExpected = copyWeights(weights.weights);
        double[][] biasesExpected = copyBiases(weights.biases);
        
        for (double[][] weightsExpectedForLayer : weightsExpected) {
            for (double[] weightsExpectedForNeuron : weightsExpectedForLayer) {
                for (int k = 0; k < weightsExpectedForNeuron.length; k++) {
                    weightsExpectedForNeuron[k] *= factor;
                }
            }
        }
        for (double[] biasesExpectedForLayer : biasesExpected) {
            for (int j = 0; j < biasesExpectedForLayer.length; j++) {
                biasesExpectedForLayer[j] *= factor;
            }
        }
        
        weights.multiply(factor);
        
        if (!weightsSameSize(weightsExpected, weights.weights)) {
            Assert.fail("Weights are wrong size");
        }
        if (!biasesSameSize(biasesExpected, weights.biases)) {
            Assert.fail("Biases are wrong size");
        }
    }
    
    @Test(expected = NullPointerException.class)
    public void testDot_NullArgument_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.dot(null);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testDot_ArgumentOfWrongInputSizes_Throws() {
        int nInputsLhs = 3;
        int nInputsRhs = 4;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputsLhs, hiddenLayerSizes, nOutputs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputsRhs, hiddenLayerSizes, nOutputs);
        
        weightsLhs.dot(weightsRhs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testDot_ArgumentOfWrongHiddenLayersNumber_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizesLhs = {2, 3};
        int[] hiddenLayerSizesRhs = {2, 3, 5};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesLhs, nOutputs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesRhs, nOutputs);
        
        weightsLhs.dot(weightsRhs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testDot_ArgumentOfWrongHiddenLayersSizes_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizesLhs = {2, 3};
        int[] hiddenLayerSizesRhs = {2, 4};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesLhs, nOutputs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizesRhs, nOutputs);
        
        weightsLhs.dot(weightsRhs);
        
        Assert.fail();
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testDot_ArgumentOfWrongOutputSizes_Throws() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputsLhs = 4;
        int nOutputsRhs = 5;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputsLhs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputsRhs);
        
        weightsLhs.dot(weightsRhs);
        
        Assert.fail();
    }
    
    @Test
    public void testDot_ArgumentOfSameSize_CorrectResult() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        int[][] weightInsertIdxsLhs = {{0, 0, 0}, {0, 1, 1}, {2, 1, 2}, {1, 1, 1}};
        double[] weightInsertValsLhs = {12.5, -45.7, 3.9, 5};
        int[][] biasInsertIdxsLhs = {{0, 0}, {2, 2}, {0, 1}};
        double[] biasInsertValsLhs = {-4.4, 34442.2, 5};
        for (int i = 0; i < weightInsertIdxsLhs.length; i++) {
            weightsLhs.setWeight(weightInsertIdxsLhs[i][0], weightInsertIdxsLhs[i][1], weightInsertIdxsLhs[i][2], weightInsertValsLhs[i]);
        }
        for (int i = 0; i < biasInsertIdxsLhs.length; i++) {
            weightsLhs.setBias(biasInsertIdxsLhs[i][0], biasInsertIdxsLhs[i][1], biasInsertValsLhs[i]);
        }
        
        int[][] weightInsertIdxsRhs = {{0, 0, 0}, {0, 1, 1}, {2, 1, 2}};
        double[] weightInsertValsRhs = {11.3, -0.7, 14214.9};
        int[][] biasInsertIdxsRhs = {{0, 0}, {2, 2}, {1, 0}};
        double[] biasInsertValsRhs = {-5.4, -3.9, 4.5};
        for (int i = 0; i < weightInsertIdxsRhs.length; i++) {
            weightsRhs.setWeight(weightInsertIdxsRhs[i][0], weightInsertIdxsRhs[i][1], weightInsertIdxsRhs[i][2], weightInsertValsRhs[i]);
        }
        for (int i = 0; i < biasInsertIdxsLhs.length; i++) {
            weightsRhs.setBias(biasInsertIdxsRhs[i][0], biasInsertIdxsRhs[i][1], biasInsertValsRhs[i]);
        }
        double expected = 12.5*11.3+(-45.7)*(-0.7)+3.9*14214.9+(-4.4)*(-5.4)+34442.2*(-3.9);
        
        double actual = weightsLhs.dot(weightsRhs);
        
        Assert.assertEquals(expected, actual, DELTA);
    }
    
    @Test
    public void testDot_ArgumentOfSameSize_LhsAndRhsStaySame() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weightsLhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        NeuralNetworkWeights weightsRhs = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        int[][] weightInsertIdxsLhs = {{0, 0, 0}, {0, 1, 1}, {2, 1, 2}, {1, 1, 1}};
        double[] weightInsertValsLhs = {12.5, -45.7, 3.9, 5};
        int[][] biasInsertIdxsLhs = {{0, 0}, {2, 2}, {0, 1}};
        double[] biasInsertValsLhs = {-4.4, 34442.2, 5};
        for (int i = 0; i < weightInsertIdxsLhs.length; i++) {
            weightsLhs.setWeight(weightInsertIdxsLhs[i][0], weightInsertIdxsLhs[i][1], weightInsertIdxsLhs[i][2], weightInsertValsLhs[i]);
        }
        for (int i = 0; i < biasInsertIdxsLhs.length; i++) {
            weightsLhs.setBias(biasInsertIdxsLhs[i][0], biasInsertIdxsLhs[i][1], biasInsertValsLhs[i]);
        }
        
        int[][] weightInsertIdxsRhs = {{0, 0, 0}, {0, 1, 1}, {2, 1, 2}};
        double[] weightInsertValsRhs = {11.3, -0.7, 14214.9};
        int[][] biasInsertIdxsRhs = {{0, 0}, {2, 2}, {1, 0}};
        double[] biasInsertValsRhs = {-5.4, -3.9, 4.5};
        for (int i = 0; i < weightInsertIdxsRhs.length; i++) {
            weightsRhs.setWeight(weightInsertIdxsRhs[i][0], weightInsertIdxsRhs[i][1], weightInsertIdxsRhs[i][2], weightInsertValsRhs[i]);
        }
        for (int i = 0; i < biasInsertIdxsLhs.length; i++) {
            weightsRhs.setBias(biasInsertIdxsRhs[i][0], biasInsertIdxsRhs[i][1], biasInsertValsRhs[i]);
        }
        double[][][] weightsLhsBefore = copyWeights(weightsLhs.weights);
        double[][] biasesLhsBefore = copyBiases(weightsLhs.biases);
        
        double[][][] weightsRhsBefore = copyWeights(weightsRhs.weights);
        double[][] biasesRhsBefore = copyBiases(weightsRhs.biases);
        
        weightsLhs.dot(weightsRhs);
        
        assertArraysEqual(weightsLhsBefore, weightsLhs.weights);
        assertArraysEqual(biasesLhsBefore, weightsLhs.biases);
        
        assertArraysEqual(weightsRhsBefore, weightsRhs.weights);
        assertArraysEqual(biasesRhsBefore, weightsRhs.biases);
    }
     
    @Test
    public void testNorm_Invoke_CorrectResult() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        int[][] weightInsertIdxs = {{0, 0, 0}, {0, 1, 1}, {2, 1, 2}, {1, 1, 1}};
        double[] weightInsertVals = {12.5, -45.7, 3.9, 5};
        int[][] biasInsertIdxs = {{0, 0}, {2, 2}, {0, 1}};
        double[] biasInsertVals = {-4.4, 34442.2, 5};
        for (int i = 0; i < weightInsertIdxs.length; i++) {
            weights.setWeight(weightInsertIdxs[i][0], weightInsertIdxs[i][1], weightInsertIdxs[i][2], weightInsertVals[i]);
        }
        for (int i = 0; i < biasInsertIdxs.length; i++) {
            weights.setBias(biasInsertIdxs[i][0], biasInsertIdxs[i][1], biasInsertVals[i]);
        }
        double expected = Math.sqrt(12.5*12.5+(-45.7)*(-45.7)+3.9*3.9+5*5+(-4.4)*(-4.4)+34442.2*34442.2 + 5*5);
        
        double actual = weights.norm();
        
        Assert.assertEquals(expected, actual, DELTA);
    }
    
    @Test
    public void testNorm_Invoke_WeightsAndBiasesStaySame() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        int[][] weightInsertIdxs = {{0, 0, 0}, {0, 1, 1}, {2, 1, 2}, {1, 1, 1}};
        double[] weightInsertVals = {12.5, -45.7, 3.9, 5};
        int[][] biasInsertIdxs = {{0, 0}, {2, 2}, {0, 1}};
        double[] biasInsertVals = {-4.4, 34442.2, 5};
        for (int i = 0; i < weightInsertIdxs.length; i++) {
            weights.setWeight(weightInsertIdxs[i][0], weightInsertIdxs[i][1], weightInsertIdxs[i][2], weightInsertVals[i]);
        }
        for (int i = 0; i < biasInsertIdxs.length; i++) {
            weights.setBias(biasInsertIdxs[i][0], biasInsertIdxs[i][1], biasInsertVals[i]);
        }
        double[][][] weightsBefore = copyWeights(weights.weights);
        double[][] biasesBefore = copyBiases(weights.biases);
        
        weights.norm();
        
        assertArraysEqual(weightsBefore, weights.weights);
        assertArraysEqual(biasesBefore, weights.biases);
    }
    
    @Test
    public void testGetBias_NotSet_ReturnsZeroForEveryIndex() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        for (int i = 0; i < hiddenLayerSizes.length; i++) {
            for (int j = 0; j < hiddenLayerSizes[i]; j++) {
                Assert.assertEquals(0.0, weights.getBias(i, j), DELTA);
            }
        }
        
        for (int j = 0; j < nOutputs; j++) {
            Assert.assertEquals(0.0, weights.getBias(hiddenLayerSizes.length, j), DELTA);
        }
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetBias_NegativeFirstArgument_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.getBias(-1, 1);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetBias_NegativeSecondArgument_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.getBias(0, -1);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetBias_FirstArgumentGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.getBias(3, 1);
        
        Assert.fail();
    }
    
    @Test(expected = IndexOutOfBoundsException.class)
    public void testGetBias_SecondArgumentGreaterThanBound_Throws() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.getBias(0, 2);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        weights.setBias(layerNum, neuronNum, toSet);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        weights.setBias(layerNum, neuronNum, toSet);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        weights.setBias(layerNum, neuronNum, toSet);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        weights.setBias(layerNum, neuronNum, toSet);
        
        Assert.fail();
    }
    
    @Test
    public void testSetBias_SetValue_ReturnsSetValue() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double expected = 45.4;
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        weights.setBias(0, 1, expected);
        
        Assert.assertEquals(expected, weights.getBias(0, 1), DELTA);
    }
    
    @Test
    public void testSetBias_SetValue_OtherValuesNotChanged() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        double toSet = 45.4;
        int layerNum = 0;
        int neuronNum = 1;
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        double[][] expectedBiases = copyBiases(weights.biases);
        expectedBiases[layerNum][neuronNum] = toSet;
        double[][][] expectedWeights = copyWeights(weights.weights);
        
        weights.setBias(layerNum, neuronNum, toSet);
        
        double[][] actualBiases = weights.biases;
        if (!Arrays.deepEquals(actualBiases, expectedBiases)) {
            Assert.fail();
        }
        
        double[][][] actualWeights = weights.weights;
        if (!Arrays.deepEquals(actualWeights, expectedWeights)) {
            Assert.fail();
        }
    }
    
    @Test
    public void testGetWeight_NotSet_ReturnsZeroForEveryIndex() {
        int nInputs = 2;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 4;
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        double[][][] actualWeights = weights.weights;
        
        for (double[][] actualWeightForLayer : actualWeights) {
            for (double[] actualWeightForNeuron : actualWeightForLayer) {
                for (int k = 0; k < actualWeightForNeuron.length; k++) {
                    Assert.assertEquals(0.0, actualWeightForNeuron[k], DELTA);
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.getWeight(layerNum, prevNeuronNum, neuronNum);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.getWeight(layerNum, prevNeuronNum, neuronNum);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.getWeight(layerNum, prevNeuronNum, neuronNum);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.getWeight(layerNum, prevNeuronNum, neuronNum);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.getWeight(layerNum, prevNeuronNum, neuronNum);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.getWeight(layerNum, prevNeuronNum, neuronNum);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        double actual = weights.getWeight(layerNum, prevNeuronNum, neuronNum);
        
        Assert.assertEquals(toSet, actual, DELTA);
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        double[][][] expectedWeights = copyWeights(weights.weights);
        expectedWeights[layerNum][neuronNum][prevNeuronNum] = toSet;
        
        double[][] expectedBiases = copyBiases(weights.biases);
        
        weights.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        double[][][] actualWeights = weights.weights;
        
        double[][] actualBiases = weights.biases;
        
        if (!Arrays.deepEquals(actualWeights, expectedWeights)) {
            Assert.fail();
        }
        
        if (!Arrays.deepEquals(actualBiases, expectedBiases)) {
            Assert.fail();
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
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
        
        NeuralNetworkWeights weights = new NeuralNetworkWeights(nInputs, hiddenLayerSizes, nOutputs);
        
        weights.setWeight(layerNum, prevNeuronNum, neuronNum, toSet);
        
        Assert.fail();
    }
}
