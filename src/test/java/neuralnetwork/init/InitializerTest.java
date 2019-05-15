package neuralnetwork.init;

import neuralnetwork.TestUtils;
import org.junit.Test;
import static org.junit.Assert.*;
import org.mockito.Mockito;

/**
 * Initializer interface test cases.
 * @author Konstantin Zhdanov
 */
public class InitializerTest {
    
    public InitializerTest() {
    }

    /**
     * Test of of method, of class Initializer.
     */
    @Test(expected = NullPointerException.class)
    public void testOfWeightsSupplierBiasesSupplier_NullWeightSupplier_Throw() {
        System.out.println("of");
        WeightsSupplier weightsSupplier = null;
        BiasesSupplier biasesSupplier = (layerNum, layerNeuronNum) -> {
            return 0;
        };
        Initializer result = Initializer.of(weightsSupplier, biasesSupplier);
        
        fail("Test case must throw");
    }
    
    @Test(expected = NullPointerException.class)
    public void testOfWeightsSupplierBiasesSupplier_NullBiasSupplier_Throw() {
        System.out.println("of");
        WeightsSupplier weightsSupplier = (layerNum, prevLayerNeuronNum, layerNeuronNum) -> {
            return 0;
        };
        BiasesSupplier biasesSupplier = null;
        Initializer result = Initializer.of(weightsSupplier, biasesSupplier);
        
        fail("Test case must throw");
    }
    
    @Test
    public void testOfWeightsSupplierBiasesSupplier_CorrectSuppliers_SuppliersCalled() {
        System.out.println("of");
        WeightsSupplier weightsSupplier = Mockito.mock(WeightsSupplier.class);
        Mockito.when(weightsSupplier.supplyWeight(Mockito.anyInt(), Mockito.anyInt(), Mockito.anyInt())).
                thenReturn(1.0);
        BiasesSupplier biasesSupplier = Mockito.mock(BiasesSupplier.class);
        Mockito.when(biasesSupplier.supplyBias(Mockito.anyInt(), Mockito.anyInt())).
                thenReturn(2.0);
        Initializer result = Initializer.of(weightsSupplier, biasesSupplier);
        
        int[][] testWeightIndices = {{0, 0, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}, 
            {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1000, 2000, 3000}};
        int[][] testBiasIndices = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {1000, 2000}};
        
        for (int[] testWeightIndice : testWeightIndices) {
            assertEquals(1.0, 
                    result.supplyWeight(testWeightIndice[0], 
                            testWeightIndice[1], testWeightIndice[2]), 
                    TestUtils.DELTA);
        }
        
        for (int[] testBiasIndice : testBiasIndices) {
            assertEquals(2.0, 
                    result.supplyBias(testBiasIndice[0], testBiasIndice[1]), 
                    TestUtils.DELTA);
        }
        
        Mockito.verify(weightsSupplier, Mockito.times(testWeightIndices.length)).
                supplyWeight(Mockito.anyInt(), Mockito.anyInt(), Mockito.anyInt());
        Mockito.verify(biasesSupplier, Mockito.times(testBiasIndices.length)).
                supplyBias(Mockito.anyInt(), Mockito.anyInt());
    }

    /**
     * Test of of method, of class Initializer.
     */
    @Test
    public void testOfDoubleDouble_Invoked_ReturnedInstanceReturnCorrectValues() {
        System.out.println("of");
        double weightsValue = 2.0;
        double biasesValue = -5.5;
        
        Initializer result = Initializer.of(weightsValue, biasesValue);
        
        int[][] testWeightIndices = {{0, 0, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}, 
            {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1000, 2000, 3000}};
        int[][] testBiasIndices = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {1000, 2000}};
        
        for (int[] testWeightIndice : testWeightIndices) {
            assertEquals(weightsValue, 
                    result.supplyWeight(testWeightIndice[0], 
                            testWeightIndice[1], testWeightIndice[2]), 
                    TestUtils.DELTA);
        }
        
        for (int[] testBiasIndice : testBiasIndices) {
            assertEquals(biasesValue, 
                    result.supplyBias(testBiasIndice[0], testBiasIndice[1]), 
                    TestUtils.DELTA);
        }
    }

    /**
     * Test of ofStdRandomRange method, of class Initializer.
     */
    @Test
    public void testOfStdRandomRange_Invoked_ValuesInCorrectRange() {
        System.out.println("ofStdRandomRange");
        final double lower = 0.0;
        final double upper = 1.0;
        Initializer result = Initializer.ofStdRandomRange();
        
        int[][] testWeightIndices = {{0, 0, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}, 
            {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1000, 2000, 3000}};
        int[][] testBiasIndices = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {1000, 2000}};
        
        for (int[] testWeightIndice : testWeightIndices) {
            double weight = result.supplyWeight(testWeightIndice[0], 
                    testWeightIndice[1], testWeightIndice[2]);
            assertTrue(weight >= lower);
            assertTrue(weight <= upper);
        }
        
        for (int[] testBiasIndice : testBiasIndices) {
            double bias = result.supplyBias(testBiasIndice[0], 
                    testBiasIndice[1]);
            assertTrue(bias >= lower);
            assertTrue(bias <= upper);
        }
    }

    /**
     * Test of ofCustomRandomRange method, of class Initializer.
     */
    @Test(expected = IllegalArgumentException.class)
    public void testOfCustomRandomRange_MinWeightEqMaxWeight_Throw() {
       System.out.println("ofCustomRandomRange");
        double minWeightsValue = 5.0;
        double maxWeightsValue = 5.0;
        double minBiasesValue = 3.0;
        double maxBiasesValue = 12.0;
        Initializer result = Initializer.ofCustomRandomRange(minWeightsValue, 
                maxWeightsValue, minBiasesValue, maxBiasesValue);
        
        fail("Test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testOfCustomRandomRange_MinWeightGreaterMaxWeight_Throw() {
       System.out.println("ofCustomRandomRange");
        double minWeightsValue = 6.7;
        double maxWeightsValue = 5.0;
        double minBiasesValue = 3.0;
        double maxBiasesValue = 12.0;
        Initializer result = Initializer.ofCustomRandomRange(minWeightsValue, 
                maxWeightsValue, minBiasesValue, maxBiasesValue);
        
        fail("Test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testOfCustomRandomRange_MinBiasEqMaxBias_Throw() {
       System.out.println("ofCustomRandomRange");
        double minWeightsValue = 2.0;
        double maxWeightsValue = 5.0;
        double minBiasesValue = 12.0;
        double maxBiasesValue = 12.0;
        Initializer result = Initializer.ofCustomRandomRange(minWeightsValue, 
                maxWeightsValue, minBiasesValue, maxBiasesValue);
        
        fail("Test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testOfCustomRandomRange_MinBiasGreaterMaxBias_Throw() {
       System.out.println("ofCustomRandomRange");
        double minWeightsValue = 2.0;
        double maxWeightsValue = 5.0;
        double minBiasesValue = 13.4;
        double maxBiasesValue = 12.0;
        Initializer result = Initializer.ofCustomRandomRange(minWeightsValue, 
                maxWeightsValue, minBiasesValue, maxBiasesValue);
        
        fail("Test case must throw");
    }
    
    @Test
    public void testOfCustomRandomRange_CorrectPassed_ValuesInCorrectRange() {
        System.out.println("ofCustomRandomRange");
        double minWeightsValue = -2.3;
        double maxWeightsValue = 5.0;
        double minBiasesValue = 3.0;
        double maxBiasesValue = 12.0;
        Initializer result = Initializer.ofCustomRandomRange(minWeightsValue, 
                maxWeightsValue, minBiasesValue, maxBiasesValue);
        
        int[][] testWeightIndices = {{0, 0, 0}, {0, 0, 1}, {1, 0, 0}, {0, 1, 0}, 
            {1, 1, 0}, {1, 0, 1}, {0, 1, 1}, {1, 1, 1}, {1000, 2000, 3000}};
        int[][] testBiasIndices = {{0, 0}, {0, 1}, {1, 0}, {1, 1}, {1000, 2000}};
        
        for (int[] testWeightIndice : testWeightIndices) {
            double weight = result.supplyWeight(testWeightIndice[0], 
                    testWeightIndice[1], testWeightIndice[2]);
            assertTrue(weight >= minWeightsValue);
            assertTrue(weight <= maxWeightsValue);
        }
        
        for (int[] testBiasIndice : testBiasIndices) {
            double bias = result.supplyBias(testBiasIndice[0], 
                    testBiasIndice[1]);
            assertTrue(bias >= minBiasesValue);
            assertTrue(bias <= maxBiasesValue);
        }
    }
}
