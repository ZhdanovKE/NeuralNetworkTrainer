package neuralnetwork.util;

import java.util.Arrays;
import java.util.HashMap;
import static org.junit.Assert.*;
import org.junit.Test;


/**
 *
 * @author Konstantin Zhdanov
 */
public class RandomizerTest {
    
    public RandomizerTest() {
    }

    /**
     * Test of getRandomElements method, of class Randomizer.
     */
    @Test(expected = NullPointerException.class)
    public void testGetRandomElements_NullArray_Throw() {
        int[] array = null;
        int num = 1;
        int from = 0;
        int to = 1;
        Randomizer instance = new Randomizer();
        instance.getRandomElements(array, num, from, to);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testGetRandomElements_NegativeNumberToTake_Throw() {
        int[] array = {1, 2};
        int num = -1;
        int from = 0;
        int to = 1;
        Randomizer instance = new Randomizer();
        instance.getRandomElements(array, num, from, to);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testGetRandomElements_NegativeFromIndex_Throw() {
        int[] array = {1, 2};
        int num = 1;
        int from = -1;
        int to = 1;
        Randomizer instance = new Randomizer();
        instance.getRandomElements(array, num, from, to);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testGetRandomElements_NegativeToIndex_Throw() {
        int[] array = {1, 2};
        int num = 1;
        int from = 1;
        int to = -1;
        Randomizer instance = new Randomizer();
        instance.getRandomElements(array, num, from, to);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testGetRandomElements_FromIndexGreaterToIndex_Throw() {
        int[] array = {1, 2};
        int num = 1;
        int from = 5;
        int to = 1;
        Randomizer instance = new Randomizer();
        instance.getRandomElements(array, num, from, to);
        
        fail("The test case must throw");
    }
    
    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testGetRandomElements_FromIndexGreaterArrayLength_Throw() {
        int[] array = {1, 2};
        int num = 1;
        int from = 5;
        int to = 6;
        Randomizer instance = new Randomizer();
        instance.getRandomElements(array, num, from, to);
        
        fail("The test case must throw");
    }
    
    @Test(expected = ArrayIndexOutOfBoundsException.class)
    public void testGetRandomElements_ToIndexGreaterArrayLength_Throw() {
        int[] array = {1, 2};
        int num = 1;
        int from = 0;
        int to = 6;
        Randomizer instance = new Randomizer();
        instance.getRandomElements(array, num, from, to);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testGetRandomElements_RangeLessThanNumToTake_Throw() {
        int[] array = {1, 2};
        int num = 2;
        int from = 0;
        int to = 1;
        Randomizer instance = new Randomizer();
        instance.getRandomElements(array, num, from, to);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testGetRandomElements_ValidArguments_ReturnsRequestedNumberElements() {
        int[] array = {1, 2, 3, 4, 5, 6};
        int num = 2;
        int from = 0;
        int to = 4;
        Randomizer instance = new Randomizer();
        int[] result = instance.getRandomElements(array, num, from, to);
        
        assertEquals("Returned array of wrong size", num, result.length);
    }
    
    @Test
    public void testGetRandomElements_RequestedZeroNumberToTake_ReturnsEmptyArray() {
        int[] array = {1, 2, 3, 4, 5};
        int num = 0;
        int from = 0;
        int to = 3;
        Randomizer instance = new Randomizer();
        int[] result = instance.getRandomElements(array, num, from, to);
        
        assertEquals("Returned array of wrong size", num, result.length);
    }
    
    @Test
    public void testGetRandomElements_UniqueElemArrayPassed_ReturnsArrayWithUniqueElems() {
        int[] array = {1, 2, 3, 4, 5, 6};
        int num = 4;
        int from = 0;
        int to = 5;
        Randomizer instance = new Randomizer();
        int[] result = instance.getRandomElements(array, num, from, to);
        
        Arrays.sort(result);
        for (int i = 1; i < result.length; i++) {
            assertNotEquals("Uniqueness of elemens not preserved", result[i - 1], result[i]);
        }
    }
    
    @Test
    public void testGetRandomElements_UniqueElemArrayPassed_ReturnsArrayWithElemsInPassedArray() {
        int[] array = {1, 2, 3, 4, 5, 6};
        int num = 4;
        int from = 0;
        int to = 5;
        Randomizer instance = new Randomizer();
        int[] result = instance.getRandomElements(array, num, from, to);
        
        outer:
        for (int i = 0; i < result.length; i++) {
            for (int j  = 0; j < array.length; j++) {
                if (array[j] == result[i]) {
                    continue outer;
                }
            }
            fail("Returned array contains an element (" + result[i] + ") not present in tht original array");
        }
    }
    
    @Test
    public void testGetRandomElements_ArrayWithRepetitionsPassed_ReturnsArrayWithCorrectFrequencies() {
        int numTests = 10;
        int[] array = {1, 2, 2, 4, 5, 5};
        int num = 4;
        int from = 0;
        int to = 5;
        
        int[] subarrayOriginal = Arrays.copyOfRange(array, from, to);
        HashMap<Integer, Integer> frequenciesOriginal = new HashMap<>();
        Arrays.stream(subarrayOriginal).forEach(k->frequenciesOriginal.merge(k, 1, (v1, v2) -> v1 + v2));
        
        for (int i = 0; i < numTests; i++) {
            Randomizer instance = new Randomizer();
            int[] result = instance.getRandomElements(array, num, from, to);

            HashMap<Integer, Integer> frequenciesActual = new HashMap<>();
            Arrays.stream(result).forEach(k->frequenciesActual.merge(k, 1, (v1, v2) -> v1 + v2));

            frequenciesActual.forEach((k,v) -> {
                if (frequenciesOriginal.getOrDefault(k, 0) < v) {
                    fail("Resulting array contains value " + k + " (" + v + 
                            " times), but original subarray contains value " + k + 
                            " (" + v + " times)");
                }
            });
        }
        
    }
    
    @Test
    public void testGetRandomElements_ArrayPassed_PassedArrayNotChanged() {
        int[] array = {1, 2, 3, 4, 5, 6};
        int[] expected = array.clone();
        int num = 4;
        int from = 0;
        int to = 5;
        Randomizer instance = new Randomizer();
        instance.getRandomElements(array, num, from, to);
        
        assertArrayEquals("Passed elements have been changed by the call", expected, array);
    }
}
