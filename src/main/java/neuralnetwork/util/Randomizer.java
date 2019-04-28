package neuralnetwork.util;

import java.util.Random;

/**
 *
 * @author Konstantin Zhdanov
 */
public class Randomizer {
    private final Random rnd = new Random();
    
    /** Randomly pick a number of elements from an array
     * <p>Get the required number {@link num} of elements from the provided array {@link array} randomly. 
     * The {@link array} is not changed after the function's call</p>
     * 
     * @param array Source array which elements will be returned randomly
     * @param num Number of elements to take randomly
     * @param from Starting index (inclusive) in the {@link array} from which to take elements
     * @param to Ending index (exclusive) in the {@link array} upto which to take elements
     * @return a new array of {@link num) randomly picked elements from the {@link array}
     **/
    public int[] getRandomElements(int[] array, int num, int from, int to) {
        checkArguments(array, num, from, to);
        
        int[] arrayCopy = array.clone();
        int[] randoms = new int[num];
        
        for (int numTaken = 0; numTaken < num; numTaken++) {
            int takeIdx = from + numTaken + rnd.nextInt(to - numTaken - from);
            randoms[numTaken] = arrayCopy[takeIdx];
            
            // swap the taken index with the first index on the left side (not taken indices)
            arrayCopy[takeIdx] = arrayCopy[numTaken + from];
        }
        
        return randoms;
    }
    
    private void checkArguments(int[] array, int num, int from, int to) {
        if (array == null) {
            throw new NullPointerException("Array cannot be null");
        }
        if (num < 0) {
            throw new IllegalArgumentException("Number of elements to pick cannot be negative");
        }
        if (from < 0 || to < 0 || from > to) {
            throw new IllegalArgumentException("Lower and/or upper bound is incorrect");
        }
        if (from > array.length || to > array.length) {
            throw new ArrayIndexOutOfBoundsException("Range is out of bounds");
        }
        if (num > (to - from)) {
            throw new IllegalArgumentException("Number of elements to pick (" + num + ") exceeds the provided range [" + from + ", " + to + ")");
        }
    }
}
