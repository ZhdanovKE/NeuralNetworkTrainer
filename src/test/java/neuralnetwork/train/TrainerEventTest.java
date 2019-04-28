package neuralnetwork.train;

import neuralnetwork.TestUtils;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Konstantin Zhdanov
 */
public class TrainerEventTest {
    
    public TrainerEventTest() {
    }

    @Test(expected = IllegalArgumentException.class)
    public void testConstructor_NegativeEpochNum_Throw() {
        System.out.println("testConstructor_NegativeEpochNum_Throw");
        
        TrainerEvent instance = new TrainerEvent(-1, 0.0);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testConstructor_ZeroEpochNum_Ok() {
        System.out.println("testConstructor_ZeroEpochNum_Ok");
        
        TrainerEvent instance = new TrainerEvent(0, 4.0);
    }
    
    /**
     * Test of getEpoch method, of class TrainerEvent.
     */
    @Test
    public void testGetEpoch_ValuePassedToConstructor_ReturnSameValue() {
        System.out.println("testGetEpoch_ValuePassedToConstructor_ReturnSameValue");
        int expectedEpoch = 154;
        double performance = 45.3;
        TrainerEvent instance = new TrainerEvent(expectedEpoch, performance);

        int result = instance.getEpoch();
        
        assertEquals("Epoch num isn't saved", expectedEpoch, result);
    }

    /**
     * Test of getPerformance method, of class TrainerEvent.
     */
    @Test
    public void testGetPerformance_ValuePassedToConstructor_ReturnSameValue() {
        System.out.println("testGetPerformance_ValuePassedToConstructor_ReturnSameValue");
        int epoch = 154;
        double expectedPerformance = 45.3;
        TrainerEvent instance = new TrainerEvent(epoch, expectedPerformance);

        double result = instance.getPerformance();
        
        assertEquals(expectedPerformance, result, TestUtils.DELTA);
    }
    
}
