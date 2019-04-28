package neuralnetwork.train;

import neuralnetwork.ActivationFunction;
import neuralnetwork.ActivationFunctions;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.TestUtils;
import org.junit.After;
import org.junit.Test;
import static org.junit.Assert.*;
import org.junit.Before;
import org.mockito.Matchers;
import org.mockito.Mockito;

/**
 *
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkTrainerWorkerTest {
    
    NeuralNetwork nn;
    
    @Before
    public void initTestCase() {
        int nInputs = 3;
        int[] hiddenLayerSizes = {2, 3};
        int nOutputs = 1;
        nn = Mockito.mock(NeuralNetwork.class);
        Mockito.when(nn.getBias(Matchers.anyInt(), Matchers.anyInt())).thenReturn(0.1);
        Mockito.when(nn.getWeight(Matchers.anyInt(), Matchers.anyInt(), Matchers.anyInt())).thenReturn(0.1);
        Mockito.when(nn.getActivationFunction()).thenReturn(ActivationFunctions.SIGMOID);
        Mockito.when(nn.getNumberInputs()).thenReturn(nInputs);
        Mockito.when(nn.getNumberOutputs()).thenReturn(nOutputs);
        Mockito.when(nn.getHiddenLayerSizes()).thenReturn(hiddenLayerSizes);
        Mockito.when(nn.getNumberHiddenLayers()).thenReturn(hiddenLayerSizes.length);
        Mockito.when(nn.getHiddenLayerSize(0)).thenReturn(hiddenLayerSizes[0]);
        Mockito.when(nn.getHiddenLayerSize(1)).thenReturn(hiddenLayerSizes[1]);
    }
    
    @After
    public void cleanUpTestCase() {
        nn = null;
    }
    
    public NeuralNetworkTrainerWorkerTest() {
    }

    /**
     * Test of setListener method, of class NeuralNetworkTrainerWorker.
     */
    @Test
    public void testSetListener_NullListenerTrainingFinished_Ok() {
        System.out.println("testSetListener_NullListenerTrainingFinished_Ok");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 1;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
       
        Listener listener = null;

        instance.setListener(listener);
        
        instance.train();
    }
    
    @Test
    public void testSetListener_ValidListenerTrainingFinished_OnTrainingCompleteCalled() {
        System.out.println("testSetListener_ValidListenerTrainingFinished_OnTrainingCompleteCalled");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 1;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
       
        Listener listener = Mockito.mock(Listener.class);

        instance.setListener(listener);
        
        instance.train();
        
        Mockito.verify(listener).onTrainingComplete(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testSetListener_ValidListenerTwoEpochsFinished_OnEpochCompleteCalledTwice() {
        System.out.println("testSetListener_ValidListenerTwoEpochsFinished_OnEpochCompleteCalledTwice");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 2;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
       
        Listener listener = Mockito.mock(Listener.class);

        instance.setListener(listener);
        
        instance.train();
        
        Mockito.verify(listener, Mockito.times(2)).onTrainingEpochComplete(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testSetListener_ValidListenerTrainingThreadInterrupted_OnTrainingCancelledCalled() {
        System.out.println("testSetListener_ValidListenerTrainingThreadInterrupted_OnTrainingCancelledCalled");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 1;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
       
        Listener listener = Mockito.mock(Listener.class);

        instance.setListener(listener);
        
        Thread.currentThread().interrupt();
        instance.train();
        
        Mockito.verify(listener).onTrainingCanceled(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testSetListener_ValidListenerTrainingFinished_OnTrainingCancelledNotCalled() {
        System.out.println("testSetListener_ValidListenerTrainingFinished_OnTrainingCancelledNotCalled");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 1;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
       
        Listener listener = Mockito.mock(Listener.class);

        instance.setListener(listener);
        
        instance.train();
        
        Mockito.verify(listener, Mockito.never()).onTrainingCanceled(Mockito.any(TrainerEvent.class));
    }
    
    /**
     * Test of call method, of class NeuralNetworkTrainerWorker.
     * @throws Exception
     */
    @Test
    public void testCall_ThreadInterruptedBefore_ReturnEqualNN() throws Exception {
        System.out.println("testCall_ThreadInterruptedBefore_ReturnEqualNN");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 10;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
        
        Thread.currentThread().interrupt();
        NeuralNetwork result = instance.call();
        
        TestUtils.assertNNEquals(nn, result);
    }
    
    @Test
    public void testCall_TenEpochs_TrainedNNHasAtLeastOneDifferentWeight() throws Exception{
        System.out.println("testCall_TenEpochs_TrainedNNHasAtLeastOneDifferentWeight");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 10;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        
        double[][][] weightsBefore = TestUtils.extractNNWeights(nn);
        double[][] biasesBefore = TestUtils.extractNNBiases(nn);
        
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
        
        NeuralNetwork trainedNN = instance.call();
        double[][][] weightsAfter = TestUtils.extractNNWeights(trainedNN);
        double[][] biasesAfter = TestUtils.extractNNBiases(trainedNN);
        
        if (TestUtils.arraysEqual(weightsBefore, weightsAfter)) {
            fail("Weights haven't changed");
        }
        if (TestUtils.arraysEqual(biasesBefore, biasesAfter)) {
            fail("Biases haven't changed");
        }
    }
    
    @Test
    public void testCall_TenEpochs_PerformanceDecreasedForReturnedNN() throws Exception {
        System.out.println("testCall_TenEpochs_PerformanceDecreasedForReturnedNN");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 10;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
        
        NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(nn);
        double performanceBefore = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            NeuralNetworkResponse resp = evaluator.evaluate(inputs[i]);
            performanceBefore += NeuralNetworkTrainer.error(resp.getOutputs(), targetResponses[i]);
        }
        performanceBefore /= inputs.length;
        
        NeuralNetwork trainedNN = instance.call();
        
        double performanceAfter = 0.0;
        NeuralNetworkEvaluator evaluatorAfter = new NeuralNetworkEvaluator(trainedNN);
        for (int i = 0; i < inputs.length; i++) {
            NeuralNetworkResponse resp = evaluatorAfter.evaluate(inputs[i]);
            performanceAfter += NeuralNetworkTrainer.error(resp.getOutputs(), targetResponses[i]);
        }
        performanceAfter /= inputs.length;
               
        System.out.println("Before: " + performanceBefore + ", after: " + performanceAfter);
        if (performanceBefore <= performanceAfter) {
            fail("Couldn't decrease performance");
        }
    }

    /**
     * Test of startTrain method, of class NeuralNetworkTrainerWorker.
     */
    @Test
    public void testTrain_TenEpochs_ArgumentNotChanged() {
        System.out.println("testTrain_TenEpochs_ArgumentNotChanged");
        Mockito.doThrow(new AssertionError()).when(nn).setBias(Matchers.anyInt(), Matchers.anyInt(), Matchers.anyDouble());
        Mockito.doThrow(new AssertionError()).when(nn).setWeight(Matchers.anyInt(), Matchers.anyInt(), Matchers.anyInt(), Matchers.anyDouble());
        Mockito.doThrow(new AssertionError()).when(nn).setActivationFunction(Matchers.any(ActivationFunction.class));
        
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 10;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
        instance.train();
    }
    
    @Test
    public void testTrain_TenEpochs_TrainedNNHasAtLeastOneDifferentWeight() {
        System.out.println("testTrain_TenEpochs_TrainedNNHasAtLeastOneDifferentWeight");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 10;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        
        double[][][] weightsBefore = TestUtils.extractNNWeights(nn);
        double[][] biasesBefore = TestUtils.extractNNBiases(nn);
        
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
        instance.train();
        
        NeuralNetwork trainedNN = instance.getTrainedNeuralNetwork();
        double[][][] weightsAfter = TestUtils.extractNNWeights(trainedNN);
        double[][] biasesAfter = TestUtils.extractNNBiases(trainedNN);
        
        if (TestUtils.arraysEqual(weightsBefore, weightsAfter)) {
            fail("Weights haven't changed");
        }
        if (TestUtils.arraysEqual(biasesBefore, biasesAfter)) {
            fail("Biases haven't changed");
        }
    }
    
    @Test
    public void testTrain_TenEpochs_PerformanceDecreased() {
        System.out.println("testTrain_TenEpochs_PerformanceDecreased");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 10;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
        
        NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(nn);
        double performanceBefore = 0.0;
        for (int i = 0; i < inputs.length; i++) {
            NeuralNetworkResponse resp = evaluator.evaluate(inputs[i]);
            performanceBefore += NeuralNetworkTrainer.error(resp.getOutputs(), targetResponses[i]);
        }
        performanceBefore /= inputs.length;
        
        instance.train();
        
        NeuralNetwork trainedNN = instance.getTrainedNeuralNetwork();
        double performanceAfter = 0.0;
        NeuralNetworkEvaluator evaluatorAfter = new NeuralNetworkEvaluator(trainedNN);
        for (int i = 0; i < inputs.length; i++) {
            NeuralNetworkResponse resp = evaluatorAfter.evaluate(inputs[i]);
            performanceAfter += NeuralNetworkTrainer.error(resp.getOutputs(), targetResponses[i]);
        }
        performanceAfter /= inputs.length;
               
        System.out.println("Before: " + performanceBefore + ", after: " + performanceAfter);
        if (performanceBefore <= performanceAfter) {
            fail("Couldn't decrease performance");
        }
        
    }

    /**
     * Test of getTrainedNeuralNetwork method, of class NeuralNetworkTrainerWorker.
     */
    @Test
    public void testGetTrainedNeuralNetwork_NotTrained_ReturnEqualNN() {
        System.out.println("testGetTrainedNeuralNetwork_NotTrained_ReturnEqualNN");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 10;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
        
        NeuralNetwork result = instance.getTrainedNeuralNetwork();
        
        TestUtils.assertNNEquals(nn, result);
    }
    
    @Test
    public void testGetTrainedNeuralNetwork_ChangeReturnedNN_OriginalNNStaySame() {
        System.out.println("testGetTrainedNeuralNetwork_ChangeReturnedNN_OriginalNNStaySame");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 10;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
        double[][][] weightsBefore = TestUtils.extractNNWeights(nn);
        double[][] biasesBefore = TestUtils.extractNNBiases(nn);
        
        NeuralNetwork result = instance.getTrainedNeuralNetwork();
        result.setWeight(0, 1, 0, result.getWeight(0, 1, 0) + 3);
        
        TestUtils.assertArraysEqual(weightsBefore, TestUtils.extractNNWeights(nn));
        TestUtils.assertArraysEqual(biasesBefore, TestUtils.extractNNBiases(nn));
    }
    
    @Test
    public void testGetTrainedNeuralNetwork_TrainedCalled_ReturnNNSameStructure() {
        System.out.println("testGetTrainedNeuralNetwork_TrainedCalled_ReturnNNSameStructure");
        NeuralNetworkTrainer.Options opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 10;
        double[][] inputs = {{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        double[][] targetResponses = {{1}, {0}, {0.5}};
        NeuralNetworkTrainerWorker instance = new NeuralNetworkTrainerWorker(nn, inputs, targetResponses, opts);
        instance.train();
        
        NeuralNetwork result = instance.getTrainedNeuralNetwork();
        
        assertEquals("Number of inputs changed", nn.getNumberInputs(), result.getNumberInputs());
        assertEquals("Number of outputs changed", nn.getNumberOutputs(), result.getNumberOutputs());
        assertEquals("Number of hidden layers changed", nn.getNumberHiddenLayers(), result.getNumberHiddenLayers());
        assertArrayEquals("Hidden layer sizes changed", nn.getHiddenLayerSizes(), result.getHiddenLayerSizes());
        assertSame("Activation function changed", nn.getActivationFunction(), result.getActivationFunction());
    }
    
    
}
