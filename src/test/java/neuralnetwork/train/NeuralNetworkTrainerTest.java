package neuralnetwork.train;

import neuralnetwork.ActivationFunctions;
import neuralnetwork.NeuralNetwork;
import neuralnetwork.TestUtils;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.TimeUnit;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import static org.junit.Assert.*;
import org.mockito.Matchers;
import org.mockito.Mockito;

/**
 *
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkTrainerTest {
    
    
    private ExecutorService serialExecutor;
    
    NeuralNetworkTrainer.Builder builder;
    NeuralNetworkTrainerWorker worker;
    NeuralNetwork nn;
    NeuralNetworkTrainer.Options opts;
    double[][] inputs;
    double[][] targetResponses;
    
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
    
        opts = new NeuralNetworkTrainer.Options();
        opts.maxEpoch = 1;
        inputs = new double[][]{{0, 0, 0}, {1, 1, 1}, {0.5, 0.5, 0.5}};
        targetResponses = new double[][]{{1}, {0}, {0.5}};
        
        worker = Mockito.mock(NeuralNetworkTrainerWorker.class);
        
        builder = new NeuralNetworkTrainer.Builder();
        
        serialExecutor = TestUtils.getDirectExecutor();
    }
    
    @After
    public void cleanUpTestCase() {
        nn = null;
        worker = null;
        opts = null;
        inputs = null;
        targetResponses = null;
        builder = null;
        serialExecutor = null;
    }
    
    public NeuralNetworkTrainerTest() {
    }

    @Test(expected = IllegalArgumentException.class)
    public void testBuilderWithMaxEpoch_ArgumentZero_Throw() {
        System.out.println("testBuilderWithMaxEpoch_ArgumentZero_Throw");
        
        builder.withMaxEpoch(0);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testBuilderWithMaxEpoch_ArgumentNegative_Throw() {
        System.out.println("testBuilderWithMaxEpoch_ArgumentNegative_Throw");
        
        builder.withMaxEpoch(-1);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testBuilderWithPerformanceGoal_ArgumentZero_Throw() {
        System.out.println("testBuilderWithPerformanceGoal_ArgumentZero_Throw");
        
        builder.withPerformanceGoal(0.0);
        
        fail("The test case must throw");
    }
            
    @Test(expected = IllegalArgumentException.class)
    public void testBuilderWithPerformanceGoal_ArgumentNegative_Throw() {
        System.out.println("testBuilderWithPerformanceGoal_ArgumentNegative_Throw");
        
        builder.withPerformanceGoal(-1.5);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testBuilderWithTrainSamplesRatio_ArgumentZero_Throw() {
        System.out.println("testBuilderWithTrainSamplesRatio_ArgumentZero_Throw");
        
        builder.withTrainSamplesRatio(0);
        
        fail("The test case must throw");
    }
            
    @Test(expected = IllegalArgumentException.class)
    public void testBuilderWithTrainSamplesRatio_ArgumentNegative_Throw() {
        System.out.println("testBuilderWithTrainSamplesRatio_ArgumentNegative_Throw");
        
        builder.withTrainSamplesRatio(-2);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testBuilderWithTrainSamplesRatio_ArgumentGreaterThanHundred_Throw() {
        System.out.println("testBuilderWithTrainSamplesRatio_ArgumentGreaterThanHundred_Throw");
        
        builder.withTrainSamplesRatio(101);
        
        fail("The test case must throw");
    }
            
    @Test(expected = IllegalArgumentException.class)
    public void testBuilderWithValidationSamplesRatio_ArgumentNegative_Throw() {
        System.out.println("testBuilderWithValidationSamplesRatio_ArgumentNegative_Throw");
        
        builder.withValidationSamplesRatio(-1);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testBuilderWithValidationSamplesRatio_ArgumentGreaterThanHundred_Throw() {
        System.out.println("testBuilderWithValidationSamplesRatio_ArgumentGreaterThanHundred_Throw");
        
        builder.withValidationSamplesRatio(101);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testBuilderWithTestSamplesRatio_ArgumentNegative_Throw() {
        System.out.println("testBuilderWithTestSamplesRatio_ArgumentNegative_Throw");
        
        builder.withTestSamplesRatio(-1);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testBuilderWithTestSamplesRatio_ArgumentGreaterThanHundred_Throw() {
        System.out.println("testBuilderWithTestSamplesRatio_ArgumentGreaterThanHundred_Throw");
        
        builder.withTestSamplesRatio(101);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalStateException.class)
    public void testBuilderBuild_SamplesRatiosSumToLessThanHundred_Throw() {
        System.out.println("testBuilderBuild_SamplesRatiosSumToLessThanHundred_Throw");
        builder.withTrainSamplesRatio(45);
        builder.withValidationSamplesRatio(50);
        builder.withTestSamplesRatio(4);
        
        builder.build();
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalStateException.class)
    public void testBuilderBuild_SamplesRatiosSumToGreaterThanHundred_Throw() {
        System.out.println("testBuilderBuild_SamplesRatiosSumToGreaterThanHundred_Throw");
        builder.withTrainSamplesRatio(45);
        builder.withValidationSamplesRatio(50);
        builder.withTestSamplesRatio(6);
        
        builder.build();
        
        fail("The test case must throw");
    }
    
    @Test
    public void testBuilderBuild_InvokedDefault_TrainSamplesRatioIsHundred() {
        System.out.println("ttestBuilderBuild_InvokedDefault_TrainSamplesRatioIsHundred");
        
        NeuralNetworkTrainer trainer = builder.build();
        
        assertEquals("Default train samples ratio is not 100", 100, trainer.getTrainSamplesRatio());
    }
    
    @Test
    public void testBuilderBuild_InvokedDefault_ValidationSamplesRatioIsZero() {
        System.out.println("testBuilderBuild_InvokedDefault_ValidationSamplesRatioIsZero");
        
        NeuralNetworkTrainer trainer = builder.build();
        
        assertEquals("Default validation samples ratio is not 0", 0, trainer.getValidationSamplesRatio());
    }
    
    @Test
    public void testBuilderBuild_InvokedDefault_TestSamplesRatioIsZero() {
        System.out.println("testBuilderBuild_InvokedDefault_TestSamplesRatioIsZero");
        
        NeuralNetworkTrainer trainer = builder.build();
        
        assertEquals("Default test samples ratio is not 0", 0, trainer.getTestSamplesRatio());
    }
    
    @Test
    public void testBuilderBuild_InvokedDefault_MaxEpochIsOne() {
        System.out.println("testBuilderBuild_InvokedDefault_MaxEpochIsOne");
        
        NeuralNetworkTrainer trainer = builder.build();
        
        assertEquals("Default max epoch number is not 1", 1, trainer.getMaxEpoch());
    }
    
    @Test
    public void testBuilderBuild_InvokedDefault_PerformanceGoalGreaterThanZero() {
        System.out.println("testBuilderBuild_InvokedDefault_PerformanceGoalGreaterThanZero");
        
        NeuralNetworkTrainer trainer = builder.build();
        
        if (trainer.getPerformanceGoal() <= 0) {
            fail("Default performance goal is zero or negative");
        }
    }
    
    @Test
    public void testBuilderWithExecutor_StartTrainCalled_ExecutorSubmitMethodCalled() {
        System.out.println("testBuilderWithExecutor_StartTrainCalled_ExecutorSubmitCalled");
        ExecutorService fakeExecutor = Mockito.mock(ExecutorService.class);
        NeuralNetworkTrainer instance = builder.withExecutor(fakeExecutor).build();
        
        instance.startTrain(nn, inputs, targetResponses);
        
        Mockito.verify(fakeExecutor, Mockito.atLeastOnce()).submit(Mockito.any(Runnable.class));
    }
    
    @Test(expected = NullPointerException.class)
    public void testBuilderWithExecutor_NullExecutor_Throw() {
        System.out.println("testBuilderWithExecutor_NullExecutor_Throw");
        ExecutorService nullExecutor = null;
        builder.withExecutor(nullExecutor).build();
        
        fail("The test case must throw");
    }
    
    /**
     * Test of registerListener method, of class NeuralNetworkTrainer.
     */
    @Test(expected = NullPointerException.class)
    public void testRegisterListener_NullArgument_Throws() {
        System.out.println("testRegisterListener_NullArgument_Throws");
        Listener listener = null;
        NeuralNetworkTrainer instance = builder.build();
        instance.registerListener(listener);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testRegisterListener_ValidArgument_Ok() {
        System.out.println("testRegisterListener_ValidArgument_Ok");
        Listener listener = Mockito.mock(Listener.class);
        
        NeuralNetworkTrainer instance = builder.build();
        instance.registerListener(listener);
    }
    
    @Test
    public void testRegisterListener_TrainingFinished_OnTrainingCompleteCalled() {
        System.out.println("testRegisterListener_TrainingFinished_OnTrainingCompleteCalled");
        Listener listener = Mockito.mock(Listener.class);
        
        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        instance.registerListener(listener);
        
        instance.startTrain(nn, inputs, targetResponses);
        
        Mockito.verify(listener).onTrainingComplete(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testRegisterListener_TrainingFinished_OnTrainingCanceledNotCalled() {
        System.out.println("testRegisterListener_TrainingFinished_OnTrainingCanceledNotCalled");
        Listener listener = Mockito.mock(Listener.class);
        
        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        instance.registerListener(listener);
        
        instance.startTrain(nn, inputs, targetResponses);
        
        Mockito.verify(listener, Mockito.never()).onTrainingCanceled(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testRegisterListener_TwoListenersRegisteredTrainingFinished_OnTrainingCompleteCalledOnEachListener() {
        System.out.println("testRegisterListener_TwoListenersRegisteredTrainingFinished_OnTrainingCompleteCalledOnEachListener");
        Listener listenerOne = Mockito.mock(Listener.class);
        Listener listenerTwo = Mockito.mock(Listener.class);
        
        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        instance.registerListener(listenerOne);
        instance.registerListener(listenerTwo);
        
        instance.startTrain(nn, inputs, targetResponses);
        
        Mockito.verify(listenerOne).onTrainingComplete(Mockito.any(TrainerEvent.class));
        Mockito.verify(listenerTwo).onTrainingComplete(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testRegisterListener_TwoListenersRegisteredTrainingFinished_OnTrainingCanceledNotCalledOnEachListener() {
        System.out.println("testRegisterListener_TwoListenersRegisteredTrainingFinished_OnTrainingCanceledNotCalledOnEachListener");
        Listener listenerOne = Mockito.mock(Listener.class);
        Listener listenerTwo = Mockito.mock(Listener.class);
        
        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        instance.registerListener(listenerOne);
        instance.registerListener(listenerTwo);
        
        instance.startTrain(nn, inputs, targetResponses);
        
        Mockito.verify(listenerOne, Mockito.never()).onTrainingCanceled(Mockito.any(TrainerEvent.class));
        Mockito.verify(listenerTwo, Mockito.never()).onTrainingCanceled(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testRegisterListener_TrainingTwoEpochsFinished_OnTrainingEpochCompleteCalledTwice() {
        System.out.println("testRegisterListener_TrainingTwoEpochsFinished_OnTrainingEpochCompleteCalledTwice");
        Listener listener = Mockito.mock(Listener.class);
        
        NeuralNetworkTrainer instance = builder.withMaxEpoch(2).
                withExecutor(serialExecutor).
                build();
        instance.registerListener(listener);
        
        instance.startTrain(nn, inputs, targetResponses);
        
        Mockito.verify(listener, Mockito.times(2)).onTrainingEpochComplete(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testRegisterListener_TwoListenersTrainingTwoEpochsFinished_OnTrainingEpochCompleteCalledTwiceOnEachListener() {
        System.out.println("testRegisterListener_TwoListenersTrainingTwoEpochsFinished_OnTrainingEpochCompleteCalledTwiceOnEachListener");
        Listener listenerOne = Mockito.mock(Listener.class);
        Listener listenerTwo = Mockito.mock(Listener.class);
        
        NeuralNetworkTrainer instance = builder.withMaxEpoch(2).
                withExecutor(serialExecutor).
                build();
        instance.registerListener(listenerOne);
        instance.registerListener(listenerTwo);
        
        instance.startTrain(nn, inputs, targetResponses);
        
        Mockito.verify(listenerOne, Mockito.times(2)).onTrainingEpochComplete(Mockito.any(TrainerEvent.class));
        Mockito.verify(listenerTwo, Mockito.times(2)).onTrainingEpochComplete(Mockito.any(TrainerEvent.class));
    }

    /**
     * Test of removeListener method, of class NeuralNetworkTrainer.
     */
    @Test(expected = NullPointerException.class)
    public void testRemoveListener_NullArgument_Throws() {
        System.out.println("testRemoveListener_NullArgument_Throws");
        Listener listener = null;
        NeuralNetworkTrainer instance = builder.build();
        instance.removeListener(listener);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testRemoveListener_NotPreviouslyAddedListenerPassed_Ok() {
        System.out.println("testRemoveListener_NotPreviouslyAddedListenerPassed_Ok");
        Listener listener = Mockito.mock(Listener.class);
        NeuralNetworkTrainer instance = builder.build();
        
        instance.removeListener(listener);

    }
    
    @Test
    public void testRemoveListener_PreviouslyAddedListenerPassed_Ok() {
        System.out.println("testRemoveListener_PreviouslyAddedListenerPassed_Ok");
        Listener listener = Mockito.mock(Listener.class);
        NeuralNetworkTrainer instance = builder.build();
        instance.registerListener(listener);
        
        instance.removeListener(listener);
    }
    
    @Test
    public void testRemoveListener_TrainingFinished_MethodsNotCalled() {
        System.out.println("testRemoveListener_TrainingFinished_MethodsNotCalled");
        Listener listener = Mockito.mock(Listener.class);
        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).build();
        instance.registerListener(listener);
        instance.removeListener(listener);
        
        instance.startTrain(nn, inputs, targetResponses);
        
        Mockito.verifyZeroInteractions(listener);
    }
    
    @Test
    public void testRemoveListener_RealExecutorTrainingFinished_MethodsNotCalled() {
        System.out.println("testRemoveListener_TrainingFinished_MethodsNotCalled");
        Listener listener = Mockito.mock(Listener.class);
        NeuralNetworkTrainer instance = builder.build();
        instance.registerListener(listener);
        instance.removeListener(listener);
        
        instance.startTrain(nn, inputs, targetResponses);
        // block till the result is ready
        instance.getTrainedNetwork();
        
        Mockito.verifyZeroInteractions(listener);
    }
    
    @Test
    public void testRemoveListener_RealExecutorTrainingStopped_MethodsNotCalled() {
        System.out.println("testRemoveListener_RealExecutorTrainingStopped_MethodsNotCalled");
        Listener listener = Mockito.mock(Listener.class);
        NeuralNetworkTrainer instance = builder.build();
        instance.registerListener(listener);
        instance.removeListener(listener);
        
        instance.startTrain(nn, inputs, targetResponses);
        instance.stopTraining();
        
        Mockito.verifyZeroInteractions(listener);
    }

    /**
     * Test of stopTraining method, of class NeuralNetworkTrainer.
     */
    @Test
    public void testStopTraining_NoTrainingStarted_Ok() {
        System.out.println("testStopTraining_NoTrainingStarted_Ok");
        NeuralNetworkTrainer instance = builder.build();
        instance.stopTraining();
    }
    
    @Test
    public void testStopTraining_RealExecutorTrainingStarted_Ok() {
        System.out.println("testStopTraining_RealExecutorTrainingStarted_Ok");
        NeuralNetworkTrainer instance = builder.build();

        instance.startTrain(nn, inputs, targetResponses);
        instance.stopTraining();
        
        instance.getTrainedNetwork();
    }
    
    @Test
    public void testStopTraining_RealExecutorTrainingStartedDoubleCall_Ok() {
        System.out.println("testStopTraining_RealExecutorTrainingStartedDoubleCall_Ok");
        NeuralNetworkTrainer instance = builder.build();
        instance.startTrain(nn, inputs, targetResponses);
        instance.stopTraining();
        instance.stopTraining();
        
        instance.getTrainedNetwork();
    }

    /**
     * Test of trainingFinished method, of class NeuralNetworkTrainer.
     */
    @Test
    public void testTrainingFinished_TrainingNotStarted_ReturnsTrue() {
        System.out.println("testTrainingFinished_TrainingNotStarted_ReturnsTrue");
        NeuralNetworkTrainer instance = builder.build();
        boolean expResult = true;
        
        boolean result = instance.trainingFinished();
        
        assertEquals(expResult, result);
    }
    
    @Test
    public void testTrainingFinished_RealExecutorTrainingTenEpochStopped_ReturnsTrue() {
        System.out.println("testTrainingFinished_RealExecutorTrainingTenEpochStopped_ReturnsTrue");
        NeuralNetworkTrainer instance = builder.withMaxEpoch(10).build();
        boolean expResult = true;
        
        instance.startTrain(nn, inputs, targetResponses);
        
        instance.stopTraining();
        
        boolean result = instance.trainingFinished();
        
        // block until the real stop is performed
        instance.getTrainedNetwork();
        assertEquals("Training not finished after requesting stop", expResult, result);
    }
    
    @Test
    public void testTrainingFinished_RealExecutorTrainingTenEpochStoppedAfterSleep_ReturnsTrue() {
        System.out.println("testTrainingFinished_RealExecutorTrainingTenEpochStoppedAfterSleep_ReturnsTrue");
        NeuralNetworkTrainer instance = builder.withMaxEpoch(10).build();
        boolean expResult = true;
        
        instance.startTrain(nn, inputs, targetResponses);
        try {
            TimeUnit.MILLISECONDS.sleep(10);
        }
        catch (InterruptedException e) {
            
        }
        instance.stopTraining();
        
        boolean result = instance.trainingFinished();
        
        // block until the real stop is performed
        instance.getTrainedNetwork();
        assertEquals("Training not finished after requesting stop", expResult, result);
    }
    
    @Test
    public void testTrainingFinished_TrainingFinished_ReturnsTrue() {
        System.out.println("testTrainingFinished_TrainingFinished_ReturnsTrue");
        builder.withExecutor(serialExecutor);
        NeuralNetworkTrainer instance = builder.build();
        instance.startTrain(nn, inputs, targetResponses);
        boolean expResult = true;
        
        boolean result = instance.trainingFinished();
        assertEquals(expResult, result);
    }
    
    @Test
    public void testTrainingFinished_RealExecutorTrainingTenEpochFinished_ReturnsTrue() {
        System.out.println("testTrainingFinished_RealExecutorTrainingTenEpochFinished_ReturnsTrue");
        NeuralNetworkTrainer instance = builder.withMaxEpoch(10).build();
        instance.startTrain(nn, inputs, targetResponses);
        boolean expResult = true;
        // block until result is ready
        instance.getTrainedNetwork();
        
        boolean result = instance.trainingFinished();
        assertEquals(expResult, result);
    }
    
    @Test
    public void testTrainingFinished_RealExecutorTrainingTenEpochStartedNotFinished_ReturnsFalse() {
        System.out.println("testTrainingFinished_RealExecutorTrainingTenEpochStartedNotFinished_ReturnsFalse");
        NeuralNetworkTrainer instance = builder.withMaxEpoch(10).build();
        boolean expResult = false;
        instance.startTrain(nn, inputs, targetResponses);
        
        boolean result = instance.trainingFinished();
        
        assertEquals(expResult, result);
    }

    /**
     * Test of startTrain method, of class NeuralNetworkTrainer.
     */
    @Test(expected = NullPointerException.class)
    public void testStartTrain_NullNN_Throw() {
        System.out.println("testStartTrain_NullNN_Throw");

        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        
        instance.startTrain(null, inputs, targetResponses);
        
        fail("The test case must throw");
    }
    
    @Test(expected = NullPointerException.class)
    public void testStartTrain_NullInputs_Throw() {
        System.out.println("testStartTrain_NullInputs_Throw");

        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        
        instance.startTrain(nn, null, targetResponses);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStartTrain_InputsDimLessNNInputs_Throw() {
        System.out.println("testStartTrain_InputsDimLessNNInputs_Throw");

        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        
        double[][] inputsSmaller = new double[inputs.length][];
        for (int i = 0; i < inputsSmaller.length; i++) {
            inputsSmaller[i] = new double[inputs[i].length - 1];
        }
        
        instance.startTrain(nn, inputsSmaller, targetResponses);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStartTrain_InputsDimGreaterNNInputs_Throw() {
        System.out.println("testStartTrain_InputsDimGreaterNNInputs_Throw");

        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        
        double[][] inputsLarger = new double[inputs.length][];
        for (int i = 0; i < inputsLarger.length; i++) {
            inputsLarger[i] = new double[inputs[i].length + 1];
        }
        
        instance.startTrain(nn, inputsLarger, targetResponses);
        
        fail("The test case must throw");
    }
    
    @Test(expected = NullPointerException.class)
    public void testStartTrain_NullTargetsThrow() {
        System.out.println("testStartTrain_NullTargets_Throw");

        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        
        instance.startTrain(nn, inputs, null);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStartTrain_TargetsDimLessNNOutputs_Throw() {
        System.out.println("testStartTrain_TargetsDimLessNNOutputs_Throw");

        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        
        double[][] targetResponsesSmaller = new double[targetResponses.length][];
        for (int i = 0; i < targetResponsesSmaller.length; i++) {
            targetResponsesSmaller[i] = new double[targetResponses[i].length - 1];
        }
        
        instance.startTrain(nn, inputs, targetResponsesSmaller);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStartTrain_TargetsDimGreaterNNOutputs_Throw() {
        System.out.println("testStartTrain_TargetsDimGreaterNNOutputs_Throw");

        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        
        double[][] targetResponsesLarger = new double[targetResponses.length][];
        for (int i = 0; i < targetResponsesLarger.length; i++) {
            targetResponsesLarger[i] = new double[targetResponses[i].length + 1];
        }
        
        instance.startTrain(nn, inputs, targetResponsesLarger);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStartTrain_NumInputsLessNumTargets_Throw() {
        System.out.println("testStartTrain_NumInputsLessNumTargets_Throw");

        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        
        double[][] inputsLess = new double[inputs.length - 1][];
        for (int i = 0; i < inputsLess.length; i++) {
            inputsLess[i] = inputs[i].clone();
        }
        instance.startTrain(nn, inputsLess, targetResponses);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testStartTrain_NumInputsGreaterNumTargets_Throw() {
        System.out.println("testStartTrain_NumInputsLessNumTargets_Throw");

        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).
                build();
        
        double[][] targetResponsesLess = new double[targetResponses.length - 1][];
        for (int i = 0; i < targetResponsesLess.length; i++) {
            targetResponsesLess[i] = targetResponses[i].clone();
        }
        instance.startTrain(nn, inputs, targetResponsesLess);
        
        fail("The test case must throw");
    }
    
    /**
     * Test of getTrainedNetwork method, of class NeuralNetworkTrainer.
     */
    @Test
    public void testGetTrainedNetwork_NoTrainingStarted_ReturnNull() {
        System.out.println("testGetTrainedNetwork_NoTrainingStarted_ReturnNull");
        NeuralNetworkTrainer instance = builder.build();
        
        NeuralNetwork result = instance.getTrainedNetwork();
        
        assertNull("Must return null for empty training", result);
    }
    
    @Test
    public void testGetTrainedNetwork_TrainingFinished_ReturnSameStructureNN() {
        System.out.println("testGetTrainedNetwork_TrainingFinished_ReturnSameStructureNN");
        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).build();
        
        instance.startTrain(nn, inputs, targetResponses);
        
        NeuralNetwork result = instance.getTrainedNetwork();
        
        if (!TestUtils.sameStructure(nn, result)) {
            fail("Structure of the trained network is different");
        }
    }
    
    @Test
    public void testGetTrainedNetwork_RealExecutorTrainingFinished_ReturnSameStructureNN() {
        System.out.println("testGetTrainedNetwork_RealExecutorTrainingFinished_ReturnSameStructureNN");
        NeuralNetworkTrainer instance = builder.build();
        
        instance.startTrain(nn, inputs, targetResponses);
        
        NeuralNetwork result = instance.getTrainedNetwork();
        
        if (!TestUtils.sameStructure(nn, result)) {
            fail("Structure of the trained network is different");
        }
    }
    
    @Test
    public void testGetTrainedNetwork_TrainingFinished_ReturnDifferentNN() {
        System.out.println("testGetTrainedNetwork_TrainingFinished_ReturnDifferentNN");
        NeuralNetworkTrainer instance = builder.withExecutor(serialExecutor).build();
        
        instance.startTrain(nn, inputs, targetResponses);
        
        NeuralNetwork result = instance.getTrainedNetwork();
        
        TestUtils.assertNNNotEquals(nn, result);
    }
    
    @Test
    public void testGetTrainedNetwork_RealExecutorTrainingFinished_ReturnDifferentNN() {
        System.out.println("testGetTrainedNetwork_RealExecutorTrainingFinished_ReturnDifferentNN");
        NeuralNetworkTrainer instance = builder.build();
        
        instance.startTrain(nn, inputs, targetResponses);
        
        NeuralNetwork result = instance.getTrainedNetwork();
        
        TestUtils.assertNNNotEquals(nn, result);
    }
    
    @Test
    public void testGetTrainedNetwork_RealExecutorTrainingStarted_TrainingFinished() {
        System.out.println("testGetTrainedNetwork_RealExecutorTrainingStarted_TrainingFinished");
        NeuralNetworkTrainer instance = builder.withMaxEpoch(10).build();
        
        instance.startTrain(nn, inputs, targetResponses);
        
        instance.getTrainedNetwork();
        
        assertEquals("Training must be finished after getting the trained network", true, instance.trainingFinished());
    }
    
    @Test
    public void testGetTrainedNetwork_RealExecutorTrainingStopped_ReturnNotNullNN() {
        System.out.println("testGetTrainedNetwork_RealExecutorTrainingStopped_ReturnNotNullNN");
        NeuralNetworkTrainer instance = builder.withMaxEpoch(10).build();
        
        instance.startTrain(nn, inputs, targetResponses);
        instance.stopTraining();
        
        NeuralNetwork result = instance.getTrainedNetwork();
        
        assertNotNull("Trained neural network cannot be null after stopping training", result);
    }
    
    /**
     * Test of error (3 arguments) method, of class NeuralNetworkTrainer.
     */
    @Test(expected = NullPointerException.class)
    public void testErrorNN__NullFirstArgument_Throws() {
        double[] input = new double[3];
        double[] target = new double[1];

        NeuralNetworkTrainer.error(null, input, target);
        
        fail("The test case must throw");
    }
    
    @Test(expected = NullPointerException.class)
    public void testErrorNN__NullSecondArgument_Throws() {
        double[] input = null;
        double[] target = new double[1];

        NeuralNetworkTrainer.error(nn, input, target);
        
        fail("The test case must throw");
    }
    
    @Test(expected = NullPointerException.class)
    public void testErrorNN__NullThirdArgument_Throws() {
        double[] input = new double[3];
        double[] target = null;

        NeuralNetworkTrainer.error(nn, input, target);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testErrorNN_InputSizeNotEqualNumInputs_Throws() {
        double[] input = new double[2];
        double[] target = new double[1];

        NeuralNetworkTrainer.error(nn, input, target);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testErrorNN_TargetSizeNotEqualNumOutputs_Throws() {
        double[] actual = new double[3];
        double[] expected = new double[2];

        NeuralNetworkTrainer.error(nn, actual, expected);
        
        fail("The test case must throw");
    }
    
    /**
     * Test of error method, of class NeuralNetworkTrainer.
     */
    @Test(expected = NullPointerException.class)
    public void testError_NullFirstArgument_Throws() {
        double[] actual = null;
        double[] expected = new double[3];

        NeuralNetworkTrainer.error(actual, expected);
        
        fail("The test case must throw");
    }

    @Test(expected = NullPointerException.class)
    public void testError_NullSecondArgument_Throws() {
        double[] actual = new double[3];
        double[] expected = null;

        NeuralNetworkTrainer.error(actual, expected);
        
        fail("The test case must throw");
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testError_ArgumentsDifferentSizes_Throws() {
        double[] actual = new double[3];
        double[] expected = new double[4];

        NeuralNetworkTrainer.error(actual, expected);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testError_ArgumentsSameSizes_CorrectResult() {
        double[] actual = {0, 0.4, 0.9};
        double[] expected = {1, 1, 0.1};
        double offset = 1e-15;
        
        double expectedResult = 0.0;
        for (int i = 0; i < actual.length; i++) {
            expectedResult += -expected[i]*Math.log(offset + actual[i]) - (1-expected[i])*Math.log(offset + 1 - actual[i]);
        }
        expectedResult /= actual.length;
        double actualResult = NeuralNetworkTrainer.error(actual, expected);
        
        assertEquals(expectedResult, actualResult, TestUtils.DELTA);
    }
    
    /**
     * Test of normalizeSamples method, of class NeuralNetworkTrainer.
     */
    @Test(expected = NullPointerException.class)
    public void testNormalizeSamples_NullArgument_Throws() {
        System.out.println("testNormalizeSamples_NullArgument_Throws");
        NeuralNetworkTrainer instance = builder.build();
        
        instance.normalizeSamples(null);
        
        fail("The test case must throw");
    }
    
    @Test
    public void testNormalizeSamples_ValidArgument_ArgumentValuesBetweenMinusZeroAndOne() {
        System.out.println("testNormalizeSamples_ValidArgument_ArgumentValuesBetweenMinusZeroAndOne");
        NeuralNetworkTrainer instance = builder.build();
        
        instance.normalizeSamples(inputs);
        
        for (double[] sample : inputs) {
            for (double value : sample) {
                if (value < 0 || value > 1) {
                    fail("Not in range");
                }
            }
        }
    }

    /**
     * Test of getPerformanceGoal method, of class NeuralNetworkTrainer.
     */
    @Test
    public void testGetPerformanceGoal_ValuePassedToBuilder_ReturnSameValue() {
        System.out.println("testPerformanceGoal_ValuePassedToBuilder_ReturnSameValue");
        double expResult = 5.3;
        NeuralNetworkTrainer instance = builder.withPerformanceGoal(expResult).build();
        double result = instance.getPerformanceGoal();
        assertEquals(expResult, result, TestUtils.DELTA);
    }

    /**
     * Test of getMaxEpoch method, of class NeuralNetworkTrainer.
     */
    @Test
    public void testGetMaxEpoch_ValuePassedToBuilder_ReturnSameValue() {
        System.out.println("testGetMaxEpoch_ValuePassedToBuilder_ReturnSameValue");
        int expResult = 14;
        NeuralNetworkTrainer instance = builder.withMaxEpoch(expResult).build();
        int result = instance.getMaxEpoch();
        assertEquals("Max epoch number hasn't been saved", expResult, result);
    }

    /**
     * Test of getTrainSamplesRatio method, of class NeuralNetworkTrainer.
     */
    @Test
    public void testGetTrainSamplesRatio_ValuePassedToBuilder_ReturnSameValue() {
        System.out.println("testGetTrainSamplesRatio_ValuePassedToBuilder_ReturnSameValue");
        int expResult = 45;
        int testRatio = 100 - expResult;
        NeuralNetworkTrainer instance = builder.withTrainSamplesRatio(expResult).
                withTestSamplesRatio(testRatio).build();
        
        int result = instance.getTrainSamplesRatio();
        assertEquals("Train samples ratio hasn't been saved", expResult, result);
    }

    /**
     * Test of getValidationSamplesRatio method, of class NeuralNetworkTrainer.
     */
    @Test
    public void testGetValidationSamplesRatio_ValuePassedToBuilder_ReturnSameValue() {
        System.out.println("testGetValidationSamplesRatio_ValuePassedToBuilder_ReturnSameValue");
        int expResult = 45;
        int trainRatio = 100 - expResult;
        NeuralNetworkTrainer instance = builder.withTrainSamplesRatio(trainRatio).
                withValidationSamplesRatio(expResult).build();
        
        int result = instance.getValidationSamplesRatio();
        assertEquals("Validation samples ratio hasn't been saved", expResult, result);
    }

    /**
     * Test of getTestSamplesRatio method, of class NeuralNetworkTrainer.
     */
    @Test
    public void testGetTestSamplesRatio_ValuePassedToBuilder_ReturnSameValue() {
        System.out.println("testGetTestSamplesRatio_ValuePassedToBuilder_ReturnSameValue");
        int expResult = 45;
        int trainRatio = 100 - expResult;
        NeuralNetworkTrainer instance = builder.withTrainSamplesRatio(trainRatio).
                withTestSamplesRatio(expResult).build();
        
        int result = instance.getTestSamplesRatio();
        assertEquals("Test samples ratio hasn't been saved", expResult, result);
    }
    
}
