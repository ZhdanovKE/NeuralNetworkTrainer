package neuralnetwork.train;

import neuralnetwork.ActivationFunctions;
import neuralnetwork.NeuralNetwork;
import java.util.concurrent.ExecutionException;
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
public class NeuralNetworkTrainerTaskTest {
    
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
    }
    
    @After
    public void cleanUpTestCase() {
        nn = null;
        worker = null;
        opts = null;
        inputs = null;
        targetResponses = null;
    }
    
    public NeuralNetworkTrainerTaskTest() {
    }

    /**
     * Test of setListener method, of class NeuralNetworkTrainerTask.
     */
    @Test
    public void testSetListener_EmptyTrainingFinished_OnTrainingCompleteCalled() {
        System.out.println("testSetListener_TrainingFinished_OnTrainingCompleteCalled");
        Listener listener = Mockito.mock(Listener.class);
        
        NeuralNetworkTrainerTask instance = new NeuralNetworkTrainerTask(worker);
        
        instance.setListener(listener);
        
        instance.run();
        Mockito.verify(listener).onTrainingComplete(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testSetListener_EmptyTrainingFinished_OnTrainingCanceledNotCalled() {
        System.out.println("testSetListener_EmptyTrainingFinished_OnTrainingCanceledNotCalled");
        Listener listener = Mockito.mock(Listener.class);
        
        NeuralNetworkTrainerTask instance = new NeuralNetworkTrainerTask(worker);
        
        instance.setListener(listener);
        
        instance.run();
        Mockito.verify(listener, Mockito.never()).onTrainingCanceled(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testSetListener_EmptyTrainingCanceled_OnTrainingCompleteNotCalled() {
        System.out.println("testSetListener_EmptyTrainingCanceled_OnTrainingCompleteNotCalled");
        Listener listener = Mockito.mock(Listener.class);
        
        NeuralNetworkTrainerTask instance = new NeuralNetworkTrainerTask(worker);
        
        instance.setListener(listener);
        
        instance.cancel(true);
        instance.run();
        
        Mockito.verify(listener, Mockito.never()).onTrainingComplete(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testSetListener_EmptyTrainingCanceledNonInterruptingly_OnTrainingCompleteNotCalled() {
        System.out.println("testSetListener_EmptyTrainingCanceledNonInterruptingly_OnTrainingCompleteNotCalled");
        Listener listener = Mockito.mock(Listener.class);
        
        NeuralNetworkTrainerTask instance = new NeuralNetworkTrainerTask(worker);
        
        instance.setListener(listener);
        
        instance.cancel(false);
        instance.run();
        
        Mockito.verify(listener, Mockito.never()).onTrainingComplete(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testSetListener_EmptyTrainingThrowException_OnTrainingCanceledCalled() throws Exception {
        System.out.println("testSetListener_EmptyTrainingThrowException_OnTrainingCanceledCalled");
        Listener listener = Mockito.mock(Listener.class);
        Mockito.when(worker.call()).thenThrow(new Exception("Artificial exception"));
        NeuralNetworkTrainerTask instance = new NeuralNetworkTrainerTask(worker);
        
        instance.setListener(listener);
        
        instance.run();
        
        Mockito.verify(listener).onTrainingCanceled(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testSetListener_EmptyTrainingThrowException_OnTrainingCompleteNotCalled() throws Exception {
        System.out.println("testSetListener_EmptyTrainingThrowException_OnTrainingCompleteNotCalled");
        Listener listener = Mockito.mock(Listener.class);
        Mockito.doThrow(new Exception("Artificial exception")).when(worker).call();
        NeuralNetworkTrainerTask instance = new NeuralNetworkTrainerTask(worker);
        
        instance.setListener(listener);
        
        instance.run();
        
        Mockito.verify(listener, Mockito.never()).onTrainingComplete(Mockito.any(TrainerEvent.class));
    }
    
    @Test
    public void testGet_WorkerReturnNN_ReturnSameNN() throws Exception {
        System.out.println("testGet_EmptyTrainingReturnNN_ReturnSameNN");
        Mockito.when(worker.call()).thenReturn(nn);
        NeuralNetworkTrainerTask instance = new NeuralNetworkTrainerTask(worker);
        
        instance.run();
        
        try {
            NeuralNetwork returnedNN = instance.get();
            assertEquals(nn, returnedNN);
        }
        catch(ExecutionException | InterruptedException e) {
            fail("Mock object don't throw exceptions and not interrupted");
        }
    }
    
    @Test(expected = ExecutionException.class)
    public void testGet_WorkerThrowException_Throw() throws Exception {
        System.out.println("testGet_WorkerThrowException_Throw");
        Mockito.when(worker.call()).thenThrow(new Exception("Artificial exception"));
        NeuralNetworkTrainerTask instance = new NeuralNetworkTrainerTask(worker);
        
        instance.run();
 
        instance.get();   
        
        fail("The test case must throw");
    }
    
    @Test
    public void testGet_RunAndCancel_ReturnNotNullNN() throws Exception {
        System.out.println("testGet_Cancel_Throw");
        NeuralNetwork expected = new NeuralNetwork(nn);
        Mockito.when(worker.getTrainedNeuralNetwork()).thenReturn(expected);
        NeuralNetworkTrainerTask instance = new NeuralNetworkTrainerTask(worker);
        
        instance.cancel(true);
        instance.run();

        NeuralNetwork result = instance.get();   
        
        assertNotNull("Paritally trained neural network cannot be null", result);
    }
    
    @Test
    public void testGet_RunAndCancel_ReturnNewNNFromWorker() throws Exception {
        System.out.println("testGet_Cancel_Throw");
        NeuralNetwork expected = new NeuralNetwork(nn);
        Mockito.when(worker.getTrainedNeuralNetwork()).thenReturn(expected);
        NeuralNetworkTrainerTask instance = new NeuralNetworkTrainerTask(worker);
        
        instance.cancel(true);
        instance.run();

        NeuralNetwork result = instance.get();   
        
        assertSame("Paritally trained neural network is different", expected, result);
    }
    
}
