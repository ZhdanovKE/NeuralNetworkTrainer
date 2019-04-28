package neuralnetwork.train;

import neuralnetwork.NeuralNetwork;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.FutureTask;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;

/**
 * An implementation of {@code FutureTask<NeuralNetwork>} representing
 * a network training task that can be performed in a separate thread and
 * supports training notifications.
 * @author Konstantin Zhdanov
 */
class NeuralNetworkTrainerTask extends FutureTask<NeuralNetwork>{
    
    /** A worker object for performing the actual training. */
    private NeuralNetworkTrainerWorker worker;
    
    /** A user-provided listener to be called when training events occur. */
    private volatile Listener listener;
    
    /** The last event' info of the finished training whether it was
     completed or canceled. */
    private volatile TrainerEvent finalEvent;
    
    /** Listener to be attached to the training worker thread. */
    private final Listener workerListener = 
            new Listener() {
                @Override
                public void onTrainingComplete(TrainerEvent event) {
                    NeuralNetworkTrainerTask.this.finalEvent = event;
                }

                @Override
                public void onTrainingCanceled(TrainerEvent event) {
                    NeuralNetworkTrainerTask.this.finalEvent = event;
                    NeuralNetworkTrainerTask.this.onTrainingCanceled(event);
                }

                @Override
                public void onTrainingEpochComplete(TrainerEvent event) {
                    NeuralNetworkTrainerTask.this.onTrainingEpochDone(event);
                }
            };
    
    NeuralNetworkTrainerTask(NeuralNetwork nn, 
                            double[][] inputs,
                            double[][] targets, 
                            NeuralNetworkTrainer.Options opts) {
        
        this(new NeuralNetworkTrainerWorker(
                    nn, inputs, targets, opts));
    }
    
    NeuralNetworkTrainerTask(NeuralNetworkTrainerWorker worker) {
       super(worker);
       this.worker = worker;
       this.worker.setListener(workerListener);
    }

    /** 
     * Set a listener to be called when training events occur.
     * @param listener {@code Listener} to attach to this task.
     */
    public void setListener(Listener listener) {
        this.listener = listener;
    }
    
    @Override
    protected void done() {
        super.done();
        if (!isCancelled()) {
            try {
                get();
                onTrainingDone(finalEvent);
            }
            catch(ExecutionException e) {
                onTrainingCanceled(finalEvent);
            }
            catch(InterruptedException e) {
                // Shouldn't happen
                assert false;
            }
        }
        // else - will be notified from the Worker class
    }
    
    /**
     * Get the resulting {@code NeuralNetwork} object of training even if the
     * training has been canceled mid-training
     * @return {@code NeuralNetwork} object representing a trained or partially trained (if training was canceled) neural network
     * @throws InterruptedException
     * @throws ExecutionException 
     */
    @Override
    public NeuralNetwork get() throws InterruptedException, ExecutionException {
        try {
            return super.get();
        }
        catch(CancellationException e) {
            if (this.worker == null) {
                return null;
            }
            return this.worker.getTrainedNeuralNetwork();
        }
    }

    /**
     * Get the resulting {@code NeuralNetwork} object of training even if the
     * training has been canceled mid-training. The call blocks for the specified time tops.
     * @param timeout Value of maximum time to wait for the result in {@link unit}
     * @param unit Unit of {@link timeout} value
     * @return {@code NeuralNetwork} object representing a trained or partially trained (if training was canceled) neural network
     * @throws InterruptedException
     * @throws ExecutionException 
     * @throws TimeoutException
     */
    @Override
    public NeuralNetwork get(long timeout, TimeUnit unit) throws InterruptedException, ExecutionException, TimeoutException {
        try {
            return super.get(timeout, unit);
        }
        catch(CancellationException e) {
            if (this.worker == null) {
                return null;
            }
            return this.worker.getTrainedNeuralNetwork();
        }
    }
    
    private void onTrainingDone(TrainerEvent event) {
        Listener shapshotListener = listener;
        if (shapshotListener != null) {
            shapshotListener.onTrainingComplete(event);
        }
    }
    
    private void onTrainingEpochDone(TrainerEvent event) {
        Listener shapshotListener = listener;
        if (shapshotListener != null) {
            shapshotListener.onTrainingEpochComplete(event);
        }
    }
    
    private void onTrainingCanceled(TrainerEvent event) {
        Listener shapshotListener = listener;
        if (shapshotListener != null) {
            shapshotListener.onTrainingCanceled(event);
        }
    }
}
