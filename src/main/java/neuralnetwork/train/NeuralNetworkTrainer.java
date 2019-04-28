package neuralnetwork.train;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.samples.NeuralNetworkSamplesNormalizer;
import neuralnetwork.samples.NeuralNetworkSamplesNormalizerAsym;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.CancellationException;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.FutureTask;

/**
 * The trainer starts a separate thread for performing the actual optimization
 * of a network.
 * The trainer uses the cross-entropy error function for calculating
 * the error between the real and the target outputs of a network.
 * Input and target values must be in [0,1] for the training algorithm
 * to work correctly. If the passed values are outside of the range,
 * the trainer automatically normalizes them to be in [0,1] for the training.
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkTrainer {
    
    /** Holder of optimization options. */
    static class Options {
    
        /** Maximum number of iterations (epochs) of training loop. */
        int maxEpoch;

        /** Target performance to stop the training. */
        double performanceGoal;

        /** Ratio of startTrain samples with respect to the number of all samples (in per cent). */
        int trainSamplesRatio;

        /** Ratio of validation samples with respect to the number of all samples (in per cent). */
        int validationSamplesRatio;

        /** Ratio of test samples with respect to the number of all samples (in per cent). */
        int testSamplesRatio;
        
        /** Create an {@code Options} object holding the passed values. */
        Options(int maxEpoch, int performanceGoal, int trainRatio, int validationRatio, int testRatio) {
            this.maxEpoch = maxEpoch;
            this.performanceGoal = performanceGoal;
            this.trainSamplesRatio = trainRatio;
            this.validationSamplesRatio = validationRatio;
            this.testSamplesRatio = testRatio;
        }
        
        /** Set default values. */
        Options() {
            maxEpoch = 1;
            performanceGoal = Math.pow(10, -2);
            trainSamplesRatio = 100;
            validationSamplesRatio = 0;
            testSamplesRatio = 0;
        }
    }
    
    /** Optimization options. */
    private Options options;
    
    /** List of user-attached listeners. */
    private List<Listener> listeners;
    
    /** Listener to register with {@code NeuralNetworkTrainerTask} object. */
    private final Listener trainingListener = new Listener() {
        @Override public void onTrainingComplete(TrainerEvent event) {
            NeuralNetworkTrainer.this.notifyTrainingComplete(event);
        }

        @Override public void onTrainingCanceled(TrainerEvent event) {
            NeuralNetworkTrainer.this.notifyTrainingCanceled(event);
        }

        @Override public void onTrainingEpochComplete(TrainerEvent event) {
            NeuralNetworkTrainer.this.notifyTrainingEpochComplete(event);
        }
    };
    
    /** Executor for running {@code NeuralNetworkTrainerTask} threads. */
    private final ExecutorService executor;
    
    /** Currently running or last finished/canceled training async task. */
    private FutureTask<NeuralNetwork> currentTraining;
    
    /** Normalizer for neural network's inputs used to map inputs into a predefined range. */
    private final NeuralNetworkSamplesNormalizer sampleNormalizer;
    
    /** Normalizer for neural network's outputs used to map targets into a predefined range. */
    private final NeuralNetworkSamplesNormalizer targetNormalizer;
    
    /** Class used for creating {@code NeuralNetworkTrainer} objects with different parameters. */
    public static class Builder {
        ExecutorService executor;
        
        double step;
        int maxEpoch;
        double performanceGoal;
    
        int trainSamplesRatio;
        int validationSamplesRatio;
        int testSamplesRatio;
        
        Options opts;
        
        public Builder() {
            initDefaultValues();
        }
        
        private void initDefaultValues() {
            opts = new Options();
            executor = null;
        }
        
        public Builder withMaxEpoch(int maxEpoch) {
            if (maxEpoch <= 0) {
                throw new IllegalArgumentException("Maximum epoch cannot be nonpositive");
            }
            this.opts.maxEpoch = maxEpoch;
            return this;
        }
        
        public Builder withPerformanceGoal(double performanceGoal) {
            if (performanceGoal <= 0) {
                throw new IllegalArgumentException("Target performance cannot be nonpositive");
            }
            this.opts.performanceGoal = performanceGoal;
            return this;
        }
        
        public Builder withTrainSamplesRatio(int ratio) {
            if (ratio < 1 || ratio > 100) {
                throw new IllegalArgumentException("Ratio of train samples must be inside [1,100]");
            }
            this.opts.trainSamplesRatio = ratio;
            return this;
        }
        
        public Builder withValidationSamplesRatio(int ratio) {
            if (ratio < 0 || ratio > 100) {
                throw new IllegalArgumentException("Ratio of validation samples must be inside [0,100]");
            }
            this.opts.validationSamplesRatio = ratio;
            return this;
        }
        
        public Builder withTestSamplesRatio(int ratio) {
            if (ratio < 0 || ratio > 100) {
                throw new IllegalArgumentException("Ratio of test samples must be inside [0,100]");
            }
            this.opts.testSamplesRatio = ratio;
            return this;
        }
        
        // package for testing
        Builder withExecutor(ExecutorService executor) {
            if (executor == null) {
                throw new NullPointerException("Executor cannot be null");
            }
            this.executor = executor;
            return this;
        }
        
        public NeuralNetworkTrainer build() {
            if (opts.trainSamplesRatio + opts.validationSamplesRatio + opts.testSamplesRatio != 100) {
                throw new IllegalStateException("Numbers of train, "
                        + "validation and test samples must sum up to 100%");
            }
            NeuralNetworkTrainer nnt;
            if (executor == null) {
                nnt = new NeuralNetworkTrainer();
            }
            else {
                nnt = new NeuralNetworkTrainer(executor);
            }
            nnt.options = this.opts;
            
            return nnt;
        }
    }
    
    // package for testing
    NeuralNetworkTrainer() {
        this(TrainerExecutors.newExecutor());
    }
    
    // package for testing
    NeuralNetworkTrainer(ExecutorService executor) {
        this.sampleNormalizer = new NeuralNetworkSamplesNormalizerAsym();
        this.targetNormalizer = new NeuralNetworkSamplesNormalizerAsym();
        this.listeners = new LinkedList<>();
        this.executor = executor;
    }
    
    /** Target performance to stop the training.
     * @return {@code double} value representing the performance when the training
     * must be stopped.
     */
    public double getPerformanceGoal() {
        return options.performanceGoal;
    }
    
    /** Maximum number of iterations (epochs) of training loop.
     * @return {@code int} value representing the maximum number of
     * iterations of the training algorithm.
     */
    public int getMaxEpoch() {
        return options.maxEpoch;
    }
    
    /** Ratio of the samples used for training with respect to the number of 
     * all samples (in per cent). 
     * @return {@code int} value representing the ratio (in per cent) of the number
     * of samples used for training to the number of all samples.
     */
    public int getTrainSamplesRatio() {
        return options.trainSamplesRatio;
    }
    
    /** Ratio of the samples used for validation of the training efficiency 
     * with respect to the number of all samples (in per cent). 
     * @return {@code int} value representing the ratio (in per cent) of the number
     * of samples used for validation to the number of all samples.
     */
    public int getValidationSamplesRatio() {
        return options.validationSamplesRatio;
    }
    
    /** Ratio of the samples used for testing of the training results
     * with respect to the number of all samples (in per cent). 
     * @return {@code int} value representing the ratio (in per cent) of the number
     * of samples used for testing to the number of all samples.
     */
    public int getTestSamplesRatio() {
        return options.testSamplesRatio;
    }
    
    /**
     * Register a listener that will be called when certain 
     * training lifecycle events occur.
     * @param listener A listener that will be called when certain 
     * training lifecycle events occur.
     * @throws NullPointerException if the {@link listener} is null.
     */
    public void registerListener(Listener listener) {
        if (listener == null) {
            throw new NullPointerException("Listener cannot be null");
        }
        listeners.add(listener);
    }
    
    /**
     * Remove the {@link listener}, so that its methods will not be called 
     * when any training lifecycle events occur. If the {@link listener} hasn't
     * been registered, this method is equivalent to no-op.
     * @param listener A listener to be removed from receiving any further 
     * training lifecycle events.
     * @throws NullPointerException if the {@link listener} is null.
     */
    public void removeListener(Listener listener) {
        if (listener == null) {
            throw new NullPointerException("Listener cannot be null");
        }
        listeners.remove(listener);
    }
    
    /**
     * Inform all registered listeners that the training has been complete.
     * @param event {@code TrainerEvent} instance holding the information about
     * the optimization at the time the event occurred.
     */
    private void notifyTrainingComplete(final TrainerEvent event) {
        listeners.forEach((listener) -> {
            listener.onTrainingComplete(event);
        });
    }
    
    /**
     * Inform all registered listeners that the training has been canceled.
     * @param event {@code TrainerEvent} instance holding the information about
     * the optimization at the time the event occurred.
     */
    private void notifyTrainingCanceled(final TrainerEvent event) {
        listeners.forEach((listener) -> {
            listener.onTrainingCanceled(event);
        });
    }
    
    /**
     * Inform all registered listeners that a training epoch has been complete.
     * @param event {@code TrainerEvent} instance holding the information about
     * the optimization at the time the event occurred.
     */
    private void notifyTrainingEpochComplete(final TrainerEvent event) {
        listeners.forEach((listener) -> {
            listener.onTrainingEpochComplete(event);
        });
    }
    
    /** 
     * Request currently running training (if any) to stop, which will result into
     * the currently running training being canceled and the corresponding lifecycle event
     * will be raised.
     */
    public void stopTraining() {
        if (currentTraining != null && !currentTraining.isDone()) {
            currentTraining.cancel(true);
        }
    }
    
    /**
     * Whether the last training (if any) has already finished normally or via cancellation
     * @return {@code true} if there hasn't been a training started yet or if the
     * last training has been finished (either completed or canceled), @{code false} otherwise
     */
    public boolean trainingFinished() {
        return (currentTraining == null || currentTraining.isDone());
    }
    
    /**
     * Map inputs to a predefined range. The {@code inputs} contain 
     * the new normalized values after the call has finished.
     * @param inputs Inputs to be normalized.
     */
    void normalizeSamples(double[][] inputs) {
        if (inputs == null) {
            throw new NullPointerException("Inputs cannot be null");
        }
        sampleNormalizer.normalize(inputs);
    }
    
    /**
     * Map targets to a predefined range. The {@code targets} contain 
     * the new normalized values after the call has finished.
     * @param targets Targets to be normalized.
     */
    void normalizeTargets(double[][] targets) {
        if (targets == null) {
            throw new NullPointerException("Targets cannot be null");
        }
        targetNormalizer.normalize(targets);
    }
    
    /**
     * Start (asynchronously) training of {@link nn} neural network on {@link inputs} samples with
     * {@link targets} results for them.
     * <p>The training will attempt to iteratively change the weights and biases of a copy of {@link nn} 
     * so that the new neural network will provide response on the provided {@link inputs}
     * as close as possible to the provided {@link targets}.</p>
     * <p>The resulting trained neural network can be retrieved with the {@code getTrainedNetwork()} method's call</>
     * @param nn NeuralNetwork object to train
     * @param inputs Samples to use in training as inputs
     * @param targets Desired (ideal) responses of {@link nn} neural network on {@link inputs}
     */
    public void startTrain(NeuralNetwork nn, double[][] inputs, double[][] targets) {
        checkStartTrainArguments(nn, inputs, targets);
        
        stopTraining();
        
        normalizeSamples(inputs);
        normalizeTargets(targets);
        
        NeuralNetworkTrainerTask newTraining = new NeuralNetworkTrainerTask(
                nn, inputs, targets, options);
        newTraining.setListener(trainingListener);
        this.currentTraining = newTraining;
        
        executor.submit(currentTraining);
    }
    
    private void checkStartTrainArguments(NeuralNetwork nn, double[][] inputs, double[][] targets) {
        if (nn == null || inputs == null || targets == null) {
            throw new NullPointerException("Arguments cannot be null");
        }
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Number of input samples and targets must be equal");
        }
        for (int sampleNum = 0; sampleNum < inputs.length; sampleNum++) {
            if (inputs[sampleNum].length != nn.getNumberInputs()) {
                throw new IllegalArgumentException("Size of an input must be equal to the number of inputs of the neural network");
            }
            if (targets[sampleNum].length != nn.getNumberOutputs()) {
                throw new IllegalArgumentException("Size of a target must be equal to the number of outputs of the neural network");
            }
        }
    }
    
    /** 
     * Get the result of the last training.
     * <p>Blocking (indefinitely) function to get the result of the last training.</p>
     * @return A new {@code NeuralNetwork} object representing the trained network or null
     * if there has been no training performed or the last training has been canceled
     * or resulted in an exception.
     */
    public NeuralNetwork getTrainedNetwork() {
        if (currentTraining == null) {
            return null;
        }
        try {
            return currentTraining.get();
        }
        catch(CancellationException | ExecutionException | InterruptedException e) {
            return null;
        }
    }
    
    /**
     * Compute a numerical value representing the difference between the real response
     * of neural network on the provided input and the provided target response.
     * @param nn {@code NeuralNetwork} object which response on {@link sample} input is
     * compared with the ideal response ({@link target}).
     * @param sample The input (values in {@literal [0,1])) to the {@link nn} 
     * to get the real response on.
     * @param target The desired (ideal) response (values in {@literal [0,1])) that the {@link nn} should give 
     * for {@link sample} input.
     * @return A non-negative decimal value representing the difference between the 
     * {@link nn}'s real response on {@link sample} and the desired response ({@link target}).
     * The smaller the value, the closer the real response to the {@link target}. 
     * If the {@link target} contains values outside of [0,1], then the 
     * returned value may be less than zero.
     */
    public static double error(NeuralNetwork nn, double[] sample, double[] target) {
        NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(nn);
        
        double[] response = evaluator.getOutput(sample);
        return error(response, target);
    }
            
    /** Cross-entropy error function. Arguments must be in [0,1] **/
    static double error(double[] actual, double[] expected) {
        if (actual.length != expected.length) {
            throw new IllegalArgumentException("Arguments must be of same size");
        }
        double error = 0;
        final double offset = 1e-15;
        for (int i = 0; i < actual.length; i++) {
            error += -expected[i]*Math.log(offset + actual[i]) - 
                    (1-expected[i])*Math.log(offset + 1 - actual[i]);
        }
        error /= actual.length;
        return error;
    }
}


