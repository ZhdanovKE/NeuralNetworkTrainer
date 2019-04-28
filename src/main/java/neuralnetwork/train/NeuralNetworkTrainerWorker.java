package neuralnetwork.train;

import neuralnetwork.NeuralNetwork;
import neuralnetwork.util.Randomizer;
import java.util.concurrent.Callable;

/**
 * An implementation of the {@code Callable<NeuralNetwork>} interface, 
 * representing a training task that will be performed in a separate thread.
 * A neural network passed to the constructor is copied and the resulting
 * network is returned as a copy so that these networks can be safely used in
 * a multi-threaded environment.
 * @author Konstantin Zhdanov
 */
class NeuralNetworkTrainerWorker implements Callable<NeuralNetwork> {
    
    /** {@code NeuralNetwork} to train. */
    private final NeuralNetwork nn;
        
    /** Target outputs of the network. */
    private final double[][] expectedOutputs;
    
    /** Indices of samples used for training, validation and testing. */
    private int[] trainSamplesIndices;
    private int[] validationSamplesIndices;
    private int[] testSamplesIndices;
    
    /** Arrays of samples used for training, validation and testing. */
    private double[][] trainSamples;
    private double[][] validationSamples;
    private double[][] testSamples;
    
    /** 
     * Ratio of the number of samples used for training with respect to the 
     * number of all samples (in per cent).
     */
    private final int trainSamplesRatio;
    
    /** 
     * Ratio of the number of samples used for validation with respect to 
     * the number of all samples (in per cent).
     */
    private final int validationSamplesRatio;
    
    /** 
     * Ratio of the number of samples used for testing with respect to 
     * the number of all samples (in per cent).
     */
    private final int testSamplesRatio;
    
    /** 
     * The value of the training's error which signalizes that the training
     * should stop. 
     */
    private final double performanceGoal;
    
    /** 
     * Max number of the training's iterations which signalizes that the training
     * should stop. 
     */
    private final int maxEpoch;
    
    /** Parameters used for the SCG training algorithm. */
    private final double minGradient = 0.0000001;
    private final double lambdaInit = 0.0000001;
    private final double sigma = 0.00001;
    
    /** 
     * A training events listener attached to this worker.
     * This listener will be called when a training iteration is finished, 
     * the training is finished and is canceled.
     */
    private volatile Listener listener = null;
    
    /**
     * Whether this training has been canceled by user or by an error.
     */
    private boolean cancelled = false;

    /**
     * Create a training task.
     * @param nn a network to be trained.
     * @param inputs an array of network's inputs to perform the training upon. 
     * Inputs are assumed to be in a valid range [0,1] for the training algorithm.
     * @param targets an array of network's ideal responses to the provided {@link inputs} 
     * to perform the training upon. 
     * Targets are assumed to be in a valid range [0,1] for the training algorithm.
     * @param opts An object containing the chosen options of the 
     * training algorithm.
     */
    public NeuralNetworkTrainerWorker(NeuralNetwork nn, double[][] inputs, double[][] targets, NeuralNetworkTrainer.Options opts) {
        this.nn = new NeuralNetwork(nn); // defensive copy

        this.expectedOutputs = targets.clone();
        for (int i = 0; i < targets.length; i++) {
            this.expectedOutputs[i] = targets[i].clone();
        }
        
        this.maxEpoch = opts.maxEpoch;
        this.performanceGoal = opts.performanceGoal;
        this.trainSamplesRatio = opts.trainSamplesRatio;
        this.validationSamplesRatio = opts.validationSamplesRatio;
        this.testSamplesRatio = opts.testSamplesRatio;
        
        splitSamplesIntoGroups(inputs);
    }
    
    /** 
     * Set the training events listener to be called every time a training event
     * occurs: a training iteration is finished, 
     * the training is finished and is canceled.
     * @param listener a {@code Listener} to attach to this worker.
     */
    public void setListener(Listener listener) {
        this.listener = listener;
    }
    
    /**
     * Called when the training has been complete.
     * @param event Information about the event.
     */
    protected void onTrainingComplete(TrainerEvent event) {
        Listener snapshotListener = listener;
        if (snapshotListener != null) {
            snapshotListener.onTrainingComplete(event);
        }
    }
    
    /**
     * Called when the training has been canceled.
     * @param event Information about the event.
     */
    protected void onTrainingCanceled(TrainerEvent event) {
        Listener snapshotListener = listener;
        if (snapshotListener != null) {
            snapshotListener.onTrainingCanceled(event);
        }
    }
    
    /**
     * Called when one training iteration (epoch) has been finished.
     * @param event Information about the event.
     */
    protected void onTrainingEpochComplete(TrainerEvent event) {
        Listener snapshotListener = listener;
        if (snapshotListener != null) {
            snapshotListener.onTrainingEpochComplete(event);
        }
    }
    
    @Override
    public NeuralNetwork call() throws Exception {
        train();
        return new NeuralNetwork(nn); // defensive copy
    }
    
    private void notifyFinalStatus(int epochNum, double performance) {
        TrainerEvent event = new TrainerEvent(epochNum, performance);
        if (cancelled) {
            onTrainingCanceled(event);
        }
        else {
            onTrainingComplete(event);
        }
    }
    
    private void notifyEpochComplete(int epochNum, double performance) {
        TrainerEvent event = new TrainerEvent(epochNum + 1, performance);
        onTrainingEpochComplete(event);
    }
    
    private static class TrainStepResults {
        NeuralNetworkWeights weightsDerivatives;
        double performance;
    }
    
    /**
     * Get the network that is the result of the training.
     * @return A {@code NeuralNetwork} with weights and biases that has been
     * trained.
     */
    public NeuralNetwork getTrainedNeuralNetwork() {
        if (nn == null) {
            return null;
        }
        return new NeuralNetwork(nn);
    }
    
    /**
     * Check interruption status and clear it.
     * @return {@code boolean} value representing whether this worker
     * has been interrupted and need to be canceled.
     */
    private boolean cancelRequested() {
        return Thread.interrupted();
    }
    
    /**
     * Perform training of the associated network according to an algorithm 
     * created in 
     * "M. F. Meiller. A Scaled Conjugate Gradient Algorithm 
     * for Fast Supervised Learning // Neural Networks, Vol. 6, pp. 525-533, 1993". 
     */
    public void train() {
        double avgPerformance = 0.0;
        int epochNum = 0;
        NeuralNetworkWeights weights = NeuralNetworkWeights.newOf(nn);
        try {
            double lambda = lambdaInit;
            double lambda_sup = 0.0;

            int numConjugateDirections = getNumberOfConjugateDirections();

            NeuralNetworkEvaluator evaluator = new NeuralNetworkEvaluator(nn);

            TrainStepResults trainStepResults = makeStepOverTrainSet(weights, evaluator);
            avgPerformance = trainStepResults.performance;

            if (cancelRequested()) {
                cancelled = true;
                return;
            }

            NeuralNetworkWeights antigradientWeights = new NeuralNetworkWeights(trainStepResults.weightsDerivatives).multiply(-1);
            NeuralNetworkWeights conjugateWeights = new NeuralNetworkWeights(antigradientWeights);
            NeuralNetworkWeights nextWeights;
            NeuralNetworkWeights s_k_weights;

            boolean success = true;
            double sigma_k;
            double delta_k = 0.0;
            double normConjugate;
            double normGradient;
            for (epochNum = 0; epochNum < maxEpoch; epochNum++) {
                normConjugate = conjugateWeights.norm();
                if (success) {
                    sigma_k = sigma/normConjugate;
                    // Calculate E'(w_k)
                    trainStepResults = makeStepOverTrainSet(weights, evaluator);

                    // Check interruption
                    if (cancelRequested()) {
                        cancelled = true;
                        return;
                    }

                    NeuralNetworkWeights conjugateWeightsCopy = new NeuralNetworkWeights(conjugateWeights);
                    // w_k + sigma_k p_k
                    nextWeights = conjugateWeightsCopy.multiply(sigma_k).add(weights);

                    // Calculate E'(w_k + sigma_k p_k)
                    TrainStepResults nextTrainStepResults = makeStepOverTrainSet(nextWeights, evaluator);

                    // Check interruption
                    if (cancelRequested()) {
                        cancelled = true;
                        return;
                    }

                    // s_k = (E'(w_k + sigma_k p_k) - E'(w_k))/sigma_k
                    s_k_weights = nextTrainStepResults.weightsDerivatives.
                            subtract(trainStepResults.weightsDerivatives).multiply(1/sigma_k);

                    delta_k = conjugateWeights.dot(s_k_weights);
                }
                
                delta_k += (lambda - lambda_sup) * Math.pow(normConjugate,2);

                if (delta_k <= 0) {
                    lambda_sup = 2 * (lambda - delta_k / Math.pow(normConjugate, 2));
                    delta_k = -delta_k + lambda * Math.pow(normConjugate, 2);
                    lambda = lambda_sup;
                }

                // Calculate step size
                double mu = conjugateWeights.dot(antigradientWeights);
                double alpha = mu / delta_k;

                // Calculate comparison parameter
                // nextWeights = weights + alpha*conjugateWeights
                NeuralNetworkWeights conjugateWeightsCopy = new NeuralNetworkWeights(conjugateWeights);
                nextWeights = conjugateWeightsCopy.multiply(alpha).add(weights);

                avgPerformance = performanceOverTrainSetForWeights(weights, evaluator);
                double nextAvgPerformance = performanceOverTrainSetForWeights(nextWeights, evaluator);

                double Delta_k = 2 * delta_k * (avgPerformance - nextAvgPerformance) / 
                        Math.pow(mu,2);
                normGradient = antigradientWeights.norm();

                if (Delta_k >= 0) {
                    // Can make a reduction in error at this step

                    // w_k += alpha * p_k
                    weights = nextWeights;

                    trainStepResults = makeStepOverTrainSet(weights, evaluator);

                    avgPerformance = trainStepResults.performance;

                    // Check interruption
                    if (cancelRequested()) {
                        cancelled = true;
                        return;
                    }

                    NeuralNetworkWeights nextAntigradientWeights = 
                            new NeuralNetworkWeights(trainStepResults.weightsDerivatives).
                                    multiply(-1);

                    lambda_sup = 0;
                    success = true;
                    if ((epochNum + 1) % numConjugateDirections == 0) {
                        // Restarting
                        conjugateWeights = nextAntigradientWeights;
                    }
                    else {
                        // Changing conjugate direction
                        double factor = (Math.pow(nextAntigradientWeights.norm(), 2) - 
                                antigradientWeights.dot(nextAntigradientWeights)) / mu;
                        // p_k = nextAntiGrad + factor*p_k
                        conjugateWeights.multiply(factor).add(nextAntigradientWeights);
                    }
                    antigradientWeights = new NeuralNetworkWeights(nextAntigradientWeights);

                    if (Delta_k >= 0.75) {
                        lambda /= 4;
                    }
                }
                else {
                    // Cannot reduce error any more
                    success = false;
                    lambda_sup = lambda;
                }
                if (Delta_k < 0.25) {
                    lambda += delta_k*(1-Delta_k)/Math.pow(normConjugate, 2);
                }

                notifyEpochComplete(epochNum, avgPerformance);

                if (normGradient <= minGradient) {
                    // Optimum has been found
                    // weigths - solution
                    break;
                }
                if (avgPerformance < performanceGoal) {
                    // Optimum has been found
                    // weigths - solution
                    break;
                }
            }

        }
        finally {
            try {
                weights.applyToNeuralNetwork(nn);
            }
            finally {
                notifyFinalStatus(epochNum, avgPerformance);
            }
        }
    }
    
    private double[] getExpectedOutputForTrainSample(int num) {
        return expectedOutputs[trainSamplesIndices[num]];
    }
    
    private TrainStepResults makeStepOverTrainSet(NeuralNetworkWeights weights, 
                                            NeuralNetworkEvaluator evaluator) {
        NeuralNetworkWeights avgWeightsDerivs = new NeuralNetworkWeights(
                nn.getNumberInputs(), 
                nn.getHiddenLayerSizes(), 
                nn.getNumberOutputs()
        );
        double performance, avgPerformance = 0.0;
        for (int trainSampleNum = 0; trainSampleNum < trainSamples.length; trainSampleNum++) {
            NeuralNetworkResponse trainResp = 
                    evaluator.evaluateWithWeights(trainSamples[trainSampleNum], 
                            weights);
            performance = NeuralNetworkTrainer.error(
                    trainResp.getOutputs(), 
                    getExpectedOutputForTrainSample(trainSampleNum));
            avgPerformance += performance;
           
            NeuralNetworkWeights weightsDerivs = evaluator.weightsDerivative(
                    trainSamples[trainSampleNum], 
                    getExpectedOutputForTrainSample(trainSampleNum), 
                    trainResp,
                    weights
            );
            avgWeightsDerivs.add(weightsDerivs);
        }
        avgPerformance /= trainSamples.length;
           
        avgWeightsDerivs.multiply(1.0/trainSamples.length);
        
        TrainStepResults results = new TrainStepResults();
        results.performance = avgPerformance;
        results.weightsDerivatives = avgWeightsDerivs;
        
        return results;
    }
    
    private double performanceOverTrainSetForWeights(NeuralNetworkWeights weights, 
                                                    NeuralNetworkEvaluator evaluator) {
        
        double avgPerformance = 0;
        double performance;

        for (int trainSampleNum = 0; trainSampleNum < trainSamples.length; trainSampleNum++) {
            NeuralNetworkResponse trainResp = 
                    evaluator.evaluateWithWeights(trainSamples[trainSampleNum], 
                            weights
            );

            performance = NeuralNetworkTrainer.error(
                    trainResp.getOutputs(), 
                    getExpectedOutputForTrainSample(trainSampleNum)
            );
            avgPerformance += performance;
        }
        avgPerformance /= trainSamples.length;
        
        return avgPerformance;
    }
    
    private int getNumberOfConjugateDirections() {
        int numConjugateDirections = nn.getNumberOutputs() * 
                nn.getHiddenLayerSize(nn.getNumberHiddenLayers() - 1) + 
                nn.getNumberOutputs();
        numConjugateDirections += nn.getNumberInputs()*nn.getHiddenLayerSize(0) + 
                nn.getHiddenLayerSize(0);
        for (int layerNum = 1; layerNum < nn.getNumberHiddenLayers(); layerNum++) {
            numConjugateDirections += nn.getHiddenLayerSize(layerNum - 1)*
                    nn.getHiddenLayerSize(layerNum) + 
                    nn.getHiddenLayerSize(layerNum);
        }
        
        return numConjugateDirections;
    }
    
    private void splitSamplesIntoGroups(double[][] samples) {
        initSamplesIndices(samples.length);
        
        trainSamples = copyTrainSamples(samples);
        validationSamples = copyValidationSamples(samples);
        testSamples = copyTestSamples(samples);
    }
    
    private double[][] copyTrainSamples(double[][] samples) {
        return copySubsetOfSamples(samples, trainSamplesIndices);
    }
    
    private double[][] copyValidationSamples(double[][] samples) {
        return copySubsetOfSamples(samples, validationSamplesIndices);
    }
    
    private double[][] copyTestSamples(double[][] samples) {
        return copySubsetOfSamples(samples, testSamplesIndices);
    }
    
    private double[][] copySubsetOfSamples(double[][] samples, int[] indices) {
        double[][] copySubset = new double[indices.length][];
        
        for (int subsetElemNum = 0; subsetElemNum < indices.length; subsetElemNum++) {
            copySubset[subsetElemNum] = samples[indices[subsetElemNum]].clone();
        }
        
        return copySubset;
    }
    
    /**
     * <p>Randomly split array of {@link nSamples} indices of samples into three (disjoint) sets: 
     * training set, validation set and test set. The indices (in the original array)
     * of elements in each created set are randomly picked and saved internally in the object</p>
     * 
     * @param nSamples Number of samples to split into three groups randomly
     */
    private void initSamplesIndices(int nSamples) {
        int[] samplesIndices = new int[nSamples];
        for (int i = 1; i < nSamples; i++) {
            samplesIndices[i] = samplesIndices[i - 1] + 1;
        }
        
        int numTrainSamples = (int)(trainSamplesRatio*nSamples/100.0);
        int numValidationSamples = (int)(validationSamplesRatio*nSamples/100.0);
        int numTestSamples = nSamples - numTrainSamples - numValidationSamples;
        
        Randomizer randomizer = new Randomizer();
        trainSamplesIndices = randomizer.getRandomElements(samplesIndices, 
                numTrainSamples, 0, nSamples);
        
        validationSamplesIndices = randomizer.getRandomElements(samplesIndices, 
                numValidationSamples, numTrainSamples, nSamples);
        
        testSamplesIndices = randomizer.getRandomElements(samplesIndices, 
                numTestSamples, numTrainSamples + numValidationSamples, nSamples);
    }
    
}
