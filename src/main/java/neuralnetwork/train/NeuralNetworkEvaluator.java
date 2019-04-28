package neuralnetwork.train;

import neuralnetwork.NeuralNetwork;

/**
 * A class for evaluating a neural network's response on a given input.
 * @author Konstantin Zhdanov
 */
public class NeuralNetworkEvaluator {
       
    private NeuralNetwork nn;
    
    public NeuralNetworkEvaluator(NeuralNetwork nn) {
        if (nn == null) {
            throw new NullPointerException("Neural network cannot be null");
        }
        this.nn = nn;
    }

    private double[] evaluateActivationFcn(double[] inputSum) {
        double[] outputs = new double[inputSum.length];
        for (int curLayerNeuron = 0; curLayerNeuron < inputSum.length; curLayerNeuron++) {
            outputs[curLayerNeuron] = nn.getActivationFunction().
                    valueAt(inputSum[curLayerNeuron]);
        }

        return outputs;
    }
    
    public double[] getOutput(double[] input) {
        if (input == null) {
            throw new NullPointerException("Input cannot be null");
        }
        if (input.length != nn.getNumberInputs()) {
            throw new IllegalArgumentException("Input must be "
                    + "the same size as the number of inputs of the neural network");
        }
        
        NeuralNetworkWeights weights = NeuralNetworkWeights.newOf(nn);
        
        int curLayerNum = 0;
        double[] prevLayerResults = input;
        int curLayerSize = nn.getHiddenLayerSize(curLayerNum);
        
        // Input -> first hidden layer
        prevLayerResults = nextLayerInputSummedWithWeights(prevLayerResults, curLayerSize, curLayerNum, weights);
        prevLayerResults = evaluateActivationFcn(prevLayerResults);

        // Between hidden layers
        for (curLayerNum = 1; curLayerNum < nn.getNumberHiddenLayers(); curLayerNum++) {
            curLayerSize = nn.getHiddenLayerSize(curLayerNum);
            prevLayerResults = nextLayerInputSummedWithWeights(prevLayerResults, curLayerSize, curLayerNum, weights);
            prevLayerResults = evaluateActivationFcn(prevLayerResults);
        }
        
        // Last hidden layer -> output
        curLayerSize = nn.getNumberOutputs();
        curLayerNum = nn.getNumberHiddenLayers();
        prevLayerResults = nextLayerInputSummedWithWeights(prevLayerResults, curLayerSize, curLayerNum, weights);
        prevLayerResults = evaluateActivationFcn(prevLayerResults);

        return prevLayerResults;
    }
    
    NeuralNetworkResponse evaluate(double[] input) {
        if (input == null) {
            throw new NullPointerException("Input must be non-null");
        }
        NeuralNetworkWeights weights = NeuralNetworkWeights.newOf(nn);
        return evaluateWithWeights(input, weights);
    }
    
    NeuralNetworkResponse evaluateWithWeights(double[] input, NeuralNetworkWeights nnWeights) {
        if (input == null) {
            throw new NullPointerException("Input cannot be null");
        }
        if (nnWeights == null) {
            throw new NullPointerException("Weights cannot be null");
        }
        double[][] neuronsInputSummed = new double[nn.getNumberHiddenLayers() + 1][];
        double[][] neuronsOutputs = new double[neuronsInputSummed.length][];

        // Input -> first hidden layer
        int curLayerNum = 0;
        int curLayerSize = nn.getHiddenLayerSize(curLayerNum);
        neuronsInputSummed[curLayerNum] = nextLayerInputSummedWithWeights(input, curLayerSize, curLayerNum, nnWeights);
        neuronsOutputs[curLayerNum] = evaluateActivationFcn(neuronsInputSummed[curLayerNum]);

        // Between hidden layers
        for (curLayerNum = 1; curLayerNum < nn.getNumberHiddenLayers(); curLayerNum++) {
            curLayerSize = nn.getHiddenLayerSize(curLayerNum);
            neuronsInputSummed[curLayerNum] = nextLayerInputSummedWithWeights(neuronsOutputs[curLayerNum - 1], curLayerSize, curLayerNum, nnWeights);
            neuronsOutputs[curLayerNum] = evaluateActivationFcn(neuronsInputSummed[curLayerNum]);
        }

        // Last hidden layer -> output
        curLayerSize = nn.getNumberOutputs();
        curLayerNum = nn.getNumberHiddenLayers();
        neuronsInputSummed[curLayerNum] = nextLayerInputSummedWithWeights(neuronsOutputs[curLayerNum - 1], curLayerSize, curLayerNum, nnWeights);
        neuronsOutputs[curLayerNum] = evaluateActivationFcn(neuronsInputSummed[curLayerNum]);

        NeuralNetworkResponse resp = new NeuralNetworkResponse(neuronsInputSummed, neuronsOutputs);

        return resp;
    }
    
    private double[] nextLayerInputSummedWithWeights(double[] prevLayerResults, int currentLayerSize, int curLayerNum, NeuralNetworkWeights nnWeights) {
        double[] currentLayerResults = new double[currentLayerSize];
        
        for (int curLayerNeuron = 0; curLayerNeuron < currentLayerResults.length; curLayerNeuron++) {
            for (int prevLayerNeuron = 0; prevLayerNeuron < prevLayerResults.length; prevLayerNeuron++) {
                currentLayerResults[curLayerNeuron] += 
                        nnWeights.getWeight(curLayerNum, prevLayerNeuron, curLayerNeuron)*
                        prevLayerResults[prevLayerNeuron];
            }
            currentLayerResults[curLayerNeuron] += nnWeights.getBias(curLayerNum, curLayerNeuron);

        }
        return currentLayerResults;
    }
    
    NeuralNetworkWeights weightsDerivative(double[] input, double[] errors) {
        NeuralNetworkResponse resp = evaluate(input);
        return weightsDerivative(input, errors, resp);
    }
    
    NeuralNetworkWeights weightsDerivative(double[] input, 
                                                double[] targets, 
                                                NeuralNetworkResponse response) {
        NeuralNetworkWeights weights = NeuralNetworkWeights.newOf(nn);
        return weightsDerivative(input, targets, response, weights);
    }
    
    /**
     * <p>Computes derivative of the weights for the Cross-entropy error function<p>
     * @param input Neural network's input
     * @param targets Target outputs for the provided input
     * @param response NeuralNetworkResponse object holding inputs and outputs 
     * for every neuron in the neural network for the provided {@link input}
     * @param weights NeuralNetworkWeights object for which to compute the derivatives
     * @return Object holding the derivative of weights and biases of the neural network
     */
    NeuralNetworkWeights weightsDerivative(double[] input, 
                                                double[] targets, 
                                                NeuralNetworkResponse response, 
                                                NeuralNetworkWeights weights) {
        
        // Compute errors - differences between target and output
        double[] errors = new double[targets.length];
        for (int i = 0; i < errors.length; i++) {
            errors[i] = response.getOutput(i) - targets[i];
        }
        
        // Backpropagate errors
        double[][] deltas = computeDeltas(errors, response, weights);

        // Propagate deltas forward
        NeuralNetworkWeights nnwDerivs = propagateDeltasForward(deltas, input, response);
        
        return nnwDerivs;
        
    }
    
    private double[][] computeDeltas(
            double[] errors, 
            NeuralNetworkResponse response, 
            NeuralNetworkWeights weights) {
        
        int nHiddenLayers = nn.getNumberHiddenLayers();
        double[][] deltas = new double[nHiddenLayers + 1][];

        // Propagate errors backwards

        // Output layer
        int nOutputs = nn.getNumberOutputs();
        deltas[nHiddenLayers] = new double[nOutputs];
        // Valid only for cross-entropy error function
        System.arraycopy(errors, 0, deltas[nHiddenLayers], 0, nOutputs);

        // Hidden layers
        int curLayerNum = nHiddenLayers - 1;
        int nextLayerNum = curLayerNum + 1;
        deltas[curLayerNum] = new double[nn.getHiddenLayerSize(curLayerNum)];
        for (int curLayerNeuron = 0; curLayerNeuron < nn.getHiddenLayerSize(curLayerNum); curLayerNeuron++) {
            for (int nextLayerNeuron = 0; nextLayerNeuron < nOutputs; nextLayerNeuron++) {
                deltas[curLayerNum][curLayerNeuron] += 
                        deltas[nextLayerNum][nextLayerNeuron]*
                        weights.getWeight(nextLayerNum, curLayerNeuron, nextLayerNeuron)*
                        nn.getActivationFunction().derivativeValueAt(
                                response.getNeuronInputSum(curLayerNum, curLayerNeuron));
            } 
        }

        // Input layer
        for (curLayerNum = nHiddenLayers - 2; curLayerNum >= 0; curLayerNum--) {
            deltas[curLayerNum] = new double[nn.getHiddenLayerSize(curLayerNum)];
            nextLayerNum = curLayerNum + 1;
            for (int curLayerNeuron = 0; curLayerNeuron < nn.getHiddenLayerSize(curLayerNum); curLayerNeuron++) {
                for (int nextLayerNeuron = 0; nextLayerNeuron < nn.getHiddenLayerSize(curLayerNum + 1); nextLayerNeuron++) {
                    deltas[curLayerNum][curLayerNeuron] += 
                            deltas[nextLayerNum][nextLayerNeuron]*
                            weights.getWeight(nextLayerNum, curLayerNeuron, nextLayerNeuron)*
                            nn.getActivationFunction().derivativeValueAt(
                                    response.getNeuronInputSum(curLayerNum, curLayerNeuron));
                } 
            }
        }
        
        return deltas;
    }
    
    private NeuralNetworkWeights propagateDeltasForward(
            double[][] deltas, 
            double[] input, 
            NeuralNetworkResponse response) {
        
        NeuralNetworkWeights nnwDerivs = new NeuralNetworkWeights(
                nn.getNumberInputs(), 
                nn.getHiddenLayerSizes(), 
                nn.getNumberOutputs()
        );
        // Input layer
        int nHiddenLayers = nn.getNumberHiddenLayers();
        int nOutputs = nn.getNumberOutputs();
        int curLayerNum = 0;
        int curLayerSize = nn.getHiddenLayerSize(curLayerNum);
        int prevLayerSize = nn.getNumberInputs();
        for (int curLayerNeuron = 0; curLayerNeuron < curLayerSize; curLayerNeuron++) {
            for (int prevLayerNeuron = 0; prevLayerNeuron < prevLayerSize; prevLayerNeuron++) {
                nnwDerivs.setWeight(curLayerNum, prevLayerNeuron, curLayerNeuron,  
                        deltas[curLayerNum][curLayerNeuron]*
                        input[prevLayerNeuron]
                );
            } 
            nnwDerivs.setBias(curLayerNum, curLayerNeuron, deltas[curLayerNum][curLayerNeuron]);
        }

        // Hidden layers
        for (curLayerNum = 1; curLayerNum < nHiddenLayers; curLayerNum++) {
            curLayerSize = nn.getHiddenLayerSize(curLayerNum);
            prevLayerSize = nn.getHiddenLayerSize(curLayerNum - 1);
            for (int curLayerNeuron = 0; curLayerNeuron < curLayerSize; curLayerNeuron++) {
                for (int prevLayerNeuron = 0; prevLayerNeuron < prevLayerSize; prevLayerNeuron++) {
                    nnwDerivs.setWeight(curLayerNum, prevLayerNeuron, curLayerNeuron, 
                            deltas[curLayerNum][curLayerNeuron]*
                            response.getNeuronOutput(curLayerNum - 1, prevLayerNeuron)
                    );
                }
                nnwDerivs.setBias(curLayerNum, curLayerNeuron, deltas[curLayerNum][curLayerNeuron]);
            }
        }

        // Output layer
        curLayerNum = nHiddenLayers;
        curLayerSize = nOutputs;
        prevLayerSize = nn.getHiddenLayerSize(curLayerNum - 1);
        for (int curLayerNeuron = 0; curLayerNeuron < curLayerSize; curLayerNeuron++) {
            for (int prevLayerNeuron = 0; prevLayerNeuron < prevLayerSize; prevLayerNeuron++) {
                nnwDerivs.setWeight(curLayerNum, prevLayerNeuron, curLayerNeuron,
                        deltas[curLayerNum][curLayerNeuron]*
                        response.getNeuronOutput(curLayerNum - 1, prevLayerNeuron)
                );
            } 
            nnwDerivs.setBias(curLayerNum, curLayerNeuron, deltas[curLayerNum][curLayerNeuron]);
        }

        return nnwDerivs;
    }
}


