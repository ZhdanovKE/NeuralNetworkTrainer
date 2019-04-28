package neuralnetwork;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.Collectors;

/**
 * Neural network with one input layer, at least one hidden layer and one output layer.
 * Inputs and outputs are {@code double}-valued. Activation function is the same for all neurons.
 * 
 * @author Konstantin Zhdanov
 */
public class NeuralNetwork implements Serializable {
    
    private static final long serialVersionUID = 1530372672624601654L;

    /** Number of input neurons **/
    private final int nInputs;
    
    /** Number of hidden layers of neurons **/
    private final int nHiddenLayers;
    
    /** Sizes of the hidden neurons layers **/
    private final int[] hiddenLayerSizes;
    
    /** Number of output neurons **/
    private final int nOutputs;
    
    /** Weights of neurons-to-neuron connections **/
    private double[][][] weights;
    
    /** Biases of neurons **/
    private double[][] biases;
    
    /** Neuron's activation function **/
    private ActivationFunction activationFcn;
    
    /** 
     * Create a {@code NeuralNetwork} instance with the provided sizes of layers
     * and fill them with the default value of 0 and the default activation function 
     * (Sigmoid).
     * @param numInputs The number of inputs of the neural network.
     * @param hiddenLayerSizes The sizes of the hidden layers of the neural network.
     * @param numOutputs The number of outputs of the neural network.
     * @throws IllegalArgumentException if any provided numerical value is zero or
     * negative.
     */
    public NeuralNetwork(int numInputs, int[] hiddenLayerSizes, int numOutputs) {
        if (numInputs <= 0 || hiddenLayerSizes.length <= 0 || numOutputs <= 0) {
            throw new IllegalArgumentException("All numbers must be positive");
        }
        for (int size : hiddenLayerSizes) {
            if (size <= 0) {
                throw new IllegalArgumentException("All numbers must be positive");
            }
        }
        
        nInputs = numInputs;
        nHiddenLayers = hiddenLayerSizes.length;
        this.hiddenLayerSizes = new int[nHiddenLayers];
        System.arraycopy(hiddenLayerSizes, 0, this.hiddenLayerSizes, 0, nHiddenLayers);
        nOutputs = numOutputs;
        
        activationFcn = ActivationFunctions.SIGMOID;
        
        init();
    }
    
    /**
     * Copy constructor.
     * @param nn a source {@code NeuralNetwork} to be copied into a newly created
     * one.
     */
    public NeuralNetwork(NeuralNetwork nn) {
        if (nn == null) {
            throw new NullPointerException("Source neural network cannot be null");
        }
        
        this.nInputs = nn.getNumberInputs();
        this.nOutputs = nn.getNumberOutputs();
        this.nHiddenLayers = nn.getNumberHiddenLayers();
        this.hiddenLayerSizes = nn.getHiddenLayerSizes().clone();
      
        this.activationFcn = nn.getActivationFunction();
        
        init();
        
        for (int i = 0; i < this.weights.length; i++) {
            for (int j = 0; j < this.weights[i].length; j++) {
                for (int k = 0; k < this.weights[i][j].length; k++) {
                    this.weights[i][j][k] = nn.getWeight(i, k, j);
                }
            }
        }
        for (int i = 0; i < this.biases.length; i++) {
            for (int j = 0; j < this.biases[i].length; j++) {
                this.biases[i][j] = nn.getBias(i, j);
            }
        }
    }
    
    /**
     * Allocate memory for the weights and biases according to the size of the network
     * and fill them with 0.
     */
    private void init() {
        initWeights();
        initBiases();
    }
    
    /**
     * Allocate memory for the weights according to the size of the network
     * and fill them with 0.
     */
    private void initWeights() {
        weights = new double[nHiddenLayers + 1][][];
        // Input layer <-> 1st hidden layer
        int curLayerNum = 0;
        int prevLayerSize = nInputs;
        int curLayerSize = hiddenLayerSizes[0];
        weights[curLayerNum] = initLayer(curLayerSize, prevLayerSize);
        prevLayerSize = curLayerSize;
        
        // Hidden layer <-> hidden layer
        for (curLayerNum = 1; curLayerNum < nHiddenLayers; curLayerNum++) {
            curLayerSize = hiddenLayerSizes[curLayerNum];
            weights[curLayerNum] = initLayer(curLayerSize, prevLayerSize);
            prevLayerSize = curLayerSize;
        }
        
        // Last hidden layer <-> Output layer
        curLayerSize = nOutputs;
        curLayerNum = nHiddenLayers;
        weights[curLayerNum] = initLayer(curLayerSize, prevLayerSize);
    }
    
    /**
     * Allocate memory for the layer's weights according to the size of
     * the previous layer and return the allocated weights.
     * @param layerSize The size of the current layer.
     * @param prevLayerSize The size of the previous layer.
     * @return an allocated 2-D {@code double} array of weights between the 
     * current layer and the previous layer.
     */
    private double[][] initLayer(int layerSize, int prevLayerSize) {
        double[][] layerWeights = new double[layerSize][];
        
        for (int layerNeuron = 0; layerNeuron < layerSize; layerNeuron++) {
            layerWeights[layerNeuron] = new double[prevLayerSize];
        }
        
        return layerWeights;
    }
    
    /**
     * Allocate memory for the biases according to the size of the network
     * and fill them with 0.
     */
    private void initBiases() {
        biases = new double[nHiddenLayers + 1][];
        for (int i = 0; i < nHiddenLayers; i++) {
            biases[i] = new double[hiddenLayerSizes[i]];
        }
        biases[nHiddenLayers] = new double[nOutputs];
    }
    
    /**
     * Number of this network's inputs
     * @return an {@code int} value of the number of the inputs
     */
    public int getNumberInputs() {
        return nInputs;
    }
    
    /**
     * Number of this network's hidden layers
     * @return an {@code int} value of the number of the hidden layers
     */
    public int getNumberHiddenLayers() {
        return nHiddenLayers;
    }
    
    /**
     * Size of this network's hidden layer at index {@link layerNum}.
     * @param layerNum an {@code} index of the hidden layer (starting with 0).
     * @return an {@code int} value of the size of the hidden layer with number
     * {@link layerNum}
     * @throws IndexOutOfBoundsException if {@link layerNum} is out of bounds
     */
    public int getHiddenLayerSize(int layerNum) {
        if (layerNum < 0 || layerNum >= nHiddenLayers) {
            throw new IndexOutOfBoundsException("Hidden layer index is out of bounds");
        }
        return hiddenLayerSizes[layerNum];
    }
    
    /**
     * Number of this network's hidden layers.
     * @return an {@code int} value of the number of the hidden layers
     */
    public int[] getHiddenLayerSizes() {
        if (hiddenLayerSizes == null) {
            return null;
        }
        return hiddenLayerSizes.clone();
    }
    
    /**
     * Number of this network's outputs
     * @return an {@code int} value of the number of the outputs
     */
    public int getNumberOutputs() {
        return nOutputs;
    }
    
    /**
     * The weight of the connection between the neuron with index {@link fromNeuronNum}
     * at the layer with index {@link layerNum} - 1
     * and the neuron with index {@link toNeuronNum} at the layer with index {@link layerNum}.
     * @param layerNum The layer's index where the neuron {@link toNeuronNum} is.
     * @param fromNeuronNum The index of a neuron at the layer {@link layerNum} - 1.
     * @param toNeuronNum The index of a neuron at the layer {@link layerNum}.
     * @return {@code double} value of the weight between the corresponding neurons.
     * @throws IndexOutOfBoundsException if any of the passed indices are out of bounds
     */
    public double getWeight(int layerNum, int fromNeuronNum, int toNeuronNum) {
        try {
            return weights[layerNum][toNeuronNum][fromNeuronNum];
        }
        catch (ArrayIndexOutOfBoundsException e) {
            throw new IndexOutOfBoundsException("Some of the indices are out of bounds");
        }
    }
    
    /**
     * Set the weight of the connection between the neuron with index {@link fromNeuronNum}
     * at the layer with index {@link layerNum} - 1
     * and the neuron with index {@link toNeuronNum} at the layer with index {@link layerNum}.
     * @param layerNum The layer's index where the neuron {@link toNeuronNum} is.
     * @param fromNeuronNum The index of a neuron at the layer {@link layerNum} - 1.
     * @param toNeuronNum The index of a neuron at the layer {@link layerNum}.
     * @param weight The new weight to be set between the neurons {@link fromNeuronNum}
     * and {@link toNeuronNum}.
     * @throws IndexOutOfBoundsException if any of the passed indices are out of bounds
     */
    public void setWeight(int layerNum, int fromNeuronNum, int toNeuronNum, double weight) {
        try {
            weights[layerNum][toNeuronNum][fromNeuronNum] = weight;
        }
        catch (ArrayIndexOutOfBoundsException e) {
            throw new IndexOutOfBoundsException("Some of the indices are out of bounds");
        }
    }
    
    /**
     * Set the same weight for all connections between the neurons of this network.
     * @param weight The new weight to be set for all neurons.
     */
    public void setWeights(double weight) {
        for (double[][] weightsForLayer : weights) {
            for (double[] weightsForNeuron : weightsForLayer) {
                for (int prevLayerNeuronNum = 0; 
                        prevLayerNeuronNum < weightsForNeuron.length; 
                        prevLayerNeuronNum++) {
                    weightsForNeuron[prevLayerNeuronNum] = weight;
                }
            }
        }
    }
    
    
    public double getBias(int layerNum, int neuronNum) {
        try {
            return biases[layerNum][neuronNum];
        }
        catch (ArrayIndexOutOfBoundsException e) {
            throw new IndexOutOfBoundsException("Some of the indices are out of bounds");
        }
    }
    
    public void setBias(int layerNum, int neuronNum, double bias) {
        try {
            biases[layerNum][neuronNum] = bias;
        }
        catch (ArrayIndexOutOfBoundsException e) {
            throw new IndexOutOfBoundsException("Some of the indices are out of bounds");
        }
    }
    
    /**
     * Set the same bias for all neurons of this network.
     * @param bias The new bias to be set for all neurons.
     */
    public void setBiases(double bias) {
        for (double[] biasesForLayer : biases) {
            for (int neuronNum = 0; neuronNum < biasesForLayer.length; neuronNum++) {
                biasesForLayer[neuronNum] = bias;
            }
        }
    }
    
    /**
     * The activation function used for every neuron of this network.
     * @return An instance of {@code ActivationFunction} representing
     * the activation function of every neuron.
     */
    public ActivationFunction getActivationFunction() {
        return activationFcn;
    }
    
    /**
     * Set the activation function used for every neuron of this network.
     * @param fcn An instance of {@code ActivationFunction} to be used
     * as the activation function in this network.
     * @throws NullPointerException If the passed {@link fcn} is null.
     */
    public void setActivationFunction(ActivationFunction fcn) {
        if (fcn == null) {
            throw new NullPointerException("Activation function is null");
        }
        activationFcn = fcn;
    }
    
    /**
     * Convert to {@code String} representing the structure of this network: 
     * the number of inputs, the sizes of each hidden layer and the number of outputs.
     * @return {@code String} representing the structure of this network
     */
    @Override
    public String toString() {
        final String delimeter = ", ";
        StringBuilder sb = new StringBuilder();
        sb.append("(");
        sb.append(nInputs);
        sb.append(delimeter);
        sb.append(Arrays.stream(hiddenLayerSizes).mapToObj(String::valueOf).
                collect(Collectors.joining(delimeter)));
        sb.append(delimeter);
        sb.append(nOutputs);
        sb.append(")");
        return sb.toString();
    }
}
