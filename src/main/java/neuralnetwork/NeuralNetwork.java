package neuralnetwork;

import neuralnetwork.init.Initializer;
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
    
    /** Cached value of signature **/
    private transient String signature = null;
    
    /** 
     * Create a {@code NeuralNetwork} instance with the provided sizes of layers
     * and fill them with the default value of 0 and the default activation function 
     * (Sigmoid).
     * @param numInputs The number of inputs of the neural network.
     * @param hiddenLayerSizes The sizes of the hidden layers of the neural network.
     * @param numOutputs The number of outputs of the neural network.
     * @param initializer {@link Initializer} instance used to initialize weights
     * and biases of this network;
     * @throws IllegalArgumentException if any provided numerical value is zero or
     * negative.
     * @throws NullPointerException if {@link hiddenLayerSizes} or 
     * {@code initializer} is null.
     */
    public NeuralNetwork(int numInputs, int[] hiddenLayerSizes, int numOutputs, 
            Initializer initializer) {
        if (numInputs <= 0 || hiddenLayerSizes.length <= 0 || numOutputs <= 0) {
            throw new IllegalArgumentException("All numbers must be positive");
        }
        if (initializer == null) {
            throw new NullPointerException("Initializer cannot be null");
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
        
        init(initializer);
    }
    
    /** 
     * Create a {@code NeuralNetwork} instance with the provided sizes of layers
     * and fill them with the default value of 0 and the default activation function 
     * (Sigmoid).
     * @param numInputs The number of inputs of the neural network.
     * @param hiddenLayerSizes The sizes of the hidden layers of the neural network.
     * @param numOutputs The number of outputs of the neural network.
     * @throws IllegalArgumentException if any provided numerical value is zero or
     * negative.
     * @throws NullPointerException if {@link hiddenLayerSizes} is null.
     */
    public NeuralNetwork(int numInputs, int[] hiddenLayerSizes, int numOutputs) {
        this(numInputs, hiddenLayerSizes, numOutputs, Initializer.of(0.0, 0.0));
    }
    
    /**
     * Copy constructor.
     * @param nn a source {@code NeuralNetwork} to be copied into a newly created
     * one.
     * @throws NullPointerException if {@link nn} is null.
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
        
        init(Initializer.of(0.0, 0.0));
        
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
     * and fill them with values supplied by {@code initializer}.
     */
    private void init(Initializer initializer) {
        initWeights(initializer);
        initBiases(initializer);
    }
    
    /**
     * Allocate memory for the weights according to the size of the network
     * and fill them value supplied by {@code initializer}.
     * @param initializer {@link Initializer} to be used to initialize weights.
     */
    private void initWeights(Initializer initializer) {
        weights = new double[nHiddenLayers + 1][][];
        // Input layer <-> 1st hidden layer
        int curLayerNum = 0;
        int prevLayerSize = nInputs;
        int curLayerSize = hiddenLayerSizes[0];
        weights[curLayerNum] = initLayer(curLayerNum, curLayerSize, prevLayerSize,
                    initializer);
        prevLayerSize = curLayerSize;
        
        // Hidden layer <-> hidden layer
        for (curLayerNum = 1; curLayerNum < nHiddenLayers; curLayerNum++) {
            curLayerSize = hiddenLayerSizes[curLayerNum];
            weights[curLayerNum] = initLayer(curLayerNum, curLayerSize, prevLayerSize,
                    initializer);
            prevLayerSize = curLayerSize;
        }
        
        // Last hidden layer <-> Output layer
        curLayerSize = nOutputs;
        curLayerNum = nHiddenLayers;
        weights[curLayerNum] = initLayer(curLayerNum, curLayerSize, prevLayerSize,
                initializer);
    }
    
    /**
     * Allocate memory for the layer's weights according to the size of
     * the previous layer, set their values to the ones supplied by 
     * {@code initializer} and return the allocated weights.
     * @param layerNum The index of the current layer.
     * @param layerSize The size of the current layer.
     * @param prevLayerSize The size of the previous layer.
     * @param initializer {@link Initializer} to be used to initialize weights.
     * @return an allocated 2-D {@code double} array of weights between the 
     * current layer and the previous layer.
     */
    private double[][] initLayer(int layerNum, int layerSize, int prevLayerSize, 
            Initializer initializer) {
        double[][] layerWeights = new double[layerSize][];
        
        for (int layerNeuron = 0; layerNeuron < layerSize; layerNeuron++) {
            layerWeights[layerNeuron] = new double[prevLayerSize];
            for (int prevLayerNeuron = 0; prevLayerNeuron < prevLayerSize; prevLayerNeuron++) {
                layerWeights[layerNeuron][prevLayerNeuron] = 
                        initializer.supplyWeight(
                                layerNum, 
                                prevLayerNeuron, 
                                layerNeuron
                        );
            }
        }
        
        return layerWeights;
    }
    
    /**
     * Allocate memory for the biases according to the size of the network
     * and fill them with values supplied by {@code initializer}.
     * @param initializer {@link Initializer} to be used to initialize biases.
     */
    private void initBiases(Initializer initializer) {
        biases = new double[nHiddenLayers + 1][];
        for (int i = 0; i < nHiddenLayers; i++) {
            biases[i] = new double[hiddenLayerSizes[i]];
            for (int j = 0; j < hiddenLayerSizes[i]; j++) {
                biases[i][j] = initializer.supplyBias(i, j);
            }
        }
        biases[nHiddenLayers] = new double[nOutputs];
        for (int j = 0; j < nOutputs; j++) {
            biases[nHiddenLayers][j] = initializer.supplyBias(nHiddenLayers, j);
        }
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
     * Get a {@code String} representation of this network's structure in the 
     * format: 
     * <pre>(
     * {@literal <}Num-of-inputs{@literal >}, 
     * {@literal <}Hidden-layer-1-size{@literal >},
     *  ...,
     * {@literal <}Hidden-layer-n-size{@literal >}
     * {@literal <}Num-of-outputs{@literal >}
     * ).
     * </pre>
     * @return {@code String} representing this network's structure.
     */
    public String getSignature() {
        if (signature == null) {
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
            signature = sb.toString();
        }
        return signature;
    }
    
    /**
     * Convert to {@code String} representing the structure of this network: 
     * the number of inputs, the sizes of each hidden layer and the number of outputs.
     * @return {@code String} representing the structure of this network
     */
    @Override
    public String toString() {
        return getSignature();
    }
}
