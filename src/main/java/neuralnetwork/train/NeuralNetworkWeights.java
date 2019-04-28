package neuralnetwork.train;

import neuralnetwork.NeuralNetwork;

/**
 * Class for holding neural network's weights and biases and performing 
 * some arithmetic operations with them.
 * @author Konstantin Zhdanov
 */
class NeuralNetworkWeights {
    double[][][] weights;
    double[][] biases;

    public NeuralNetworkWeights(NeuralNetworkWeights src) {
        if (src == null) {
            throw new NullPointerException("Source weigths cannot be null");
        }
        
        biases = (double[][])src.biases.clone();
        for (int i = 0; i < src.biases.length; i++) {
            biases[i] = (double[])src.biases[i].clone();
        }

        weights = (double[][][])src.weights.clone();
        for (int i = 0; i < src.weights.length; i++) {
            weights[i] = (double[][])src.weights[i].clone();
            for (int j = 0; j < src.weights[i].length; j++) {
                weights[i][j] = (double[])src.weights[i][j].clone();
            }
        }
    }
    
     public NeuralNetworkWeights(int nInputs, int[] hiddenLayerSizes, int nOutputs) {
        if (nInputs <= 0 || hiddenLayerSizes.length <= 0 || nOutputs <= 0) {
            throw new IllegalArgumentException("All numbers must be positive");
        }
        for (int size : hiddenLayerSizes) {
            if (size <= 0) {
                throw new IllegalArgumentException("All numbers must be positive");
            }
        }
        initWeights(nInputs, hiddenLayerSizes, nOutputs);
        initBiases(hiddenLayerSizes, nOutputs);
    }
     
    private void initWeights(int nInputs, int[] hiddenLayerSizes, int nOutputs) {
        int nHiddenLayers = hiddenLayerSizes.length;
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
    
    public static NeuralNetworkWeights newOf(NeuralNetwork nn) {
        NeuralNetworkWeights nnWeights = new NeuralNetworkWeights(
                nn.getNumberInputs(), 
                nn.getHiddenLayerSizes(), 
                nn.getNumberOutputs()
        );
        
        for (int layerNeuron = 0; layerNeuron < nn.getHiddenLayerSize(0); layerNeuron++) {
            for (int prevLayerNeuron = 0; prevLayerNeuron < nn.getNumberInputs(); prevLayerNeuron++) {
                nnWeights.setWeight(0, prevLayerNeuron, layerNeuron, nn.getWeight(0, prevLayerNeuron, layerNeuron));
            }
            nnWeights.setBias(0, layerNeuron, nn.getBias(0, layerNeuron));
        }
        for (int layerNum = 1; layerNum < nn.getNumberHiddenLayers(); layerNum++) {
            for (int layerNeuron = 0; layerNeuron < nn.getHiddenLayerSize(layerNum); layerNeuron++) {
                for (int prevLayerNeuron = 0; prevLayerNeuron < nn.getHiddenLayerSize(layerNum - 1); prevLayerNeuron++) {
                    nnWeights.setWeight(layerNum, prevLayerNeuron, layerNeuron, nn.getWeight(layerNum, prevLayerNeuron, layerNeuron));
                }
                nnWeights.setBias(layerNum, layerNeuron, nn.getBias(layerNum, layerNeuron));
            }
        }
        for (int layerNeuron = 0; layerNeuron < nn.getNumberOutputs(); layerNeuron++) {
            for (int prevLayerNeuron = 0; prevLayerNeuron < nn.getHiddenLayerSize(nn.getNumberHiddenLayers() - 1); prevLayerNeuron++) {
                nnWeights.setWeight(nn.getNumberHiddenLayers(), prevLayerNeuron, layerNeuron, nn.getWeight(nn.getNumberHiddenLayers(), prevLayerNeuron, layerNeuron));
            }
            nnWeights.setBias(nn.getNumberHiddenLayers(), layerNeuron, nn.getBias(nn.getNumberHiddenLayers(), layerNeuron));
        }
        
        return nnWeights;
    }
    
    // Can be applied to a neural network of lesser size than this object
    void applyToNeuralNetwork(NeuralNetwork nn) {
        try {
            for (int layerNeuron = 0; layerNeuron < nn.getHiddenLayerSize(0); layerNeuron++) {
                for (int prevLayerNeuron = 0; prevLayerNeuron < nn.getNumberInputs(); prevLayerNeuron++) {
                    nn.setWeight(0, prevLayerNeuron, layerNeuron, getWeight(0, prevLayerNeuron, layerNeuron));
                }
                nn.setBias(0, layerNeuron, getBias(0, layerNeuron));
            }
            for (int layerNum = 1; layerNum < nn.getNumberHiddenLayers(); layerNum++) {
                for (int layerNeuron = 0; layerNeuron < nn.getHiddenLayerSize(layerNum); layerNeuron++) {
                    for (int prevLayerNeuron = 0; prevLayerNeuron < nn.getHiddenLayerSize(layerNum - 1); prevLayerNeuron++) {
                        nn.setWeight(layerNum, prevLayerNeuron, layerNeuron, getWeight(layerNum, prevLayerNeuron, layerNeuron));
                    }
                    nn.setBias(layerNum, layerNeuron, getBias(layerNum, layerNeuron));
                }
            }
            for (int layerNeuron = 0; layerNeuron < nn.getNumberOutputs(); layerNeuron++) {
                for (int prevLayerNeuron = 0; prevLayerNeuron < nn.getHiddenLayerSize(nn.getNumberHiddenLayers() - 1); prevLayerNeuron++) {
                    nn.setWeight(nn.getNumberHiddenLayers(), prevLayerNeuron, layerNeuron, getWeight(nn.getNumberHiddenLayers(), prevLayerNeuron, layerNeuron));
                }
                nn.setBias(nn.getNumberHiddenLayers(), layerNeuron, getBias(nn.getNumberHiddenLayers(), layerNeuron));
            }
        }
        catch(IndexOutOfBoundsException e) {
            throw new IllegalArgumentException("Neural Network size is incompatible", e);
        }
    }
    
    private double[][] initLayer(int layerSize, int prevLayerSize) {
        double[][] layerWeights = new double[layerSize][];
        
        for (int layerNeuron = 0; layerNeuron < layerSize; layerNeuron++) {
            layerWeights[layerNeuron] = new double[prevLayerSize];
        }
        
        return layerWeights;
    }
    
    private void initBiases(int[] hiddenLayerSizes, int nOutputs) {
        int nHiddenLayers = hiddenLayerSizes.length;
        biases = new double[nHiddenLayers + 1][];
        for (int i = 0; i < nHiddenLayers; i++) {
            biases[i] = new double[hiddenLayerSizes[i]];
        }
        biases[nHiddenLayers] = new double[nOutputs];
    }  
   
//    public int getNumberInputs() {
//        return nInputs;
//    }
//    
//    public int getNumberHiddenLayers() {
//        return nHiddenLayers;
//    }
//    
//    public int getHiddenLayerSize(int layerNum) {
//        if (layerNum < 0 || layerNum >= nHiddenLayers) {
//            throw new IndexOutOfBoundsException("Hidden layer index is out of bounds");
//        }
//        return hiddenLayerSizes[layerNum];
//    }
//    
//    public int[] getHiddenLayerSizes() {
//        if (hiddenLayerSizes == null) {
//            return null;
//        }
//        return hiddenLayerSizes.clone();
//    }
//    
//    public int getNumberOutputs() {
//        return nOutputs;
//    }
    
    public double getWeight(int layerNum, int fromNeuronNum, int toNeuronNum) {
        try {
            return weights[layerNum][toNeuronNum][fromNeuronNum];
        }
        catch (ArrayIndexOutOfBoundsException e) {
            throw new IndexOutOfBoundsException("Some of the indices are out of bounds");
        }
    }
    
    public void setWeight(int layerNum, int fromNeuronNum, int toNeuronNum, double weight) {
        try {
            weights[layerNum][toNeuronNum][fromNeuronNum] = weight;
        }
        catch (ArrayIndexOutOfBoundsException e) {
            throw new IndexOutOfBoundsException("Some of the indices are out of bounds");
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
    
    private boolean sameSize(NeuralNetworkWeights arg) {
        if (arg.weights.length != this.weights.length ||
                arg.biases.length != this.biases.length) {
            return false;
        }
        for (int i = 0; i < arg.weights.length; i++) {
            if (arg.weights[i].length != this.weights[i].length) {
                return false;
            }
            for (int j = 0; j < arg.weights[i].length; j++) {
                if (arg.weights[i][j].length != this.weights[i][j].length) {
                    return false;
                }
            }
        }
        for (int i = 0; i < arg.biases.length; i++) {
            if (arg.biases[i].length != this.biases[i].length) {
                return false;
            }
        }
        return true;
    }
    
    public NeuralNetworkWeights add(NeuralNetworkWeights toAdd) {
        if (!sameSize(toAdd)) {
            throw new IllegalArgumentException("Weights to add must be the same size as the lhs object");
        }
        for (int layerNum = 0; layerNum < weights.length; layerNum++) {
            for (int curLayerNeuron = 0; curLayerNeuron < weights[layerNum].length; curLayerNeuron++) {
                for (int prevLayerNeuron = 0; prevLayerNeuron < weights[layerNum][curLayerNeuron].length; prevLayerNeuron++) {
                    weights[layerNum][curLayerNeuron][prevLayerNeuron] += toAdd.getWeight(layerNum, prevLayerNeuron, curLayerNeuron);
                }
                biases[layerNum][curLayerNeuron] += toAdd.getBias(layerNum, curLayerNeuron);
            }
        }
        return this;
    }

    public NeuralNetworkWeights setTo(double value) {
        for (int layerNum = 0; layerNum < weights.length; layerNum++) {
            for (int curLayerNeuron = 0; curLayerNeuron < weights[layerNum].length; curLayerNeuron++) {
                for (int prevLayerNeuron = 0; prevLayerNeuron < weights[layerNum][curLayerNeuron].length; prevLayerNeuron++) {
                    weights[layerNum][curLayerNeuron][prevLayerNeuron] = value;
                }
                biases[layerNum][curLayerNeuron] = value;
            }
        }
        return this;
    }
    
    public NeuralNetworkWeights subtract(NeuralNetworkWeights toSubtract) {
        if (!sameSize(toSubtract)) {
            throw new IllegalArgumentException("Weights to subtract must be the same size as the lhs object");
        }
        for (int layerNum = 0; layerNum < weights.length; layerNum++) {
            for (int curLayerNeuron = 0; curLayerNeuron < weights[layerNum].length; curLayerNeuron++) {
                for (int prevLayerNeuron = 0; prevLayerNeuron < weights[layerNum][curLayerNeuron].length; prevLayerNeuron++) {
                    weights[layerNum][curLayerNeuron][prevLayerNeuron] -= toSubtract.getWeight(layerNum, prevLayerNeuron, curLayerNeuron);
                }
                biases[layerNum][curLayerNeuron] -= toSubtract.getBias(layerNum, curLayerNeuron);
            }
        }
        return this;
    }

    public NeuralNetworkWeights multiply(double factor) {
        for (int layerNum = 0; layerNum < weights.length; layerNum++) {
            for (int curLayerNeuron = 0; curLayerNeuron < weights[layerNum].length; curLayerNeuron++) {
                for (int prevLayerNeuron = 0; prevLayerNeuron < weights[layerNum][curLayerNeuron].length; prevLayerNeuron++) {
                    weights[layerNum][curLayerNeuron][prevLayerNeuron] *= factor;
                }
                biases[layerNum][curLayerNeuron] *= factor;
            }
        }
        return this;
    }

    public double dot(NeuralNetworkWeights toDot) {
        if (!sameSize(toDot)) {
            throw new IllegalArgumentException("Weights to perform inner product with must be the same size as the lhs object");
        }
        double result = 0.0;

        for (int layerNum = 0; layerNum < weights.length; layerNum++) {
            for (int curLayerNeuron = 0; curLayerNeuron < weights[layerNum].length; curLayerNeuron++) {
                for (int prevLayerNeuron = 0; prevLayerNeuron < weights[layerNum][curLayerNeuron].length; prevLayerNeuron++) {
                    result += weights[layerNum][curLayerNeuron][prevLayerNeuron]*toDot.getWeight(layerNum, prevLayerNeuron, curLayerNeuron);
                }
                result += biases[layerNum][curLayerNeuron]*toDot.getBias(layerNum, curLayerNeuron);
            }
        }

        return result;
    }
    
    public double norm() {
        return Math.sqrt(this.dot(this));
    }
}
