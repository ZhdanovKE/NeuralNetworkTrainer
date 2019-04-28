package neuralnetwork.train;

/**
 * Holder for an array of each neuron's summed input and for an array
 * of each neuron's output.
 * @author Konstantin Zhdanov
 */
class NeuralNetworkResponse {
    double[][] neuronsInputSums;
    double[][] neuronsOutputs;

    NeuralNetworkResponse(double[][] neuronsInputSums, double[][] neuronsOutputs) {
        if (neuronsInputSums == null || neuronsOutputs == null) {
            throw new NullPointerException("Input sums and outputs cannot be null");
        }
        this.neuronsInputSums = neuronsInputSums;
        this.neuronsOutputs = neuronsOutputs;
    }

    public double getNeuronInputSum(int layerNum, int neuronNum) {
        try {
            return neuronsInputSums[layerNum][neuronNum];
        }
        catch(ArrayIndexOutOfBoundsException e) {
            throw new IndexOutOfBoundsException("Layer number or neuron number is out of range");
        }
    }

    public double getNeuronOutput(int layerNum, int neuronNum) {
        try {
            return neuronsOutputs[layerNum][neuronNum];
        }
        catch(ArrayIndexOutOfBoundsException e) {
            throw new IndexOutOfBoundsException("Layer number or neuron number is out of range");
        }
    }

    public double getOutput(int neuronNum) {
        try{
            return neuronsOutputs[neuronsOutputs.length - 1][neuronNum];
        }
        catch(ArrayIndexOutOfBoundsException e) {
            throw new IndexOutOfBoundsException("Neuron number is out of range");
        }
    }

    public double[] getOutputs() {
        return neuronsOutputs[neuronsOutputs.length - 1];
    }
}
