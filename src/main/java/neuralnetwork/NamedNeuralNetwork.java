package neuralnetwork;

import java.io.Serializable;
import neuralnetwork.init.Initializer;

/**
 * A neural network that contains a name associated with it.
 * @author Konstantin Zhdanov
 */
public class NamedNeuralNetwork extends NeuralNetwork implements Serializable {

    private static final long serialVersionUID = -6259252896064493552L;
    
    /** Name of this Neural Network **/
    private String name;
    
    /**
     * Create a neural network of particular size with a name.
     * @param numInputs Number of input neurons.
     * @param hiddenLayerSizes Number of hidden layers neurons.
     * @param numOutputs Number of outputs neurons.
     * @param name Name of this network.
     * @param initializer {@link Initializer} to be used to initialize weights 
     * and biases.
     * @throws IllegalArgumentException if any provided numerical value is zero or
     * negative.
     * @throws NullPointerException if {@link hiddenLayerSizes} is null or 
     * if {@link name} is null.
     */
    public NamedNeuralNetwork(int numInputs, int[] hiddenLayerSizes, int numOutputs, 
            String name, Initializer initializer) {
        super(numInputs, hiddenLayerSizes, numOutputs, initializer);
        if (name == null) {
            throw new NullPointerException("Name cannot be null");
        }
        this.name = name;
    }
    
    /**
     * Create a neural network of particular size with a name.
     * @param numInputs Number of input neurons.
     * @param hiddenLayerSizes Number of hidden layers neurons.
     * @param numOutputs Number of outputs neurons.
     * @param name Name of this network.
     * @throws IllegalArgumentException if any provided numerical value is zero or
     * negative.
     * @throws NullPointerException if {@link hiddenLayerSizes} is null or 
     * if {@link name} is null.
     */
    public NamedNeuralNetwork(int numInputs, int[] hiddenLayerSizes, int numOutputs, String name) {
        super(numInputs, hiddenLayerSizes, numOutputs);
        if (name == null) {
            throw new NullPointerException("Name cannot be null");
        }
        this.name = name;
    }

    /**
     * Create a neural network as a copy of passed {@link nn} and with name
     * {@link name}.
     * @param nn {@code NeuralNetwork} to be copied into this instance.
     * @param name Name of this network.
     * @throws NullPointerException if {@link nn} is null or 
     * if {@link name} is null.
     */
    public NamedNeuralNetwork(NeuralNetwork nn, String name) {
        super(nn);
        if (name == null) {
            throw new NullPointerException("Name cannot be null");
        }
        this.name = name;
    }
    
    /**
     * Create a neural network as a copy of passed named network {@link nn}.
     * @param nn {@code NamedNeuralNetwork} to be copied into this instance.
     * @throws NullPointerException if {@link nn} is null or 
     */
    public NamedNeuralNetwork(NamedNeuralNetwork nn) {
        this(nn, nn.getName());
    }
    
    /**
     * Name of this network.
     * @return {@code String} name of this network.
     */
    public String getName() {
        return name;
    }
    
    /**
     * Set name for this network.
     * @param name {@code String} to be used as name of this network.
     * @throws NullPointerException if {@link name} is null.
     */
    public void setName(String name) {
        if (name == null) {
            throw new NullPointerException("Name cannot be null");
        }
        this.name = name;
    }
    
    /**
     * Convert to {@code String} containing the name and the structure of the network
     * @return {@code String} containing the name and the structure of the network
     */
    @Override
    public String toString() {
        String nnStructure = super.toString();
        return String.format("%s %s", name, nnStructure);
    }
}
