package neuralnetwork;

/**
 * Neural network's neurons' activation function interface.
 * @author Konstantin Zhdanov
 */
public interface ActivationFunction {
    
    /**
     * Calculate the value of the activation function at the
     * specified point {@link at}.
     * @param at a {@code double} value of the point to calculate 
     * the activation function's value at
     * @return A {@code double} value of the activation function at the
     * point {@link at}.
     */
    double valueAt(double at);
    
    /**
     * Calculate the value of the activation function's derivative at the
     * specified point {@link at}.
     * @param at a {@code double} value of the point to calculate 
     * the activation function's derivative's value at
     * @return A {@code double} value of the activation function's derivative
     * at the point {@link at}.
     */
    double derivativeValueAt(double at);
}
