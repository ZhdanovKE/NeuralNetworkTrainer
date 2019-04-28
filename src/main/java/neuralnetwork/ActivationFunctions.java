package neuralnetwork;

/**
 * Enumeration of the supported types of activation function
 * @author Konstantin Zhdanov
 */
public enum ActivationFunctions implements ActivationFunction {
    /**
     * Sigmoid activation function with values in [0,1].
     */
    SIGMOID {
        @Override
        public double valueAt(double at) {
            return 1 / (1 + Math.exp(-at));
        }

        @Override
        public double derivativeValueAt(double at) {
            return valueAt(at) * (1 - valueAt(at));
        }   
    },
    
    /**
     * Tangent activation function with values in [-1,1].
     */
    TAN {
        @Override
        public double valueAt(double at) {
            return 2*SIGMOID.valueAt(2*at) - 1;
        }

        @Override
        public double derivativeValueAt(double at) {
            return 4*SIGMOID.valueAt(2*at);
        }
    };
}