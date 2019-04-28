package neuralnetwork.train;

/**
 * A class for storing information about a neural network's training 
 * lifecycle event.
 * @author Konstantin Zhdanov
 */
public class TrainerEvent {
    private final int epoch;
    private final double performance;
    
    public TrainerEvent(int epoch, double performance) {
        if (epoch < 0) {
            throw new IllegalArgumentException("Number of epochs cannot be negative");
        }
        this.epoch = epoch;
        this.performance = performance;
    }
    
    /**
     * The epoch's number when this event occurred.
     * @return The epoch's number.
     */
    public int getEpoch() {
        return epoch;
    }
    
    /**
     * The performance (error) of the training when this event occurred.
     * @return The performance value.
     */
    public double getPerformance() {
        return performance;
    }
}
