/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package neuralnetwork.train;

/**
 * Provides methods to be called on certain NeuralNetworkTrainer's lifecycle events.
 * @author Konstantin Zhdanov
 */
public interface Listener {
    /**
     * Called when training has finished.
     * @param event {@code TrainerEvent} object containing information about the training:
     * the number of optimization iterations (epochs) performed and the final achieved
     * performance (measure of neural network's accuracy on training set).
     */
    void onTrainingComplete(TrainerEvent event);

    /**
     * Called when training has been canceled.
     * @param event {@code TrainerEvent} object containing information about the training
     * at the moment when it was canceled:
     * the number of optimization iterations (epochs) completely performed (0 if none) and the achieved
     * performance (measure of neural network's accuracy on training set).
     */
    void onTrainingCanceled(TrainerEvent event);

    /**
     * Called when one training epoch has finished.
     * @param event {@code TrainerEvent} object containing information about the training
     * after the last epoch:
     * the number of this optimization iteration (epoch) and the achieved
     * performance (measure of neural network's accuracy on training set).
     */
    void onTrainingEpochComplete(TrainerEvent event);
}
