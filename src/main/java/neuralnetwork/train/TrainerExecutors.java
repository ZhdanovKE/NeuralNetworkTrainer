package neuralnetwork.train;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;

/**
 * Utility class for creating daemon-threaded executors.
 * 
 * @author Konstantin Zhdanov
 */
public class TrainerExecutors {
    
    private static class DaemonThreadFactory implements ThreadFactory {
        @Override
        public Thread newThread(Runnable r) {
            Thread daemonThread = new Thread(r);
            daemonThread.setDaemon(true);
            return daemonThread;
        }
    }
    
    private final static DaemonThreadFactory DEFAULT_FACTORY = 
            new DaemonThreadFactory();
    
    /**
     * Create a new single-threaded {@code ExecutorService} instance which 
     * creates only daemon threads.
     * @return A single-daemon-threaded {@code ExecutorService} instance.
     */
    public static ExecutorService newExecutor() {
        ExecutorService executor = Executors.newSingleThreadExecutor(DEFAULT_FACTORY);
        return executor;
    }
}
