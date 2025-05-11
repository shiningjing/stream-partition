package partitioning.dqn;

import java.io.Serializable;
import java.util.*;

public class DQNReplayMemory implements Serializable {
    private static final long serialVersionUID = 1L;
    private final int capacity;
    private final List<DQNTransition> memory;
    private final Random random;

    public DQNReplayMemory(int capacity) {
        this.capacity = capacity;
        this.memory = new ArrayList<>();
        this.random = new Random();
    }

    public void add(double[] state, int action, double reward, double[] nextState) {
        if (memory.size() >= capacity) {
            memory.remove(0);
        }
        memory.add(new DQNTransition(state, action, reward, nextState));
    }

    public List<DQNTransition> sample(int batchSize) {
        List<DQNTransition> batch = new ArrayList<>();
        for (int i = 0; i < batchSize; i++) {
            int index = random.nextInt(memory.size());
            batch.add(memory.get(index));
        }
        return batch;
    }

    public int size() {
        return memory.size();
    }

    public static class DQNTransition implements Serializable {
        private static final long serialVersionUID = 1L;
        public final double[] state;
        public final int action;
        public final double reward;
        public final double[] nextState;

        public DQNTransition(double[] state, int action, double reward, double[] nextState) {
            this.state = state;
            this.action = action;
            this.reward = reward;
            this.nextState = nextState;
        }
    }
} 