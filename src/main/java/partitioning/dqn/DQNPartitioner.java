/*Copyright (c) 2022 Data Intensive Applications and Systems Laboratory (DIAS)
                   Ecole Polytechnique Federale de Lausanne
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.*/

package partitioning.dqn;

import record.Record;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;
import partitioning.Partitioner;
import partitioning.dalton.state.State;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.BitSet;
import java.util.concurrent.ConcurrentHashMap;
import java.util.List;
import java.io.IOException;

/**
 * 带有目标网络的双线程DQN (Deep Q-Network) Partitioner
 * 使用深度强化学习进行自适应分区，更新和推理使用不同网络并行处理
 */
public class DQNPartitioner extends Partitioner {
    private final State state;
    private final transient MultiLayerNetwork mainNetwork;     // 主网络用于训练
    private final transient MultiLayerNetwork targetNetwork;   // 目标网络用于预测
    private final int stateSize;
    private final int actionSize;
    private static final double GAMMA = 0.99;
    private static final double EPSILON_START = 1.0;
    private static final double EPSILON_END = 0.1;
    private static final double EPSILON_DECAY = 0.995;
    private double epsilon;
    
    // 定时更新目标网络的参数
    private static final int TARGET_UPDATE_FREQUENCY = 100; // 每处理100个样本更新一次目标网络
    private int sampleCounter = 0;
    
    // 线程相关
    private transient ExecutorService trainingExecutor;
    private transient ExecutorService inferenceExecutor;
    private final AtomicBoolean isTraining = new AtomicBoolean(false);
    private final AtomicBoolean isInferencing = new AtomicBoolean(false);
    private transient boolean executorsInitialized = false;
    private final ReentrantReadWriteLock networkLock = new ReentrantReadWriteLock();
    
    // 训练数据队列
    private final ConcurrentLinkedQueue<TrainingExample> trainingQueue = new ConcurrentLinkedQueue<>();

    // 热键缓存结构
    private final ConcurrentHashMap<Integer, List<Record>> hotKeyBuffer = new ConcurrentHashMap<>();
    private static final int HOT_KEY_BATCH_SIZE = 64; // 热键处理批次大小
    private final AtomicBoolean isProcessingHotKeys = new AtomicBoolean(false);

    // 用于存储训练样本的类
    private static class TrainingExample {
        double[] state;
        int action;
        double reward;
        
        public TrainingExample(double[] state, int action, double reward) {
            this.state = state;
            this.action = action;
            this.reward = reward;
        }
    }
    
    public DQNPartitioner(int numWorkers, int slide, int size, int numOfKeys) {
        super(numWorkers);
        this.state = new State(size, slide, numWorkers, numOfKeys);
        this.stateSize = numWorkers + numWorkers + 16; // 负载 + 分片向量 + 键编码
        this.actionSize = numWorkers;
        this.epsilon = EPSILON_START;
        
        // 构建简化DQN网络 - 只有一个隐藏层
        MultiLayerConfiguration conf = createNetworkConfig();
        
        // 初始化主网络和目标网络
        this.mainNetwork = new MultiLayerNetwork(conf);
        this.mainNetwork.init();
        
        this.targetNetwork = new MultiLayerNetwork(conf);
        this.targetNetwork.init();
        
        // 复制主网络权重到目标网络
        copyNetworkWeights();
        
        // 初始化线程池
        initializeExecutors();
    }
    
    /**
     * 初始化线程池
     */
    private void initializeExecutors() {
        if (!executorsInitialized) {
            trainingExecutor = Executors.newSingleThreadExecutor();
            inferenceExecutor = Executors.newSingleThreadExecutor();
            executorsInitialized = true;
        }
    }
    
    /**
     * 创建神经网络配置
     */
    private MultiLayerConfiguration createNetworkConfig() {
        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.001))
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(stateSize)
                .nOut(64)  // 只有一个隐藏层
                .activation(Activation.RELU)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(64)
                .nOut(actionSize)
                .activation(Activation.IDENTITY)
                .build())
            .build();
    }

    /**
     * 从主网络复制权重到目标网络
     */
    private void copyNetworkWeights() {
        // 检查网络是否已初始化
        if (mainNetwork == null || targetNetwork == null) {
            System.err.println("网络未初始化，跳过权重复制");
            return;
        }
        
        try {
            networkLock.writeLock().lock();
            // 获取主网络参数
            targetNetwork.setParameters(mainNetwork.params().dup());
        } finally {
            networkLock.writeLock().unlock();
        }
    }

    /**
     * 在关闭时清理资源
     */
    public void close() {
        if (trainingExecutor != null) {
            trainingExecutor.shutdown();
        }
        if (inferenceExecutor != null) {
            inferenceExecutor.shutdown();
        }
    }
    
    @Override
    public void flatMap(Record record, Collector<Tuple2<Integer, Record>> out) throws Exception {
        // 确保ExecutorService已初始化
        if (!executorsInitialized) {
            initializeExecutors();
        }
        
        // 1. 首先判断是否为热键
        boolean isHot = state.isHot(record, null) == 1;
        
        int worker;
        
        if (!isHot) {
            // 非热键使用简单哈希分区
            worker = Math.abs(record.getKeyId() % parallelism);
            // 更新状态
            state.update(record, worker);
            // 直接输出结果
            out.collect(new Tuple2<>(worker, record));
            return;
        }
        
        // 热键处理 - 直接进行推理
        // 构建状态向量
        double[] stateVector = buildStateVector(record);
        
        // 使用目标网络进行推理
        if (targetNetwork == null) {
            // 网络未初始化，使用简单哈希分区
            worker = Math.abs(record.getKeyId() % parallelism);
        } else {
            try {
                networkLock.readLock().lock();
                INDArray stateInput = Nd4j.create(stateVector).reshape(1, stateSize);
                INDArray qValues = targetNetwork.output(stateInput);
                worker = Nd4j.argMax(qValues, 1).getInt(0);
            } finally {
                networkLock.readLock().unlock();
            }
        }
        
        // 将热键添加到缓冲区用于批量训练
        hotKeyBuffer.computeIfAbsent(record.getKeyId(), k -> new ArrayList<>())
                    .add(new Record(record.getKeyId(), record.getTs()));
        
        // 检查是否需要处理热键进行批量训练
        int totalHotKeys = hotKeyBuffer.keySet().size();
        if (totalHotKeys >= HOT_KEY_BATCH_SIZE && !isProcessingHotKeys.get()) {
            processHotKeysForTraining();
        }
        
        // 更新状态并输出结果
        state.update(record, worker);
        out.collect(new Tuple2<>(worker, record));
        
        // 添加到训练队列
        double reward = calculateReward(record, worker);
        trainingQueue.offer(new TrainingExample(stateVector, worker, reward));
        
        // 启动异步训练
        if (!isTraining.get()) {
            startAsyncTraining();
        }
        
        // 更新探索率
        epsilon = Math.max(EPSILON_END, epsilon * EPSILON_DECAY);
        
        // 更新样本计数器并检查是否需要更新目标网络
        sampleCounter++;
        if (sampleCounter >= TARGET_UPDATE_FREQUENCY) {
            copyNetworkWeights();
            sampleCounter = 0;
        }
    }
    
    /**
     * 处理积累的热键用于训练
     */
    private void processHotKeysForTraining() {
        if (isProcessingHotKeys.compareAndSet(false, true)) {
            trainingExecutor.submit(() -> {
                try {
                    // 获取所有等待处理的热键
                    List<Integer> keyIds = new ArrayList<>(hotKeyBuffer.keySet());
                    
                    for (Integer keyId : keyIds) {
                        List<Record> records = hotKeyBuffer.get(keyId);
                        if (records != null && !records.isEmpty()) {
                            // 对每条记录：创建训练样本
                            for (Record record : records) {
                                // 构建状态向量
                                double[] stateVector = buildStateVector(record);
                                
                                // 使用目标网络进行推理
                                int bestWorker;
                                if (targetNetwork == null) {
                                    // 网络未初始化，使用简单哈希分区
                                    bestWorker = Math.abs(record.getKeyId() % parallelism);
                                } else {
                                    try {
                                        networkLock.readLock().lock();
                                        INDArray stateInput = Nd4j.create(stateVector).reshape(1, stateSize);
                                        INDArray qValues = targetNetwork.output(stateInput);
                                        bestWorker = Nd4j.argMax(qValues, 1).getInt(0);
                                    } finally {
                                        networkLock.readLock().unlock();
                                    }
                                }
                                
                                // 加入训练队列
                                double reward = calculateReward(record, bestWorker);
                                trainingQueue.offer(new TrainingExample(stateVector, bestWorker, reward));
                            }
                            
                            // 清除已处理的记录
                            hotKeyBuffer.remove(keyId);
                        }
                    }
                    
                    // 检查训练队列
                    if (!isTraining.get() && !trainingQueue.isEmpty()) {
                        startAsyncTraining();
                    }
                    
                } catch (Exception e) {
                    System.err.println("热键处理过程中发生严重错误，程序即将退出: " + e.getMessage());
                    e.printStackTrace();
                    System.exit(1);
                } finally {
                    isProcessingHotKeys.set(false);
                    
                    // 检查是否还有足够的热键继续处理
                    if (hotKeyBuffer.size() >= HOT_KEY_BATCH_SIZE) {
                        processHotKeysForTraining();
                    }
                }
            });
        }
    }

    private double[] buildStateVector(Record record) {
        double[] stateVector = new double[stateSize];
        
        // 1. 负载信息
        for (int i = 0; i < parallelism; i++) {
            stateVector[i] = state.getLoad(i);
        }
        
        // 2. 分片情况 - 使用完整的分片向量而不是单一比例
        BitSet keyFragmentation = state.keyfragmentation(record.getKeyId());
        for (int i = 0; i < parallelism; i++) {
            stateVector[parallelism + i] = keyFragmentation.get(i) ? 1.0 : 0.0;
        }
        
        // 3. 键编码
        double[] keyEncoding = encodeKey(record.getKeyId());
        System.arraycopy(keyEncoding, 0, stateVector, parallelism + parallelism, 16);
        
        return stateVector;
    }

    private double[] encodeKey(int keyId) {
        double[] encoding = new double[16];
        for (int i = 0; i < 16; i++) {
            encoding[i] = ((keyId >> i) & 1) == 1 ? 1.0 : 0.0;
        }
        return encoding;
    }

    private double calculateReward(Record record, int action) {
        double reward = 0.0;
        
        // 1. 负载均衡奖励
        double avgLoad = state.avgLoad();
        double loadDiff = Math.abs(state.getLoad(action) - avgLoad);
        reward -= loadDiff;
        
        // 2. 分片惩罚
        double fragmentation = state.keyfragmentation(record.getKeyId()).cardinality() / (double)parallelism;
        reward -= fragmentation * 0.5;
        
        // 3. 热点处理奖励
        if (state.isHot(record, null) == 1) {
            reward += 1.0;
        }
        
        return reward;
    }

    /**
     * 启动异步训练过程
     */
    private void startAsyncTraining() {
        if (isTraining.compareAndSet(false, true)) {
            trainingExecutor.submit(() -> {
                try {
                    // 一次最多处理32个样本
                    int batchSize = Math.min(32, trainingQueue.size());
                    
                    if (batchSize == 0) {
                        return; // 没有样本可训练
                    }
                    
                    for (int i = 0; i < batchSize; i++) {
                        TrainingExample example = trainingQueue.poll();
                        if (example == null) break;
                        
                        // 训练主网络
                        trainMainNetwork(example.state, example.action, example.reward);
                    }
                } catch (Exception e) {
                    System.err.println("训练过程中发生严重错误，程序即将退出: " + e.getMessage());
                    e.printStackTrace();
                    System.exit(1);
                } finally {
                    isTraining.set(false);
                    
                    // 如果队列中还有数据，继续训练
                    if (!trainingQueue.isEmpty()) {
                        startAsyncTraining();
                    }
                }
            });
        }
    }

    /**
     * 训练主网络
     */
    private void trainMainNetwork(double[] stateVector, int action, double reward) {
        // 检查网络是否已初始化
        if (mainNetwork == null) {
            System.err.println("主网络未初始化，跳过训练");
            return;
        }
        
        try {
            networkLock.writeLock().lock();
            // 1. 准备状态输入
            INDArray stateInput = Nd4j.create(stateVector).reshape(1, stateSize);
            
            // 2. 获取当前Q值
            INDArray currentQ = mainNetwork.output(stateInput);
            
            // 3. 计算目标Q值 (使用奖励)
            double targetQ = reward;
            
            // 4. 更新Q值
            INDArray targetQValues = currentQ.dup();
            targetQValues.putScalar(new int[]{0, action}, targetQ);
            
            // 5. 训练主网络
            mainNetwork.fit(stateInput, targetQValues);
        } finally {
            networkLock.writeLock().unlock();
        }
    }
    
    /**
     * 自定义序列化方法 - 确保不序列化执行器
     */
    private void writeObject(java.io.ObjectOutputStream out) throws java.io.IOException {
        out.defaultWriteObject();
    }
    
    /**
     * 自定义反序列化方法 - 初始化执行器和网络
     */
    private void readObject(java.io.ObjectInputStream in) throws java.io.IOException, ClassNotFoundException {
        in.defaultReadObject();
        executorsInitialized = false;  // 将在下一次调用时重新初始化
        
        // 重新初始化网络
        MultiLayerConfiguration conf = createNetworkConfig();
        
        // 初始化主网络和目标网络
        MultiLayerNetwork tempMainNetwork = new MultiLayerNetwork(conf);
        tempMainNetwork.init();
        
        MultiLayerNetwork tempTargetNetwork = new MultiLayerNetwork(conf);
        tempTargetNetwork.init();
        
        // 复制主网络权重到目标网络
        tempTargetNetwork.setParameters(tempMainNetwork.params().dup());
        
        // 使用反射设置final字段
        try {
            java.lang.reflect.Field mainNetworkField = this.getClass().getDeclaredField("mainNetwork");
            mainNetworkField.setAccessible(true);
            mainNetworkField.set(this, tempMainNetwork);
            
            java.lang.reflect.Field targetNetworkField = this.getClass().getDeclaredField("targetNetwork");
            targetNetworkField.setAccessible(true);
            targetNetworkField.set(this, tempTargetNetwork);
        } catch (Exception e) {
            throw new IOException("Failed to initialize networks during deserialization", e);
        }
    }
} 