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
import java.util.concurrent.atomic.AtomicLong;
import partitioning.dqn.containers.NormalizeObservation;
import partitioning.dqn.containers.ScaleReward;
import java.util.concurrent.CompletableFuture;

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
    
    // 热键缓存结构
    private final ConcurrentHashMap<Integer, List<Record>> hotKeyBuffer = new ConcurrentHashMap<>();
    private static final int HOT_KEY_BATCH_SIZE = 2; // 热键处理批次大小
    private final AtomicBoolean isProcessingHotKeys = new AtomicBoolean(false);
    
    // 添加并行处理线程数
    private static final int PARALLEL_PROCESSING_THREADS = 4;
    private transient ExecutorService hotKeyExecutor;
    
    // 路由表：缓存键到worker的映射
    private final Map<Integer, Integer> routingTable = new ConcurrentHashMap<>();
    private final AtomicBoolean routingTableValid = new AtomicBoolean(true);
    
    // 路由表统计
    private final AtomicLong routingTableHits = new AtomicLong(0);
    private final AtomicLong routingTableMisses = new AtomicLong(0);
    
    // 添加LRU缓存大小限制
    private static final int MAX_ROUTING_TABLE_SIZE = 10000;
    
    // 归一化组件
    private final transient NormalizeObservation obsNormalizer;
    private final transient ScaleReward rewardScaler;
    private static final double EPSILON = 1e-8; // 用于归一化的小常数
    
    // 批量训练相关
    private static final int BATCH_SIZE = 32; // 批量训练的大小
    private final List<TrainingSample> trainingBuffer = new ArrayList<>();
    
    // 训练样本类
    private static class TrainingSample {
        final double[] stateVector;
        final int action;
        final double reward;
        
        TrainingSample(double[] stateVector, int action, double reward) {
            this.stateVector = stateVector;
            this.action = action;
            this.reward = reward;
        }
    }
    
    public DQNPartitioner(int numWorkers, int slide, int size, int numOfKeys) {
        super(numWorkers);
        this.state = new State(size, slide, numWorkers, numOfKeys);
        this.stateSize = numWorkers + numWorkers; // 只有负载 + 分片向量，移除键编码
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
        
        // 初始化归一化组件
        this.obsNormalizer = new NormalizeObservation(stateSize, EPSILON);
        this.rewardScaler = new ScaleReward(GAMMA, EPSILON);
        
        // 验证归一化组件已正确初始化
        if (this.obsNormalizer == null || this.rewardScaler == null) {
            throw new IllegalStateException("归一化组件初始化失败");
        }
    }
    
    /**
     * 初始化线程池
     */
    private void initializeExecutors() {
        if (!executorsInitialized) {
            trainingExecutor = Executors.newSingleThreadExecutor();
            inferenceExecutor = Executors.newSingleThreadExecutor();
            hotKeyExecutor = Executors.newFixedThreadPool(PARALLEL_PROCESSING_THREADS);
            executorsInitialized = true;
        }
    }
    
    /**
     * 创建神经网络配置
     */
    private MultiLayerConfiguration createNetworkConfig() {
        // 计算中间层神经元数量：输入层 + 输出层
        int hiddenLayerSize = (stateSize + actionSize)*2;

        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.001))
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(stateSize)
                .nOut(hiddenLayerSize)
                .activation(Activation.RELU)
                .build())
            .layer(1, new DenseLayer.Builder()
                .nIn(hiddenLayerSize)
                .nOut(hiddenLayerSize)
                .activation(Activation.RELU)
                .build())
            .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(hiddenLayerSize)
                .nOut(actionSize)
                .activation(Activation.IDENTITY)
                .build())
            .build();
    }

    /**
     * 从主网络复制权重到目标网络
     */
    private void copyNetworkWeights() {
      
        try {
            networkLock.writeLock().lock();
            // 获取主网络参数
            targetNetwork.setParameters(mainNetwork.params().dup());
            
            // 目标网络更新后，重置路由表
            routingTable.clear();
            routingTableValid.set(true);
        } finally {
            networkLock.writeLock().unlock();
        }
    }

    /**
     * 重置路由表
     * 在神经网络训练完成后调用，确保使用最新的网络权重进行推理
     */
    private void resetRoutingTable() {
        int oldSize = routingTable.size();
        long hits = routingTableHits.get();
        long misses = routingTableMisses.get();
        long total = hits + misses;
        double hitRate = total > 0 ? (double) hits / total * 100 : 0.0;
        
        routingTable.clear();
        routingTableValid.set(true);
        
        // 重置统计计数器
        routingTableHits.set(0);
        routingTableMisses.set(0);
        
       
    }
    
    /**
     * 检查归一化组件是否已正确初始化
     * @throws IllegalStateException 如果任何归一化组件未初始化
     */
    private void validateNormalizationComponents() {
        if (obsNormalizer == null) {
            throw new IllegalStateException("观察值归一化器未初始化");
        }
        if (rewardScaler == null) {
            throw new IllegalStateException("奖励缩放器未初始化");
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
        if (hotKeyExecutor != null) {
            hotKeyExecutor.shutdown();
        }
    }
    
    @Override
    public void flatMap(Record record, Collector<Tuple2<Integer, Record>> out) throws Exception {
        // 确保ExecutorService已初始化
        if (!executorsInitialized) {
            initializeExecutors();
        }
        
        int keyId = record.getKeyId();
        int worker;
        
        // 1. 热键检查
        boolean isHot = state.isHot(record, null) == 1;
        
        // 新增：滑动窗口过期机制
        state.updateExpired(record, isHot);
        
        // 预先构建状态向量，避免重复计算
        double[] stateVector = null;
        
        if (!isHot) {
            // 非热键使用简单哈希分区
            worker = Math.abs(keyId % parallelism);
            // 更新状态
            state.update(record, worker);
            // 直接输出结果
            out.collect(new Tuple2<>(worker, record));
            return;
        }
        
        // 2. 热键处理 - 首先检查路由表
        if (routingTableValid.get() && routingTable.containsKey(keyId)) {
            // 使用缓存的路由结果
            worker = routingTable.get(keyId);
            routingTableHits.incrementAndGet();
        } else {
            // 路由表中没有或已失效，需要进行神经网络推理
            routingTableMisses.incrementAndGet();
            stateVector = buildStateVector(record);
            
            // 使用目标网络进行推理
            if (targetNetwork == null) {
                // 网络未初始化，使用简单哈希分区
                worker = Math.abs(keyId % parallelism);
            } else {
                try {
                    networkLock.readLock().lock();
                    INDArray stateInput = Nd4j.create(stateVector).reshape(1, stateSize);
                    INDArray qValues = targetNetwork.output(stateInput);
                    worker = Nd4j.argMax(qValues, 1).getInt(0);
                    System.out.println("key:"+keyId+",QValues: " + qValues + ", Selected worker: " + worker);
                } finally {
                    networkLock.readLock().unlock();
                }
            }
            
            // 将结果缓存到路由表
            if (routingTableValid.get()) {
                updateRoutingTable(keyId, worker);
            }
        }
        
        // 3. 热键缓冲区处理
        hotKeyBuffer.computeIfAbsent(keyId, k -> new ArrayList<>())
                    .add(new Record(keyId, record.getTs()));
        
        // 检查是否需要处理热键进行批量训练
        int totalHotKeys = hotKeyBuffer.keySet().size();
        if (totalHotKeys >= HOT_KEY_BATCH_SIZE && !isProcessingHotKeys.get()) {
            processHotKeysForTraining();
        }
        
        // 4. 更新状态并输出结果
        state.update(record, worker);
        out.collect(new Tuple2<>(worker, record));
        
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
            hotKeyExecutor.submit(() -> {
                try {
                    // 获取所有等待处理的热键
                    List<Integer> keyIds = new ArrayList<>(hotKeyBuffer.keySet());
                    
                    // 并行处理热键
                    List<CompletableFuture<Void>> futures = new ArrayList<>();
                    
                    for (Integer keyId : keyIds) {
                        CompletableFuture<Void> future = CompletableFuture.runAsync(() -> {
                            List<Record> records = hotKeyBuffer.get(keyId);
                            if (records != null && !records.isEmpty()) {
                                processHotKeyRecords(keyId, records);
                            }
                        }, hotKeyExecutor);
                        
                        futures.add(future);
                    }
                    
                    // 等待所有处理完成
                    CompletableFuture.allOf(futures.toArray(new CompletableFuture[0])).join();
                    
                } catch (Exception e) {
                    System.err.println("热键处理过程中发生错误: " + e.getMessage());
                    e.printStackTrace();
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
    
    /**
     * 处理单个热键的记录
     */
    private void processHotKeyRecords(Integer keyId, List<Record> records) {
        for (Record record : records) {
            // 构建状态向量
            double[] stateVector = buildStateVector(record);
            
            // 使用目标网络进行推理
            int bestWorker;
            if (targetNetwork == null) {
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
            
            // 计算奖励并直接训练网络
            double reward = calculateReward(record, bestWorker);
            trainMainNetwork(stateVector, bestWorker, reward);
        }
        
        // 清除已处理的记录
        hotKeyBuffer.remove(keyId);
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
        
        // 3. 对状态向量进行归一化（如果归一化器可用）
        if (obsNormalizer != null) {
            return obsNormalizer.process(stateVector);
        } else {
            throw new IllegalStateException("观察值归一化器未初始化，无法处理状态向量");
        }
    }

    private double calculateReward(Record record, int action) {
        double reward = 0.0;
        
        // 1. 负载均衡奖励
        double avgLoad = state.avgLoad();
        double L = state.getLoad(action);
        double loadDiff = (L - avgLoad)/Math.max(L,avgLoad);
        reward -= loadDiff*10;
        
        // 2. 分片惩罚
        double fragmentation = state.keyfragmentation(record.getKeyId()).cardinality() / (double)parallelism;
        reward -= fragmentation * 10;

        return reward;
        // 4. 对奖励进行缩放归一化（如果缩放器可用）
        /*if (rewardScaler != null) {
            return rewardScaler.process(reward, false); // 在流处理中通常不会终止
        } else {
            throw new IllegalStateException("奖励缩放器未初始化，无法处理奖励值");
        }
        
         */

        
    }

    /**
     * 使用经验进行DQN训练 - 异步批量执行
     */
    private void trainMainNetwork(double[] stateVector, int action, double reward) {
        // 检查网络是否已初始化
        if (mainNetwork == null) {
            return;
        }
        
        // 确保执行器已初始化
        if (!executorsInitialized) {
            initializeExecutors();
        }
        
        // 添加训练样本到缓冲区
        synchronized (trainingBuffer) {
            trainingBuffer.add(new TrainingSample(stateVector, action, reward));
            
            // 如果缓冲区达到批量大小，启动训练
            if (trainingBuffer.size() >= BATCH_SIZE && !isTraining.get()) {
                // 复制当前缓冲区的内容
                List<TrainingSample> batch = new ArrayList<>(trainingBuffer);
                trainingBuffer.clear();
                
                // 异步执行批量训练
                trainingExecutor.submit(() -> trainBatch(batch));
            }
        }
    }
    
    /**
     * 批量训练主网络
     */
    private void trainBatch(List<TrainingSample> batch) {
        if (isTraining.compareAndSet(false, true)) {
            try {
                networkLock.writeLock().lock();
                
                // 1. 准备批量输入
                int batchSize = batch.size();
                INDArray stateInputs = Nd4j.create(batchSize, stateSize);
                INDArray targetQValues = Nd4j.create(batchSize, actionSize);
                
                // 2. 填充状态输入和目标Q值
                for (int i = 0; i < batchSize; i++) {
                    TrainingSample sample = batch.get(i);
                    
                    // 填充状态输入
                    stateInputs.putRow(i, Nd4j.create(sample.stateVector));
                    
                    // 获取当前Q值
                    INDArray currentQ = mainNetwork.output(stateInputs.getRow(i).reshape(1, stateSize));
                    
                    // 更新目标Q值
                    INDArray targetRow = currentQ.dup();
                    targetRow.putScalar(new int[]{0, sample.action}, currentQ.getDouble(0, sample.action) *(1-GAMMA)+ GAMMA * sample.reward);
                    targetQValues.putRow(i, targetRow);
                }
                
                // 3. 批量训练主网络
                mainNetwork.fit(stateInputs, targetQValues);
                
                // 4. 训练完成后重置路由表
                resetRoutingTable();
                
            } catch (Exception e) {
                System.err.println("批量训练过程中发生错误: " + e.getMessage());
                e.printStackTrace();
            } finally {
                networkLock.writeLock().unlock();
                isTraining.set(false);
            }
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
            
            // 重新初始化归一化组件
            java.lang.reflect.Field obsNormalizerField = this.getClass().getDeclaredField("obsNormalizer");
            obsNormalizerField.setAccessible(true);
            obsNormalizerField.set(this, new NormalizeObservation(stateSize, EPSILON));
            
            java.lang.reflect.Field rewardScalerField = this.getClass().getDeclaredField("rewardScaler");
            rewardScalerField.setAccessible(true);
            rewardScalerField.set(this, new ScaleReward(GAMMA, EPSILON));
            
            // 验证归一化组件已正确初始化
            if (obsNormalizerField.get(this) == null || rewardScalerField.get(this) == null) {
                throw new IOException("反序列化后归一化组件初始化失败");
            }
            
        } catch (Exception e) {
            throw new IOException("Failed to initialize networks and normalizers during deserialization", e);
        }
    }

    /**
     * 更新路由表，使用LRU策略
     */
    private void updateRoutingTable(int keyId, int worker) {
        if (!routingTableValid.get()) {
            return;
        }
        
        // 如果达到最大容量，移除最旧的条目
        if (routingTable.size() >= MAX_ROUTING_TABLE_SIZE) {
            // 使用ConcurrentHashMap的keySet().iterator()获取第一个元素（最旧的）
            routingTable.remove(routingTable.keySet().iterator().next());
        }
        
        routingTable.put(keyId, worker);
    }
} 