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
 * 带有目标网络的双线程DRL (Deep Reinforcement Learning) Partitioner
 * 使用深度强化学习进行自适应分区，更新和推理使用不同网络并行处理
 * 新增基于键频率的动态分区限制机制
 */
public class DRLPartitioner extends Partitioner {
    private final State state;
    private final int windowSize; // 窗口大小
    private final transient MultiLayerNetwork mainNetwork;     // 主网络用于训练
    private final transient MultiLayerNetwork targetNetwork1;  // 目标网络1
    private final transient MultiLayerNetwork targetNetwork2;  // 目标网络2
    private volatile int activeTargetNetwork = 1; // 当前使用的目标网络 (1 或 2)
    private final AtomicBoolean isUpdatingTargetNetwork = new AtomicBoolean(false); // 是否正在更新目标网络
    
    private final int stateSize;
    private final int actionSize;
    private static final double GAMMA = 0.99;
    private static final double EPSILON_START = 1.0;
    private static final double EPSILON_END = 0.05;
    private static final double EPSILON_DECAY = 0.9999;
    private double epsilon;
    
    // 定时更新目标网络的参数
    private static final int TARGET_UPDATE_FREQUENCY = 256; // 每处理1000个样本更新一次目标网络（从100改为1000）
    private int sampleCounter = 0;
    
    // 线程相关
    private transient ExecutorService trainingExecutor;
    private transient ExecutorService inferenceExecutor;
    private final AtomicBoolean isTraining = new AtomicBoolean(false);
    private final AtomicBoolean isInferencing = new AtomicBoolean(false);
    private transient boolean executorsInitialized = false;
    private final ReentrantReadWriteLock networkLock = new ReentrantReadWriteLock();
    
    // 记录计数器，用于触发训练
    private final AtomicLong recordCounter = new AtomicLong(0);
    
    // 添加记录缓存，用于批量训练
    private final transient List<TrainingRecord> recordBuffer = new ArrayList<>();
    private final transient Object recordBufferLock = new Object();
    
    // 路由表：缓存键到worker的映射
    private final Map<Integer, Integer> routingTable = new ConcurrentHashMap<>();
    private final AtomicBoolean routingTableValid = new AtomicBoolean(true);
    
    // 路由表统计
    private final AtomicLong routingTableHits = new AtomicLong(0);
    private final AtomicLong routingTableMisses = new AtomicLong(0);
    
    // 添加LRU缓存大小限制
    private static final int MAX_ROUTING_TABLE_SIZE = 100;
    
    // 归一化组件
    private final transient NormalizeObservation obsNormalizer;
    private final transient ScaleReward rewardScaler;
    private static final double EPSILON = 1e-8; // 用于归一化的小常数
    
    // 批量训练相关
    private static final int BATCH_SIZE = 128; // 批量训练的大小，也用作缓冲区上限
    
    // ===== 新增：基于State键分区信息的动态分区限制机制 =====
    
    // 分区限制相关参数
    private static final int MIN_PARTITIONS_FOR_CONSTRAINT = 2;  // 最少需要2个分区才启用约束
    private static final boolean ENABLE_PARTITION_CONSTRAINT = true; // 是否启用分区约束
    
    // ===== 新增：基于频率的动态分区上限控制参数 =====
    private static final boolean ENABLE_FREQUENCY_BASED_LIMIT = true;  // 启用基于频率的动态上限
    private static final int MIN_PARTITIONS_PER_KEY = 1;              // 每个键最少分区数
    private static final int MAX_PARTITIONS_PER_KEY = 16;              // 每个键最多分区数（绝对上限）
    private static final double FREQUENCY_LOG_BASE = 1.3;             // 对数函数底数
    private static final double FREQUENCY_SCALE_FACTOR = 1.0;         // 频率缩放因子
    
    // 频率统计和缓存
    private final Map<Integer, Integer> keyFrequencyCache = new ConcurrentHashMap<>();
    private final Map<Integer, Integer> keyPartitionLimitCache = new ConcurrentHashMap<>();
    private double globalAverageFrequency = 1.0;  // 全局平均频率
    private int frequencyUpdateCounter = 0;
    private static final int FREQUENCY_UPDATE_INTERVAL = 1000; // 频率统计更新间隔
    private static final int STATS_OUTPUT_INTERVAL = 10000; // 统计信息输出间隔：每10000个记录
    private final AtomicLong processedRecords = new AtomicLong(0); // 已处理记录总数
    
    // 约束统计
    private final AtomicLong constrainedDecisions = new AtomicLong(0);
    private final AtomicLong unconstrainedDecisions = new AtomicLong(0);
    private final AtomicLong limitedExpansions = new AtomicLong(0);  // 因上限被阻止的扩展次数
    private final AtomicLong allowedExpansions = new AtomicLong(0);   // 允许的扩展次数
    
    // 训练记录类
    private static class TrainingRecord implements java.io.Serializable {
        final double[] stateVector;
        final int action;
        final double reward;
        
        TrainingRecord(double[] stateVector, int action, double reward) {
            this.stateVector = stateVector;
            this.action = action;
            this.reward = reward;
        }
    }
    
    public DRLPartitioner(int numWorkers, int slide, int size, int numOfKeys) {
        super(numWorkers);
        this.state = new State(size, slide, numWorkers, numOfKeys);
        this.windowSize = slide; // 初始化窗口大小
        this.stateSize = numWorkers + numWorkers; // 只有负载 + 分片向量，移除键编码
        this.actionSize = numWorkers;
        this.epsilon = EPSILON_START;
        
        // 构建简化DRL网络 - 只有一个隐藏层
        MultiLayerConfiguration conf = createNetworkConfig();
        
        // 初始化主网络和两个目标网络
        this.mainNetwork = new MultiLayerNetwork(conf);
        this.mainNetwork.init();
        
        this.targetNetwork1 = new MultiLayerNetwork(conf);
        this.targetNetwork1.init();
        
        this.targetNetwork2 = new MultiLayerNetwork(conf);
        this.targetNetwork2.init();
        
        // 复制主网络权重到两个目标网络
        this.targetNetwork1.setParameters(mainNetwork.params());
        this.targetNetwork2.setParameters(mainNetwork.params());
        
        // 初始化线程池
        initializeExecutors();
        
        // 初始化归一化组件
        this.obsNormalizer = new NormalizeObservation(stateSize, EPSILON);
        this.rewardScaler = new ScaleReward(GAMMA, EPSILON);
        
        // 验证归一化组件已正确初始化
        if (this.obsNormalizer == null || this.rewardScaler == null) {
            throw new IllegalStateException("归一化组件初始化失败");
        }
       
        // 设置合理的热键检测阈值，避免初始阈值过高的问题
        int initialThreshold = Math.max(10, numWorkers * 2);
        state.setFrequencyThreshold(initialThreshold);
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
        // 计算中间层神经元数量：输入层 + 输出层
        int hiddenLayerSize = (stateSize + actionSize);

        return new NeuralNetConfiguration.Builder()
            .seed(83)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.0001))
            .list()
            .layer(0, new DenseLayer.Builder()
                .nIn(stateSize)
                .nOut(hiddenLayerSize)
                .activation(Activation.RELU)
                .build())
            .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                .nIn(hiddenLayerSize)
                .nOut(actionSize)
                .activation(Activation.IDENTITY)
                .build())
            .build();
    }

    /**
     * 获取当前活跃的目标网络
     */
    private MultiLayerNetwork getActiveTargetNetwork() {
        return activeTargetNetwork == 1 ? targetNetwork1 : targetNetwork2;
    }
    
    /**
     * 获取备用的目标网络（用于更新）
     */
    private MultiLayerNetwork getInactiveTargetNetwork() {
        return activeTargetNetwork == 1 ? targetNetwork2 : targetNetwork1;
    }
    
    /**
     * 异步更新备用目标网络并切换
     */
    private void updateTargetNetworkAsync() {
        if (isUpdatingTargetNetwork.compareAndSet(false, true)) {
            if (trainingExecutor != null) {
                trainingExecutor.submit(() -> {
                    try {
                        // 获取备用网络并更新其参数
                        MultiLayerNetwork inactiveNetwork = getInactiveTargetNetwork();
                        INDArray mainNetworkParams = mainNetwork.params();
                        
                        inactiveNetwork.setParameters(mainNetworkParams);
                        
                        // 切换活跃网络
                        activeTargetNetwork = activeTargetNetwork == 1 ? 2 : 1;
                        
                    } catch (Exception e) {
                        System.err.println("目标网络更新失败: " + e.getMessage());
                        e.printStackTrace();
                    } finally {
                        isUpdatingTargetNetwork.set(false);
                    }
                });
            } else {
                isUpdatingTargetNetwork.set(false);
            }
        }
    }
    
    /**
     * 从主网络复制权重到目标网络 - 已弃用，使用异步双缓冲方案
     */
    @Deprecated
    private void copyNetworkWeights() {
        // 保留此方法以兼容序列化，但不再使用
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
    

    
    
    @Override
    public void flatMap(Record record, Collector<Tuple2<Integer, Record>> out) throws Exception {
        // 确保ExecutorService已初始化
        if (!executorsInitialized) {
            initializeExecutors();
        }
        
        int keyId = record.getKeyId();
        int worker;
        
        // 计数已处理的记录
        long recordCount = processedRecords.incrementAndGet();
        
        // 每处理10000个记录输出统计信息
        if (recordCount % STATS_OUTPUT_INTERVAL == 0) {
            //outputStatistics(recordCount);
        }

        // 1. 热键检查
        boolean isHot = state.isHot(record, null) == 1;
        state.updateExpired(record, isHot);
        
        double[] stateVector = null;
        
        if (!isHot) {
            // 对于非热键，仍使用简单的哈希分区
            worker = Math.abs(keyId % parallelism);
            state.update(record, worker);
            out.collect(new Tuple2<>(worker, record));
            return;
        }
        
        // 2. 获取键的当前分区分布和动态上限
        BitSet currentPartitions = state.keyfragmentation(keyId);
        int dynamicLimit = getDynamicPartitionLimit(keyId);
        
        // 3. 路由表查询（需要检查选择的worker是否在当前分区内）
        if (routingTableValid.get() && routingTable.containsKey(keyId)) {
            worker = routingTable.get(keyId);
            
            // 如果启用约束且键已有分区历史，验证缓存的worker是否在已分配分区内
            if (ENABLE_PARTITION_CONSTRAINT && currentPartitions.cardinality() >= MIN_PARTITIONS_FOR_CONSTRAINT) {
                if (currentPartitions.get(worker)) {
                    routingTableHits.incrementAndGet();
                } else {
                    // 缓存的worker不在已分配分区内，需要重新计算
                    routingTableMisses.incrementAndGet();
                    routingTable.remove(keyId); // 移除无效缓存
                    stateVector = buildStateVector(record);
                    worker = performConstrainedInferenceWithDynamicLimit(stateVector, keyId, currentPartitions, dynamicLimit);
                    if (routingTableValid.get()) {
                        updateRoutingTable(keyId, worker);
                    }
                }
            } else {
                routingTableHits.incrementAndGet();
            }
        } else {
            routingTableMisses.incrementAndGet();
            stateVector = buildStateVector(record);
            worker = performConstrainedInferenceWithDynamicLimit(stateVector, keyId, currentPartitions, dynamicLimit);
            
            if (routingTableValid.get()) {
                updateRoutingTable(keyId, worker);
            }
        }
        
        // 4. 训练
        if (stateVector == null) {
            stateVector = buildStateVector(record);
        }
        
        double reward = calculateReward(record, worker);
        trainMainNetwork(stateVector, worker, reward);
        
        state.update(record, worker);
        out.collect(new Tuple2<>(worker, record));
        
        epsilon = Math.max(EPSILON_END, epsilon * EPSILON_DECAY);
        
        sampleCounter++;
        if (sampleCounter >= TARGET_UPDATE_FREQUENCY) {
            updateTargetNetworkAsync();
            
             // 清理路由表
            routingTable.clear();
            routingTableValid.set(true);
            sampleCounter = 0;
        }
    }
    
    /**
     * 从已分配分区中选择Q值最高的
     */
    private int selectBestFromExistingPartitions(INDArray qValues, BitSet currentPartitions) {
        int bestAction = -1;
        double bestQValue = Double.NEGATIVE_INFINITY;
        
        for (int i = 0; i < parallelism; i++) {
            if (currentPartitions.get(i)) {
                double qValue = qValues.getDouble(0, i);
                if (qValue > bestQValue) {
                    bestQValue = qValue;
                    bestAction = i;
                }
            }
        }
        
        return bestAction;
    }
    
    /**
     * 获取已分配的分区列表
     */
    private List<Integer> getExistingPartitions(BitSet currentPartitions) {
        List<Integer> existingPartitions = new ArrayList<>();
        for (int i = 0; i < parallelism; i++) {
            if (currentPartitions.get(i)) {
                existingPartitions.add(i);
            }
        }
        return existingPartitions;
    }
    
    private double[] buildStateVector(Record record) {
        double[] stateVector = new double[stateSize];
        
        // 1. 计算负载平衡信息
        double avgLoad = state.avgLoad();
        for (int i = 0; i < parallelism; i++) {
            double workerLoad = state.getLoad(i);
            // 计算每个节点的负载偏差率
            stateVector[i] = avgLoad > 0 ? (workerLoad - avgLoad) / avgLoad : 0.0;
        }
        
        // 2. 分片情况 - 使用完整的分片向量而不是单一比例
        BitSet keyFragmentation = state.keyfragmentation(record.getKeyId());
        for (int i = 0; i < parallelism; i++) {
            stateVector[parallelism + i] = keyFragmentation.get(i) ? 1.0 : 0.0;
        }
        
        // 3. 对状态向量进行归一化（如果归一化器可用）
        if (obsNormalizer != null) {
            double[] normalizedState = obsNormalizer.process(stateVector);
            return normalizedState;
        } else {
            // 归一化器未初始化时，直接返回原始状态向量
            return stateVector;
        }
    }

    private double calculateReward(Record record, int action) {
        double reward = 0.0;
        
        // 1. 负载均衡奖励
        double avgLoad = state.avgLoad();
        double L = state.getLoad(action);
        double loadDiff = (L - avgLoad)/avgLoad;
        double loadReward = -loadDiff * 0.5;
        reward += loadReward;
        
        // 2. 分片惩罚
        double fragmentation = state.keyfragmentation(record.getKeyId()).cardinality() / (double)parallelism;
        double fragmentationPenalty = -fragmentation * 0.5;
        reward += fragmentationPenalty;

        // 3. 对奖励进行缩放归一化（如果缩放器可用）
        if (rewardScaler != null) {
            double scaledReward = rewardScaler.process(reward, false);
            return scaledReward;
        } else {
            // 奖励缩放器未初始化时，直接返回原始reward
            return reward;
        }
    }

    /**
     * 使用经验进行DRL训练 - 异步批量执行，限制缓冲区大小
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
        synchronized (recordBuffer) {
            // 如果缓冲区已满，丢弃新记录，避免过度训练
            if (recordBuffer.size() >= BATCH_SIZE) {
                return; // 直接丢弃新记录
            }
            
            recordBuffer.add(new TrainingRecord(stateVector, action, reward));
            
            // 如果缓冲区达到批量大小，启动训练
            if (recordBuffer.size() >= BATCH_SIZE && !isTraining.get()) {
                // 复制当前缓冲区的内容
                List<TrainingRecord> batch = new ArrayList<>(recordBuffer);
                recordBuffer.clear();
                
                // 异步执行批量训练
                trainingExecutor.submit(() -> trainBatch(batch));
            }
        }
    }
    
    /**
     * 批量训练主网络
     */
    private void trainBatch(List<TrainingRecord> batch) {
        if (isTraining.compareAndSet(false, true)) {
            try {
                //networkLock.writeLock().lock();
                
                // 1. 准备批量输入
                int batchSize = batch.size();
                INDArray stateInputs = Nd4j.create(batchSize, stateSize);
                INDArray targetQValues = Nd4j.create(batchSize, actionSize);
                
                // 2. 填充状态输入和目标Q值
                for (int i = 0; i < batchSize; i++) {
                    TrainingRecord sample = batch.get(i);
                    
                    // 填充状态输入
                    stateInputs.putRow(i, Nd4j.create(sample.stateVector));
                    
                    // 获取当前Q值
                    INDArray currentQ = mainNetwork.output(stateInputs.getRow(i).reshape(1, stateSize));
                    
                    // 更新目标Q值
                    INDArray targetRow = currentQ.dup();
                    double newQValue = currentQ.getDouble(0, sample.action) *(1-GAMMA)+ GAMMA * sample.reward;
                    targetRow.putScalar(new int[]{0, sample.action}, newQValue);
                    targetQValues.putRow(i, targetRow);
                }
                
                // 3. 批量训练主网络
                mainNetwork.fit(stateInputs, targetQValues);
                
                // 4. 训练完成后重置路由表
                resetRoutingTable();
                
            } catch (Exception e) {
                System.err.println("批量训练失败: " + e.getMessage());
                e.printStackTrace();
            } finally {
                //networkLock.writeLock().unlock();
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
        
        // 重新初始化记录缓冲区
        try {
            java.lang.reflect.Field recordBufferField = this.getClass().getDeclaredField("recordBuffer");
            recordBufferField.setAccessible(true);
            recordBufferField.set(this, new ArrayList<>());
            
            java.lang.reflect.Field recordBufferLockField = this.getClass().getDeclaredField("recordBufferLock");
            recordBufferLockField.setAccessible(true);
            recordBufferLockField.set(this, new Object());
        } catch (Exception e) {
            throw new IOException("Failed to initialize record buffer during deserialization", e);
        }
        
        // 重新初始化网络
        MultiLayerConfiguration conf = createNetworkConfig();
        
        // 初始化主网络和两个目标网络
        MultiLayerNetwork tempMainNetwork = new MultiLayerNetwork(conf);
        tempMainNetwork.init();
        
        MultiLayerNetwork tempTargetNetwork1 = new MultiLayerNetwork(conf);
        tempTargetNetwork1.init();
        
        MultiLayerNetwork tempTargetNetwork2 = new MultiLayerNetwork(conf);
        tempTargetNetwork2.init();
        
        // 复制主网络权重到两个目标网络
        tempTargetNetwork1.setParameters(tempMainNetwork.params().dup());
        tempTargetNetwork2.setParameters(tempMainNetwork.params().dup());
        
        // 使用反射设置final字段
        try {
            java.lang.reflect.Field mainNetworkField = this.getClass().getDeclaredField("mainNetwork");
            mainNetworkField.setAccessible(true);
            mainNetworkField.set(this, tempMainNetwork);
            
            java.lang.reflect.Field targetNetwork1Field = this.getClass().getDeclaredField("targetNetwork1");
            targetNetwork1Field.setAccessible(true);
            targetNetwork1Field.set(this, tempTargetNetwork1);
            
            java.lang.reflect.Field targetNetwork2Field = this.getClass().getDeclaredField("targetNetwork2");
            targetNetwork2Field.setAccessible(true);
            targetNetwork2Field.set(this, tempTargetNetwork2);
            
            // 重新初始化归一化组件
            NormalizeObservation tempObsNormalizer = new NormalizeObservation(stateSize, EPSILON);
            ScaleReward tempRewardScaler = new ScaleReward(GAMMA, EPSILON);
            
            java.lang.reflect.Field obsNormalizerField = this.getClass().getDeclaredField("obsNormalizer");
            obsNormalizerField.setAccessible(true);
            obsNormalizerField.set(this, tempObsNormalizer);
            
            java.lang.reflect.Field rewardScalerField = this.getClass().getDeclaredField("rewardScaler");
            rewardScalerField.setAccessible(true);
            rewardScalerField.set(this, tempRewardScaler);
            
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

    /**
     * 获取分区约束统计信息（增强版）
     * 
     * @return 统计信息字符串
     */
    public String getPartitionConstraintStats() {
        long totalDecisions = constrainedDecisions.get() + unconstrainedDecisions.get();
        long totalExpansionAttempts = limitedExpansions.get() + allowedExpansions.get();
        
        double constraintRatio = totalDecisions > 0 ? 
            (double) constrainedDecisions.get() / totalDecisions * 100 : 0;
        double limitRatio = totalExpansionAttempts > 0 ?
            (double) limitedExpansions.get() / totalExpansionAttempts * 100 : 0;
            
        StringBuilder stats = new StringBuilder();
        stats.append("基于频率的动态分区约束统计:\n");
        stats.append(String.format("  约束状态: %s\n", ENABLE_PARTITION_CONSTRAINT ? "启用" : "禁用"));
        stats.append(String.format("  动态上限控制: %s\n", ENABLE_FREQUENCY_BASED_LIMIT ? "启用" : "禁用"));
        stats.append(String.format("  最小约束分区数: %d\n", MIN_PARTITIONS_FOR_CONSTRAINT));
        stats.append(String.format("  分区数范围: %d - %d\n", MIN_PARTITIONS_PER_KEY, MAX_PARTITIONS_PER_KEY));
        stats.append(String.format("  对数函数底数: %.2f\n", FREQUENCY_LOG_BASE));
        stats.append(String.format("  全局平均频率: %.2f\n", globalAverageFrequency));
        stats.append(String.format("  总决策数: %d\n", totalDecisions));
        stats.append(String.format("  约束决策数: %d (%.2f%%)\n", constrainedDecisions.get(), constraintRatio));
        stats.append(String.format("  无约束决策数: %d (%.2f%%)\n", unconstrainedDecisions.get(), 100 - constraintRatio));
        stats.append(String.format("  被限制的扩展: %d (%.2f%%)\n", limitedExpansions.get(), limitRatio));
        stats.append(String.format("  允许的扩展: %d (%.2f%%)\n", allowedExpansions.get(), 100 - limitRatio));
        
        // 添加频率和分区上限分布统计
        if (!keyPartitionLimitCache.isEmpty()) {
            Map<Integer, Integer> limitDistribution = new HashMap<>();
            for (Integer limit : keyPartitionLimitCache.values()) {
                limitDistribution.put(limit, limitDistribution.getOrDefault(limit, 0) + 1);
            }
            stats.append("  分区上限分布:\n");
            for (Map.Entry<Integer, Integer> entry : limitDistribution.entrySet()) {
                stats.append(String.format("    %d个分区: %d个键\n", entry.getKey(), entry.getValue()));
            }
        }
        
        return stats.toString();
    }
    
    /**
     * 检查键是否当前受到分区约束
     * 
     * @param keyId 键ID
     * @return true如果键受到约束，false否则
     */
    public boolean isKeyCurrentlyConstrained(int keyId) {
        if (!ENABLE_PARTITION_CONSTRAINT) {
            return false;
        }
        BitSet fragmentation = state.keyfragmentation(keyId);
        return fragmentation.cardinality() >= MIN_PARTITIONS_FOR_CONSTRAINT;
    }
    
    /**
     * 获取键的当前分区数
     * 
     * @param keyId 键ID
     * @return 当前分区数
     */
    public int getKeyCurrentPartitionCount(int keyId) {
        return state.keyfragmentation(keyId).cardinality();
    }
    
    /**
     * 获取键的当前分区集合
     * 
     * @param keyId 键ID
     * @return 当前分区集合
     */
    public Set<Integer> getKeyCurrentPartitions(int keyId) {
        BitSet fragmentation = state.keyfragmentation(keyId);
        Set<Integer> partitions = new HashSet<>();
        for (int i = 0; i < parallelism; i++) {
            if (fragmentation.get(i)) {
                partitions.add(i);
            }
        }
        return partitions;
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
        
        // 输出最终的分区约束统计
        System.out.println("=== DRL分区器关闭时统计 ===");
        System.out.println(getPartitionConstraintStats());
    }

    /**
     * 检查键是否达到分区上限
     * 
     * @param keyId 键ID
     * @return true如果达到上限，false否则
     */
    public boolean isKeyAtPartitionLimit(int keyId) {
        if (!ENABLE_FREQUENCY_BASED_LIMIT) {
            return false;
        }
        BitSet fragmentation = state.keyfragmentation(keyId);
        int dynamicLimit = getDynamicPartitionLimit(keyId);
        return fragmentation.cardinality() >= dynamicLimit;
    }

    /**
     * 获取键的动态分区上限
     * 
     * @param keyId 键ID
     * @return 键的分区上限
     */
    public int getKeyDynamicPartitionLimit(int keyId) {
        return getDynamicPartitionLimit(keyId);
    }

    
    /**
     * 监测指定键的详细状态信息
     * 
     * @param keyId 键ID
     * @return 键的详细状态字符串
     */
    public String monitorKeyStatus(int keyId) {
        StringBuilder status = new StringBuilder();
        status.append(String.format("=== 键 %d 状态监测 ===\n", keyId));
        
        // 基本信息
        int frequency = getKeyFrequency(keyId);
        int dynamicLimit = getDynamicPartitionLimit(keyId);
        int currentPartitions = getKeyCurrentPartitionCount(keyId);
        Set<Integer> partitionSet = getKeyCurrentPartitions(keyId);
        
        status.append(String.format("  键频率: %d\n", frequency));
        status.append(String.format("  动态分区上限: %d\n", dynamicLimit));
        status.append(String.format("  当前分区数: %d\n", currentPartitions));
        status.append(String.format("  当前分区集合: %s\n", partitionSet));
        
        // 计算详细信息
        double normalizedFrequency = (double) frequency / windowSize * parallelism;
        status.append(String.format("  归一化频率密度: %.6f\n", normalizedFrequency));
        status.append(String.format("  计算公式: %d / %d * %d = %.6f\n", frequency, windowSize, parallelism, normalizedFrequency));
        
        // 约束状态
        boolean isConstrained = isKeyCurrentlyConstrained(keyId);
        boolean atLimit = isKeyAtPartitionLimit(keyId);
        status.append(String.format("  是否受约束: %s\n", isConstrained ? "是" : "否"));
        status.append(String.format("  是否达到上限: %s\n", atLimit ? "是" : "否"));
        
        // 上限计算过程
        status.append(String.format("  上限计算过程:\n"));
        status.append(String.format("    归一化频率密度: %.6f\n", normalizedFrequency));
        status.append(String.format("    floor(%.6f) = %d\n", normalizedFrequency, (int) Math.floor(normalizedFrequency)));
        status.append(String.format("    最终上限: %d + %d = %d\n", 
            MIN_PARTITIONS_PER_KEY, (int) Math.floor(normalizedFrequency), dynamicLimit));
            
        status.append("========================\n");
        
        return status.toString();
    }
    
    /**
     * 监测多个键的状态
     * 
     * @param keyIds 键ID数组
     * @return 所有键的状态字符串
     */
    public String monitorMultipleKeys(int... keyIds) {
        StringBuilder result = new StringBuilder();
        result.append("=== 多键状态监测 ===\n");
        for (int keyId : keyIds) {
            result.append(monitorKeyStatus(keyId));
        }
        return result.toString();
    }
    
    /**
     * 基于键的分区分布和动态上限进行约束推理
     */
    private int performConstrainedInferenceWithDynamicLimit(double[] stateVector, int keyId, 
                                                          BitSet currentPartitions, int dynamicLimit) {
        if (getActiveTargetNetwork() == null) {
            // 如果网络未初始化，优先在已分配分区中选择，否则使用哈希
            if (ENABLE_PARTITION_CONSTRAINT && currentPartitions.cardinality() > 0) {
                List<Integer> existingPartitions = getExistingPartitions(currentPartitions);
                int selectedWorker = existingPartitions.get(Math.abs(keyId % existingPartitions.size()));
                return selectedWorker;
            }
            int hashWorker = Math.abs(keyId % parallelism);
            return hashWorker;
        }
        
        return selectActionWithDynamicLimit(stateVector, keyId, currentPartitions, dynamicLimit);
    }
    
    /**
     * 在动态分区上限约束下进行动作选择
     */
    private int selectActionWithDynamicLimit(double[] stateVector, int keyId, 
                                           BitSet currentPartitions, int dynamicLimit) {
        try {
            //networkLock.readLock().lock();
            
            // 检查是否需要应用约束
            boolean shouldConstrain = ENABLE_PARTITION_CONSTRAINT && 
                                    currentPartitions.cardinality() >= MIN_PARTITIONS_FOR_CONSTRAINT;
            
            // 检查是否达到动态分区上限
            boolean reachedDynamicLimit = ENABLE_FREQUENCY_BASED_LIMIT && 
                                        currentPartitions.cardinality() >= dynamicLimit;
            
            if (Math.random() < epsilon) {
                // 探索阶段
                return handleExplorationWithDynamicLimits(keyId, currentPartitions, shouldConstrain, reachedDynamicLimit);
            } else {
                // 利用阶段：使用Q值进行选择
                return handleExploitationWithDynamicLimits(stateVector, keyId, currentPartitions, shouldConstrain, reachedDynamicLimit);
            }
        } finally {
            //networkLock.readLock().unlock();
        }
    }
    
    /**
     * 处理探索阶段的分区选择（考虑动态上限）
     */
    private int handleExplorationWithDynamicLimits(int keyId, BitSet currentPartitions, 
                                                  boolean shouldConstrain, boolean reachedDynamicLimit) {
        // 如果有任何约束条件（分区数约束或动态上限），在已分配分区中选择
        if (shouldConstrain || reachedDynamicLimit) {
            List<Integer> existingPartitions = getExistingPartitions(currentPartitions);
            if (!existingPartitions.isEmpty()) {
                // 根据约束类型更新统计
                if (reachedDynamicLimit) {
                    limitedExpansions.incrementAndGet();
                }
                constrainedDecisions.incrementAndGet();
                int selectedWorker = existingPartitions.get((int) (Math.random() * existingPartitions.size()));
                return selectedWorker;
            }
        }
        
        // 如果没有历史分区或不启用约束，全局随机选择
        unconstrainedDecisions.incrementAndGet();
        int globalWorker = (int) (Math.random() * parallelism);
        return globalWorker;
    }
    
    /**
     * 处理利用阶段的分区选择（考虑动态上限）
     */
    private int handleExploitationWithDynamicLimits(double[] stateVector, int keyId, BitSet currentPartitions,
                                                   boolean shouldConstrain, boolean reachedDynamicLimit) {
        INDArray stateInput = Nd4j.create(stateVector).reshape(1, stateSize);
        INDArray qValues = getActiveTargetNetwork().output(stateInput);
        
        if (reachedDynamicLimit) {
            // 达到动态上限，强制在已分配分区中选择Q值最高的
            int bestAction = selectBestFromExistingPartitions(qValues, currentPartitions);
            if (bestAction != -1) {
                limitedExpansions.incrementAndGet();
                constrainedDecisions.incrementAndGet();
                return bestAction;
            }
        } else if (shouldConstrain) {
            // 未达到动态上限，直接使用全局最佳Q值
            int bestGlobal = Nd4j.argMax(qValues, 1).getInt(0);
            unconstrainedDecisions.incrementAndGet();
            return bestGlobal;
        }
        
        // 如果没有历史分区或不启用约束，选择全局最优Q值
        unconstrainedDecisions.incrementAndGet();
        int globalBest = Nd4j.argMax(qValues, 1).getInt(0);
        return globalBest;
    }
    
    /**
     * 判断是否应该允许扩展到新分区（简化版：只基于频率上限）
     */
    private boolean shouldAllowExpansionWithDynamicLimit(int keyId, BitSet currentPartitions) {
        // 移除负载感知机制，直接允许扩展（受动态上限控制）
        return true;
    }
    
    /**
     * 根据键频率计算动态分区上限
     * 使用对数函数映射：log_base(frequency * scale + 1) 
     * 
     * @param keyId 键ID
     * @return 该键的分区上限
     */
    private int getDynamicPartitionLimit(int keyId) {
        if (!ENABLE_FREQUENCY_BASED_LIMIT) {
            return MAX_PARTITIONS_PER_KEY; // 如果未启用动态上限，返回最大值
        }

        // 获取键的频率
        int frequency = getKeyFrequency(keyId);
        
        // 使用对数函数映射频率到分区上限
        int dynamicLimit = calculatePartitionLimitFromFrequency(frequency);

        return dynamicLimit;
    }
    
    /**
     * 基于频率计算分区上限
     * 新版：使用归一化频率密度进行对数映射
     * 
     * @param frequency 键的频率
     * @return 分区上限
     */
    private int calculatePartitionLimitFromFrequency(int frequency) {
        if (frequency <= 0) {
            return MIN_PARTITIONS_PER_KEY;
        }
        
        // 获取窗口大小和分区数
        int numPartitions = parallelism; // 分区数
        
        // 计算归一化频率密度：真实频率 / (窗口大小 * 分区数)
        double normalizedFrequency = (double) frequency / windowSize * numPartitions;
        
        // 对数映射后向下取整
        int dynamicLimit = MIN_PARTITIONS_PER_KEY + (int) Math.floor(normalizedFrequency);
        
        // 确保结果在有效范围内
        int finalLimit = Math.max(MIN_PARTITIONS_PER_KEY, Math.min(MAX_PARTITIONS_PER_KEY, dynamicLimit));
        
        return finalLimit;
    }
    
    /**
     * 获取键的频率统计（使用HotStatistics中的真实频率数据）
     * 
     * @param keyId 键ID
     * @return 键的频率（基于HotStatistics）
     */
    private int getKeyFrequency(int keyId) {
        // 首先尝试从缓存获取
        
        // 使用State中HotStatistics的真实键频率统计
        int frequency = state.getKeyFrequency(keyId); // 至少为1，避免零频率
        
        // 缓存结果
        return frequency;
    }
    
    /**
     * 更新全局频率统计
     */
    /**
     * 输出统计信息
     */
    private void outputStatistics(long recordCount) {
        long totalDecisions = constrainedDecisions.get() + unconstrainedDecisions.get();
        long totalExpansionAttempts = limitedExpansions.get() + allowedExpansions.get();
        
        double constraintRatio = totalDecisions > 0 ? 
            (double) constrainedDecisions.get() / totalDecisions * 100 : 0;
        double limitRatio = totalExpansionAttempts > 0 ?
            (double) limitedExpansions.get() / totalExpansionAttempts * 100 : 0;
            
        System.out.println("=== DRL分区器统计信息 (记录数: " + recordCount + ") ===");
        System.out.println(String.format("  约束状态: %s", ENABLE_PARTITION_CONSTRAINT ? "启用" : "禁用"));
        System.out.println(String.format("  动态上限控制: %s", ENABLE_FREQUENCY_BASED_LIMIT ? "启用" : "禁用"));
        System.out.println(String.format("  最小约束分区数: %d", MIN_PARTITIONS_FOR_CONSTRAINT));
        System.out.println(String.format("  分区数范围: %d - %d", MIN_PARTITIONS_PER_KEY, MAX_PARTITIONS_PER_KEY));
        System.out.println(String.format("  对数函数底数: %.2f", FREQUENCY_LOG_BASE));
        System.out.println(String.format("  全局平均频率: %.2f", globalAverageFrequency));
        System.out.println(String.format("  热键检测总记录数: %d", state.getTotalCountOfRecords()));
        System.out.println(String.format("  总决策数: %d", totalDecisions));
        System.out.println(String.format("  约束决策数: %d (%.2f%%)", constrainedDecisions.get(), constraintRatio));
        System.out.println(String.format("  无约束决策数: %d (%.2f%%)", unconstrainedDecisions.get(), 100 - constraintRatio));
        System.out.println(String.format("  被限制的扩展: %d (%.2f%%)", limitedExpansions.get(), limitRatio));
        System.out.println(String.format("  允许的扩展: %d (%.2f%%)", allowedExpansions.get(), 100 - limitRatio));
        
        // 添加频率和分区上限分布统计
        if (!keyPartitionLimitCache.isEmpty()) {
            Map<Integer, Integer> limitDistribution = new HashMap<>();
            for (Integer limit : keyPartitionLimitCache.values()) {
                limitDistribution.put(limit, limitDistribution.getOrDefault(limit, 0) + 1);
            }
            System.out.println("  分区上限分布:");
            for (Map.Entry<Integer, Integer> entry : limitDistribution.entrySet()) {
                System.out.println(String.format("    %d个分区: %d个键", entry.getKey(), entry.getValue()));
            }
        }
        System.out.println("========================");
    }
} 