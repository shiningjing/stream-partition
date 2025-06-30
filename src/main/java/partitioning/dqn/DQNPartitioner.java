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
    private static final double EPSILON_END = 0.05;
    private static final double EPSILON_DECAY = 0.9999;
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
    private static final int MAX_ROUTING_TABLE_SIZE = 1000;
    
    // 归一化组件
    //private final transient NormalizeObservation obsNormalizer;
    //private final transient ScaleReward rewardScaler;
    private static final double EPSILON = 1e-8; // 用于归一化的小常数
    
    // 批量训练相关
    private static final int BATCH_SIZE = 32; // 批量训练的大小，也用于记录触发训练
    
    // 添加时间统计变量
    private final AtomicLong hotKeyCheckTime = new AtomicLong(0);
    private final AtomicLong routingTableTime = new AtomicLong(0);
    private final AtomicLong neuralNetworkTime = new AtomicLong(0);
    private final AtomicLong trainingTime = new AtomicLong(0);
    private final AtomicLong totalProcessingTime = new AtomicLong(0);
    private final AtomicLong recordCount = new AtomicLong(0);
    
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
        //this.obsNormalizer = new NormalizeObservation(stateSize, EPSILON);
        //this.rewardScaler = new ScaleReward(GAMMA, EPSILON);
        
        // 验证归一化组件已正确初始化
       // if (this.obsNormalizer == null || this.rewardScaler == null) {throw new IllegalStateException("归一化组件初始化失败");}
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
        int hiddenLayerSize = (stateSize + actionSize)*2;

        return new NeuralNetConfiguration.Builder()
            .seed(123)
            .weightInit(WeightInit.XAVIER)
            .updater(new Adam(0.0001))
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
        long startTime = System.nanoTime();
        
        // 确保ExecutorService已初始化
        if (!executorsInitialized) {
            initializeExecutors();
        }
        
        int keyId = record.getKeyId();
        int worker;
        
        // 1. 热键检查时间统计
        long hotKeyCheckStart = System.nanoTime();
        boolean isHot = state.isHot(record, null) == 1;
        state.updateExpired(record, isHot);
        hotKeyCheckTime.addAndGet(System.nanoTime() - hotKeyCheckStart);
        
        double[] stateVector = null;
        
        if (!isHot) {
            worker = Math.abs(keyId % parallelism);
            state.update(record, worker);
            out.collect(new Tuple2<>(worker, record));
            totalProcessingTime.addAndGet(System.nanoTime() - startTime);
            recordCount.incrementAndGet();
            return;
        }
        
        // 2. 路由表查询时间统计
        long routingStart = System.nanoTime();
        if (routingTableValid.get() && routingTable.containsKey(keyId)) {
            worker = routingTable.get(keyId);
            routingTableHits.incrementAndGet();
        } else {
            routingTableMisses.incrementAndGet();
            stateVector = buildStateVector(record);
            
            // 3. 神经网络推理时间统计
            long nnStart = System.nanoTime();
            if (targetNetwork == null) {
                worker = Math.abs(keyId % parallelism);
            } else {
                try {
                    networkLock.readLock().lock();
                    if (Math.random() < epsilon) {
                        worker = (int) (Math.random() * parallelism);
                    } else {
                        INDArray stateInput = Nd4j.create(stateVector).reshape(1, stateSize);
                        INDArray qValues = targetNetwork.output(stateInput);
                        worker = Nd4j.argMax(qValues, 1).getInt(0);
                    }
                } finally {
                    networkLock.readLock().unlock();
                }
            }
            neuralNetworkTime.addAndGet(System.nanoTime() - nnStart);
            
            if (routingTableValid.get()) {
                updateRoutingTable(keyId, worker);
            }
        }
        routingTableTime.addAndGet(System.nanoTime() - routingStart);
        
        // 4. 训练时间统计
        long trainingStart = System.nanoTime();
        if (stateVector == null) {
            stateVector = buildStateVector(record);
        }
        
        double reward = calculateReward(record, worker);
        trainMainNetwork(stateVector, worker, reward);
        
        long currentRecordCount = recordCounter.incrementAndGet();
        
        if (currentRecordCount >= BATCH_SIZE) {
            recordCounter.set(0);
        }
        trainingTime.addAndGet(System.nanoTime() - trainingStart);
        
        state.update(record, worker);
        out.collect(new Tuple2<>(worker, record));
        
        epsilon = Math.max(EPSILON_END, epsilon * EPSILON_DECAY);
        
        sampleCounter++;
        if (sampleCounter >= TARGET_UPDATE_FREQUENCY) {
            copyNetworkWeights();
            sampleCounter = 0;
        }
        
        totalProcessingTime.addAndGet(System.nanoTime() - startTime);
        recordCount.incrementAndGet();
        
        // 每处理1000条记录输出一次统计信息
        if (recordCount.get() % 1000 == 0) {
            printTimingStats();
        }
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
       //if (obsNormalizer != null) {
            //return obsNormalizer.process(stateVector);
            //} else {// 归一化器未初始化时，直接返回原始状态向量
            return stateVector;
            //}
}

    private double calculateReward(Record record, int action) {
        double reward = 0.0;
        
        // 1. 负载均衡奖励
        double avgLoad = state.avgLoad();
        double L = state.getLoad(action);
        double loadDiff = (L - avgLoad)/avgLoad;
        reward -= loadDiff * 0.5;
        
        // 2. 分片惩罚
        double fragmentation = state.keyfragmentation(record.getKeyId()).cardinality() / (double)parallelism;
        reward -= fragmentation * 0.5;


        // 3. 对奖励进行缩放归一化（如果缩放器可用）
       //if (rewardScaler != null) {
            //return rewardScaler.process(reward, false); // 在流处理中通常不会终止
        //} else { // 奖励缩放器未初始化时，直接返回原始reward
            return reward;
//}
        
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
        synchronized (recordBuffer) {
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
                networkLock.writeLock().lock();
                
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
            /*java.lang.reflect.Field obsNormalizerField = this.getClass().getDeclaredField("obsNormalizer");
            obsNormalizerField.setAccessible(true);
            obsNormalizerField.set(this, new NormalizeObservation(stateSize, EPSILON));
            
            java.lang.reflect.Field rewardScalerField = this.getClass().getDeclaredField("rewardScaler");
            rewardScalerField.setAccessible(true);
            rewardScalerField.set(this, new ScaleReward(GAMMA, EPSILON));
            
            // 验证归一化组件已正确初始化
            if (obsNormalizerField.get(this) == null || rewardScalerField.get(this) == null) {
                throw new IOException("反序列化后归一化组件初始化失败");
            }

             */
            
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
     * 打印时间统计信息
     */
    private void printTimingStats() {
        long totalRecords = recordCount.get();
        if (totalRecords == 0) return;
        
        // 将纳秒转换为微秒 (1 微秒 = 1000 纳秒)
        double avgTotal = totalProcessingTime.get() / (totalRecords * 1_000.0);
        double avgHotKey = hotKeyCheckTime.get() / (totalRecords * 1_000.0);
        double avgRouting = routingTableTime.get() / (totalRecords * 1_000.0);
        double avgNN = neuralNetworkTime.get() / (totalRecords * 1_000.0);
        double avgTraining = trainingTime.get() / (totalRecords * 1_000.0);
        
        System.out.printf("\n时间统计信息（每条记录平均时间，单位：微秒）：\n");
        System.out.printf("总处理时间: %.3f\n", avgTotal);
        System.out.printf("热键检查时间: %.3f\n", avgHotKey);
        System.out.printf("路由表操作时间: %.3f\n", avgRouting);
        System.out.printf("神经网络推理时间: %.3f\n", avgNN);
        System.out.printf("训练时间: %.3f\n", avgTraining);
        System.out.printf("已处理记录数: %d\n", totalRecords);
        
        double hitRate = routingTableHits.get() + routingTableMisses.get() > 0 
            ? (double) routingTableHits.get() / (routingTableHits.get() + routingTableMisses.get()) * 100 
            : 0.0;
        System.out.printf("路由表命中率: %.2f%%\n", hitRate);
        
        // 添加组件时间占比分析
        if (avgTotal > 0) {
            System.out.println("\n各组件时间占比：");
            System.out.printf("热键检查: %.1f%%\n", (avgHotKey / avgTotal) * 100);
            System.out.printf("路由表操作: %.1f%%\n", (avgRouting / avgTotal) * 100);
            System.out.printf("神经网络推理: %.1f%%\n", (avgNN / avgTotal) * 100);
            System.out.printf("训练: %.1f%%\n", (avgTraining / avgTotal) * 100);
        }
    }
} 