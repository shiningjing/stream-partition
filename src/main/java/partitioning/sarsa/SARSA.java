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

package partitioning.sarsa;

import streamSARSA.StreamSARSA;
import partitioning.dalton.DaltonCooperative;
import partitioning.dalton.state.State;
import record.Record;

import java.io.Serializable;
import java.util.BitSet;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * SARSA算法实现的自适应分区决策
 * 使用深度强化学习来替代ContextualBandits中的Q表
 * 单一模型处理所有键，键ID作为状态的一部分
 */
public class SARSA implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private State state;
    private int numOfWorkers;
    private StreamSARSA model; // 单一全局SARSA模型
    private double epsilon; // 探索率
    private Set<Integer> hotKeys; // 跟踪当前热点键
    
    // 模型参数配置（可序列化）
    private final int stateSize;
    private final int hiddenSize;
    private final double learningRate;
    private final double gamma;
    private final double lambda;
    private final double kappa;
    
    // 键ID编码参数
    private static final int KEY_ENCODING_BITS = 16; // 使用多少位编码键ID

    /**
     * 构造函数
     * 
     * @param numOfWorkers 工作节点数量
     * @param slide 滑动窗口滑动步长
     * @param size 滑动窗口大小
     * @param numOfKeys 预估键数量
     */
    public SARSA(int numOfWorkers, int slide, int size, int numOfKeys) {
        this.numOfWorkers = numOfWorkers;
        this.epsilon = 0.1;
        this.state = new State(size, slide, numOfWorkers, numOfKeys);
        this.hotKeys = new HashSet<>();
        
        // 初始化参数
        this.hiddenSize = 64;
        this.learningRate = 0.01;
        this.gamma = 0.99;
        this.lambda = 0;
        this.kappa = 0.01;
        
        // 状态维度：负载 + 分片 + 键编码
        this.stateSize = numOfWorkers + 1 + KEY_ENCODING_BITS;
        
        // 初始化全局模型
        initializeModel(this.stateSize, numOfWorkers);
    }
    
    /**
     * 在序列化前保存网络权重
     */
    private void writeObject(java.io.ObjectOutputStream out) throws java.io.IOException {
        // 在序列化前保存网络权重
        if (model != null) {
            model.saveWeights();
        }
        
        // 默认序列化
        out.defaultWriteObject();
    }
    
    /**
     * 在反序列化后加载网络权重
     */
    private void readObject(java.io.ObjectInputStream in) throws java.io.IOException, ClassNotFoundException {
        // 默认反序列化
        in.defaultReadObject();
        
        // 加载网络权重
        if (model != null) {
            model.loadWeights();
        }
    }

    /**
     * 基础哈希函数作为冷键分区方法
     * 
     * @param key 键ID
     * @return 工作节点ID
     */
    public int hash(int key) {
        return key % numOfWorkers;
    }

    /**
     * 检测记录是否为热点
     * 
     * @param r 输入记录
     * @param topKeys 热点键列表
     * @return 是否为热点
     */
    public boolean isHot(Record r, List<DaltonCooperative.Frequency> topKeys) {
        int result = state.isHot(r, topKeys);
        if (result == 0) { // 非热点
            if (hotKeys.contains(r.getKeyId())) {
                // 检查是否过期
                if (r.getTs() > state.getExpirationTs()) {
                    hotKeys.remove(r.getKeyId());
                    return false;
                }
                return true;
            }
            return false;
        } else if (result != 1) { // 新热点
            if (!hotKeys.contains(r.getKeyId())) {
                // 标记为热点键
                hotKeys.add(r.getKeyId());
            }
        }
        // result == 1 表示已经是热点
        return true;
    }
    
    /**
     * 简化版的热点检测方法
     * 
     * @param r 输入记录
     * @return 是否为热点
     */
    public boolean isHot(Record r) {
        return isHot(r, null);
    }
    
    /**
     * 初始化SARSA模型
     * 
     * @param stateSize 状态空间维度
     * @param actionSize 动作空间维度
     */
    private void initializeModel(int stateSize, int actionSize) {
        model = new StreamSARSA(
            stateSize,         // 观察空间维度：负载 + 分片 + 键编码
            actionSize,        // 动作空间维度：可选worker数量
            hiddenSize,        // 隐藏层大小
            learningRate,      // 学习率
            0.01,              // 最终探索率
            epsilon,           // 初始探索率
            0.5,               // 探索衰减率
            1000000,           // 总步数估计
            gamma,             // 折扣因子
            lambda,            // λ值
            kappa              // 权重衰减
        );
    }

    /**
     * 更新状态过期信息
     * 
     * @param r 输入记录
     * @param isHot 是否为热点
     */
    public void expireState(Record r, boolean isHot) {
        state.updateExpired(r, isHot);
    }

    /**
     * 根据记录是否为热点选择分区策略
     * 
     * @param r 输入记录
     * @param isHot 是否为热点
     * @return 工作节点ID
     */
    public int partition(Record r, boolean isHot) {
        //return hash(r.getKeyId());
        return isHot ? partitionHot(r) : hash(r.getKeyId());
    }

    /**
     * 将键ID编码为二进制特征
     * 
     * @param keyId 键ID
     * @return 编码后的特征数组
     */
    private double[] encodeKey(int keyId) {
        double[] encoding = new double[KEY_ENCODING_BITS];
        
        // 简单的二进制编码
        for (int i = 0; i < KEY_ENCODING_BITS; i++) {
            encoding[i] = ((keyId >> i) & 1);
        }
        
        return encoding;
    }
    
    /**
     * 构建完整状态向量（包括负载、分片和键ID编码）
     * 
     * @param r 输入记录
     * @return 状态向量
     */
    private double[] buildStateVector(Record r) {
        // 状态向量维度：负载 + 分片 + 键编码
        int stateSize = numOfWorkers + 1 + KEY_ENCODING_BITS;
        double[] stateVector = new double[stateSize];
        
        // 1. 负载信息
        for (int i = 0; i < numOfWorkers; i++) {
            stateVector[i] = state.getLoad(i);
        }
        
        // 2. 分片情况
        BitSet fragmentation = state.keyfragmentation(r.getKeyId());
        stateVector[numOfWorkers] = (double)fragmentation.cardinality() / numOfWorkers;
        
        // 3. 键ID编码
        double[] keyEncoding = encodeKey(r.getKeyId());
        System.arraycopy(keyEncoding, 0, stateVector, numOfWorkers + 1, KEY_ENCODING_BITS);
        
        return stateVector;
    }

    /**
     * 热点记录分区决策
     * 
     * @param r 输入记录
     * @return 选择的工作节点ID
     */
    public int partitionHot(Record r) {
        // 构建包含键ID的状态向量
        double[] stateVector = buildStateVector(r);
        
        // 使用模型选择动作
        int worker = model.sampleAction(stateVector);
        return worker;
    }

    /**
     * 更新内部状态
     * 
     * @param r 输入记录
     * @param worker 选择的工作节点
     */
    public void updateState(Record r, int worker) {
        state.update(r, worker);
    }

    /**
     * 更新学习模型
     * 保持与ContextualBandits兼容的方法名
     * 
     * @param r 输入记录
     * @param isHot 是否为热点
     * @param worker 选择的工作节点
     * @return 计算的奖励值
     */
    public double updateQtable(Record r, boolean isHot, int worker) {
        if (!isHot) {
            return -3.0;
        }
        
        // 构建当前状态向量
        double[] currentState = buildStateVector(r);
        
        // 计算奖励
        double reward = state.reward(r.getKeyId(), worker, hotKeys.size());
        
        // 模拟更新后的状态
        BitSet fragmentation = state.keyfragmentation(r.getKeyId());
        boolean isNewWorker = !fragmentation.get(worker);
        
        // 预测下一个状态
        double[] nextState = new double[currentState.length];
        System.arraycopy(currentState, 0, nextState, 0, currentState.length);
        
        // 更新负载部分
        nextState[worker] = currentState[worker] + 1.0/state.avgLoad(); // 模拟增加负载
        
        // 更新分片部分
        int nextFragmentation = isNewWorker ? 
                                fragmentation.cardinality() + 1 : 
                                fragmentation.cardinality();
        nextState[numOfWorkers] = (double)nextFragmentation / numOfWorkers;
        
        // 键编码部分保持不变
        
        // 选择下一个动作
        int nextWorker = model.sampleAction(nextState);
        
        // 更新SARSA模型
        try {
            model.updateParams(currentState, worker, reward, nextState, nextWorker, false);
        } catch (Exception e) {
            // 捕获并记录可能的学习错误，防止影响主流程
            System.err.println("SARSA学习更新出错: " + e.getMessage());
        }
        
        return reward;
    }
    
    /**
     * 获取内部状态
     * 
     * @return 状态对象
     */
    public State getState() {
        return state;
    }

    /**
     * 获取记录总数
     * 
     * @return 记录总数
     */
    public int getTotalCountOfRecords() {
        return state.getTotalCountOfRecords();
    }

    /**
     * 设置频率阈值
     * 
     * @param t 阈值
     */
    public void setFrequencyThreshold(int t) {
        state.setFrequencyThreshold(t);
    }

    /**
     * 设置热点检测间隔
     * 
     * @param h 间隔
     */
    public void setHotInterval(int h) {
        state.setHotInterval(h);
    }
    
    /**
     * 获取当前热点键数量
     * 
     * @return 热点键数量
     */
    public int getHotKeysCount() {
        return hotKeys.size();
    }

    /**
     * 获取底层的StreamSARSA模型
     * 
     * @return StreamSARSA模型实例
     */
    public StreamSARSA getSarsa() {
        return model;
    }
} 