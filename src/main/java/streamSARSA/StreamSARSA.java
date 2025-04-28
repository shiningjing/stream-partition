package streamSARSA;

import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;
import org.nd4j.autodiff.samediff.TrainingConfig;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.GradientUpdater;

import java.io.Serializable;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Arrays;
import java.util.Collections;

/**
 * Java实现的StreamSARSA，使用SameDiff构建网络
 */
public class StreamSARSA implements Serializable {
    private static final long serialVersionUID = 1L;
    
    /** SameDiff计算图：用于定义和执行神经网络计算 */
    private transient SameDiff sd;
    
    /** 输入变量：网络的输入占位符 */
    private transient SDVariable input;
    
    /** 输出变量：网络的输出计算结果 */
    private transient SDVariable output;
    
    /** 标签变量：用于训练的标签占位符 */
    private transient SDVariable labels;
    
    /** 下一个状态的输出变量 */
    private transient SDVariable nextOutput;
    
    /** 差异变量 */
    private transient SDVariable diff;
    
    /** 平方差异变量 */
    private transient SDVariable squaredDiff;
    
    /** 损失变量 */
    private transient SDVariable loss;
    
    /** 输入形状：输入状态的维度 */
    private final long[] inputShape;
    
    /** 输出形状：网络输出的维度 */
    private final long[] outputShape;
    
    /** 动作空间大小：可选动作的数量 */
    private final int nActions;
    
    /** 隐藏层大小 */
    private final int hiddenSize;
    
    /** 折扣因子：控制未来奖励的重要性，范围通常为0到1 */
    private double gamma;
    
    /** 初始探索率：epsilon-greedy策略的起始探索概率 */
    private double epsilonStart;
    
    /** 目标探索率：epsilon-greedy策略的最终探索概率 */
    private double epsilonTarget;
    
    /** 当前探索率：随时间调整的实时探索概率 */
    private double epsilon;
    
    /** 探索分数：控制探索率从初始值到目标值的衰减速度 */
    private double explorationFraction;
    
    /** 总步数：算法预计运行的总时间步数 */
    private long totalSteps;
    
    /** 当前时间步：记录当前已执行的步数，用于调整探索率 */
    private long timeStep;
    
    /** 随机数生成器：用于epsilon-greedy策略中的随机决策 */
    private transient Random random;
    
    /** 学习率：控制参数更新的步长大小 */
    private double learningRate;
    
    /** lambda参数：控制资格迹衰减率 */
    private double lambda;
    
    /** kappa参数：控制动态步长调整的强度 */
    private double kappa;
    
    /** 参数梯度记录：存储上一次参数更新的梯度信息 */
    private transient Map<String, INDArray> lastGradients;
    
    /** 参数累积梯度：实现资格迹的累积梯度 */
    private transient Map<String, INDArray> eligibilityTraces;
    
    /** 锁定参数集合：这些参数在训练过程中不会被更新 */
    private Set<String> lockedParams;
    
    /** 存储网络权重的可序列化数据结构 */
    private Map<String, float[]> serializedWeights;

    /**
     * 构造函数
     */
    public StreamSARSA(int nObs, int nActions, int hiddenSize, double learningRate,
                      double epsilonTarget, double epsilonStart, double explorationFraction,
                      long totalSteps, double gamma, double lambda, double kappa) {
        this.nActions = nActions;
        this.hiddenSize = hiddenSize;
        this.gamma = gamma;
        this.epsilonStart = epsilonStart;
        this.epsilonTarget = epsilonTarget;
        this.epsilon = epsilonStart;
        this.explorationFraction = explorationFraction;
        this.totalSteps = totalSteps;
        this.timeStep = 0;
        this.random = new Random();
        this.lambda = lambda;
        this.kappa = kappa;
        this.learningRate = learningRate;
        this.lastGradients = new HashMap<>();
        this.eligibilityTraces = new HashMap<>();
        this.inputShape = new long[]{1, nObs};
        this.outputShape = new long[]{1, nActions};
        this.lockedParams = new HashSet<>();
        this.serializedWeights = new HashMap<>();
        this.sd = SameDiff.create();

        // 创建SameDiff计算图
        buildNetwork();
        
        // 应用稀疏初始化
        initializeWeights(0.6);
        
        // 锁定层归一化参数，使其在训练过程中不更新
        lockLayerNormParams();
    }

    /**
     * 锁定层归一化参数，使其在训练过程中保持不变
     */
    private void lockLayerNormParams() {
        // 锁定第一个层归一化层的参数
        lockedParams.add("scale1");
        lockedParams.add("offset1");
        
        // 锁定第二个层归一化层的参数
        lockedParams.add("scale2");
        lockedParams.add("offset2");
        
    }

    /**
     * 构建神经网络计算图
     */
    private void buildNetwork() {
        // 输入占位符（显式指定形状）
        input = sd.placeHolder("input", Nd4j.dataType(), inputShape);
        
        // 标签占位符
        labels = sd.placeHolder("labels", Nd4j.dataType(), outputShape);
        
        // 下一个状态的输出变量
        nextOutput = sd.var("nextOutput", Nd4j.zeros(outputShape));
        
        // 差异变量
        diff = nextOutput.sub(labels).rename("diff");
        
        // 平方差异变量
        squaredDiff = diff.mul(diff).rename("squared_diff");
        
        // 损失变量
        loss = squaredDiff.mean().rename("loss");

        // 第一层隐藏层
        SDVariable w1 = sd.var("w1", Nd4j.randn(inputShape[1], hiddenSize));
        SDVariable b1 = sd.var("b1", Nd4j.zeros(hiddenSize));
        SDVariable h1 = sd.nn().leakyRelu(sd.nn().linear(input, w1, b1), 0.01f);

        // 层归一化（显式指定归一化轴）
        SDVariable scale1 = sd.var("scale1", Nd4j.ones(hiddenSize));
        SDVariable offset1 = sd.var("offset1", Nd4j.zeros(hiddenSize));
        SDVariable h2 = sd.nn().layerNorm(h1, scale1, offset1, false, 1);  // 显式指定输出形状

        // 第二层隐藏层
        SDVariable w2 = sd.var("w2", Nd4j.randn(hiddenSize, hiddenSize));
        SDVariable b2 = sd.var("b2", Nd4j.zeros(hiddenSize));
        SDVariable h3 = sd.nn().leakyRelu(sd.nn().linear(h2, w2, b2), 0.01f);

        // 层归一化（显式指定归一化轴）
        SDVariable scale2 = sd.var("scale2", Nd4j.ones(hiddenSize));
        SDVariable offset2 = sd.var("offset2", Nd4j.zeros(hiddenSize));
        SDVariable h4 = sd.nn().layerNorm(h3, scale2, offset2, false, 1);  // 显式指定输出形状

        // 输出层
        SDVariable w3 = sd.var("w3", Nd4j.randn(hiddenSize, nActions));
        SDVariable b3 = sd.var("b3", Nd4j.zeros(nActions));
        output = sd.nn().linear(h4, w3, b3);  // 使用类属性指定输出形状
        output = output.rename("output");
        sd.setLossVariables("loss");
        // 执行一次前向传播以初始化所有变量
        try {
            // 创建随机输入数据
            INDArray randomInput = Nd4j.randn(inputShape);
            INDArray randomLabels = Nd4j.randn(outputShape);
            
            // 设置占位符
            Map<String, INDArray> placeholders = new HashMap<>();
            placeholders.put("input", randomInput);
            placeholders.put("labels", randomLabels);
            
            // 执行前向传播
            sd.output(placeholders, "output");
        } catch (Exception e) {
            throw new RuntimeException("网络初始化失败：" + e.getMessage());
        }
    }
    
    /**
     * 线性调度函数，对应Python的linear_schedule
     */
    private double linearSchedule(double startE, double endE, long duration, long t) {
        double slope = (endE - startE) / duration;
        return Math.max(slope * t + startE, endE);
    }

    /**
     * 初始化权重为稀疏矩阵
     */
    private void initializeWeights(double sparsity) {
        // 获取网络所有权重变量
        for (String name : sd.variableMap().keySet()) {
            SDVariable variable = sd.getVariable(name);
            if (name.startsWith("w")) {
                // 获取当前权重
                INDArray weights = variable.getArr();
                if (weights != null) {
                    // 使用SparseInitializer初始化权重
                    INDArray sparseWeights = SparseInitializer.sparseInit(
                            weights, sparsity, "uniform");
                    variable.setArray(sparseWeights);
                }
            } else if (name.startsWith("b")) {
                // 确保偏置初始化为0
                INDArray bias = variable.getArr();
                if (bias != null) {
                    variable.setArray(Nd4j.zeros(bias.shape()));
                }
            }
        }
    }

    /**
     * 确保输入状态为二维数组（添加批次维度）
     */
    private INDArray ensureBatchDimension(INDArray state) {
        if (state.rank() == 1) {
            // 如果输入是一维的，则添加批次维度（转为形状为[1, features]的矩阵）
            return state.reshape(1, state.length());
        }
        return state;
    }

    /**
     * 保存网络权重为可序列化格式
     */
    public void saveWeights() {
        serializedWeights.clear();
        
        // 保存所有变量的权重
        for (String name : sd.variableMap().keySet()) {
            SDVariable variable = sd.getVariable(name);
            if (variable != null && variable.getArr() != null) {
                // 将INDArray转换为float数组
                INDArray array = variable.getArr();
                float[] data = array.data().asFloat();
                long[] shape = array.shape();
                
                // 保存权重数据和形状信息
                serializedWeights.put(name, data);
                serializedWeights.put(name + "_shape", toFloatArray(shape));
            }
        }
    }
    
    /**
     * 从可序列化格式加载网络权重
     */
    public void loadWeights() {
        if (sd == null) {
            // 如果SameDiff对象不存在，重新创建
            sd = SameDiff.create();
            buildNetwork();
        }
        
        if (serializedWeights.isEmpty()) {
            return;
        }
        
        // 恢复所有变量的权重
        for (String name : sd.variableMap().keySet()) {
            if (serializedWeights.containsKey(name)) {
                float[] data = serializedWeights.get(name);
                float[] shapeData = serializedWeights.get(name + "_shape");
                
                if (data != null && shapeData != null) {
                    // 从float数组重建形状
                    long[] shape = toLongArray(shapeData);
                    
                    // 从数据和形状创建INDArray
                    INDArray array = Nd4j.create(data, shape);
                    
                    // 设置回变量
                    sd.getVariable(name).setArray(array);
                }
            }
        }
        
        // 重新初始化transient变量
        if (random == null) {
            random = new Random();
        }
        if (lastGradients == null) {
            lastGradients = new HashMap<>();
        }
        if (eligibilityTraces == null) {
            eligibilityTraces = new HashMap<>();
        }
    }
    
    /**
     * 辅助方法：将long数组转换为float数组
     */
    private float[] toFloatArray(long[] array) {
        float[] result = new float[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = array[i];
        }
        return result;
    }
    
    /**
     * 辅助方法：将float数组转换为long数组
     */
    private long[] toLongArray(float[] array) {
        long[] result = new long[array.length];
        for (int i = 0; i < array.length; i++) {
            result[i] = (long) array[i];
        }
        return result;
    }
    
    /**
     * 确保所有transient对象已经初始化
     */
    private void ensureInitialized() {
        if (sd == null) {
            loadWeights();
        }
    }
    
    /**
     * 计算Q值，对应Python的q方法
     */
    public INDArray q(INDArray state) {
        ensureInitialized();
        
        // 确保输入状态包含批次维度
        INDArray batchState = ensureBatchDimension(state);
        
        // 执行前向传播
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("input", batchState);
        // 使用随机标签进行前向传播
        placeholders.put("labels", Nd4j.zeros(outputShape));
        Map<String, INDArray> outputs = sd.outputAll(placeholders);
        
        return outputs.get("output");
    }

    /**
     * 采样动作，对应Python的sample_action方法
     */
    public int sampleAction(double[] state) {
        ensureInitialized();
        
        timeStep++;
        epsilon = linearSchedule(epsilonStart, epsilonTarget, 
                (long)(explorationFraction * totalSteps), timeStep);

        // 创建状态张量
        INDArray stateArray = Nd4j.create(state);

        // Epsilon-greedy策略
        if (random == null) {
            random = new Random();
        }
        
        if (random.nextDouble() < epsilon) {
            return random.nextInt(nActions);
        } else {
            INDArray qValues = q(stateArray);
            return qValues.argMax(1).getInt(0);
        }
    }

    /**
     * 更新参数，对应Python的update_params方法
     */
    public void updateParams(double[] s, int a, double r, double[] sPrime, int aPrime, boolean done) {
        updateParams(s, a, r, sPrime, aPrime, done, false);
    }

    /**
     * 带有超前检测的更新参数
     */
    public void updateParams(double[] s, int a, double r, double[] sPrime, int aPrime, boolean done, boolean overshooting) {
        // 转换为ND4J张量
        INDArray stateArray = Nd4j.create(s);
        INDArray nextStateArray = Nd4j.create(sPrime);
        double doneMask = done ? 0.0 : 1.0;

        // 确保输入状态包含批次维度
        INDArray batchState = ensureBatchDimension(stateArray);
        INDArray batchNextState = ensureBatchDimension(nextStateArray);
        
        // 计算当前状态动作对的Q值
        INDArray qValues = q(stateArray);
        double qSA = qValues.getDouble(0, a);

        // 计算下一个状态动作对的Q值
        INDArray nextQValues = q(nextStateArray);
        double qSPrimeAPrime = nextQValues.getDouble(0, aPrime);

        // 计算TD目标和误差
        double tdTarget = r + gamma * qSPrimeAPrime * doneMask;
        double delta = tdTarget - qSA;

        // 创建目标Q值
        INDArray targetValues = qValues.dup();
        targetValues.putScalar(0, a, tdTarget);
        
        // 计算下一个状态的输出
        Map<String, INDArray> nextPlaceholders = new HashMap<>();
        nextPlaceholders.put("input", batchNextState);
        INDArray nextOutputValue = sd.output(nextPlaceholders, "output").get("output");
        
        // 更新nextOutput变量的值
        nextOutput.setArray(nextOutputValue);
        
        // 计算梯度
        Map<String, INDArray> placeholders = new HashMap<>();
        placeholders.put("input", batchState);
        placeholders.put("labels", targetValues);
        
        // 计算梯度
        Map<String, INDArray> gradients = new HashMap<>();
        try {
            // 创建变量名集合，排除未初始化的变量
            Set<String> variableNames = new HashSet<>();
            for (String varName : sd.variableMap().keySet()) {
                SDVariable var = sd.getVariable(varName);
                if (var.getArr() != null) {
                    variableNames.add(varName);
                }
            }
            
            // 排除输入和标签变量
            variableNames.remove("input");
            variableNames.remove("labels");
            
            for (String varName : variableNames) {
                Map<String, INDArray> singleGradient = sd.calculateGradients(placeholders, varName, "loss");
                gradients.putAll(singleGradient);
            }
        } catch (Exception e) {
            throw new RuntimeException("梯度计算失败：" + e.getMessage());
        }
        
        // 更新参数
        updateParamsWithEligibilityTraces(gradients, delta, done);
        
        // 处理超前检测
        if (overshooting) {
            // 重新计算Q值
            INDArray newQValues = q(stateArray);
            double newQSA = newQValues.getDouble(0, a);
            
            INDArray newNextQValues = q(nextStateArray);
            double newQSPrimeAPrime = newNextQValues.getDouble(0, aPrime);
            
            double newTdTarget = r + gamma * newQSPrimeAPrime * doneMask;
            double deltaBar = newTdTarget - newQSA;
            
            // 检测超前
            if (Math.signum(deltaBar * delta) == -1) {
                System.out.println("超前检测：发现超前误差！");
            }
        }
    }
    
    /**
     * 使用资格迹更新参数
     */
    private void updateParamsWithEligibilityTraces(Map<String, INDArray> gradients, double delta, boolean done) {
        // 保存最新梯度
        lastGradients.clear();
        lastGradients.putAll(gradients);
        
        // 创建ObGDUpdater实例
        ObGDUpdater updater = ObGDUpdater.create(learningRate, gamma, lambda, kappa);
        
        // 为每个参数创建GradientUpdater
        GradientUpdater gradientUpdater = updater.instantiate(gradients, true);
        
        // 更新每个参数
        for (Map.Entry<String, INDArray> entry : gradients.entrySet()) {
            String paramName = entry.getKey();
            
            // 如果参数被锁定，则跳过更新
            if (lockedParams.contains(paramName)) {
                continue;
            }
            
            // 获取参数变量
            SDVariable param = sd.getVariable(paramName);
            
            // 只更新参数变量（权重和偏置）
            if (param != null && param.getArr() != null && 
                (paramName.startsWith("w") || paramName.startsWith("b"))) {
                // 使用ObGDUpdater更新参数
                gradientUpdater.applyUpdater(entry.getValue(), (int)timeStep, (int)timeStep);
                
                // 更新网络参数
                param.setArray(entry.getValue());
            }
        }
        
        // 如果是终止状态，重置资格迹
        if (done) {
            for (INDArray trace : eligibilityTraces.values()) {
                trace.assign(0);
            }
        }
    }
    
    /**
     * 获取当前的探索率
     */
    public double getEpsilon() {
        return epsilon;
    }
    
    /**
     * 获取当前的时间步
     */
    public long getTimeStep() {
        return timeStep;
    }
    
    /**
     * 检查参数是否被锁定
     */
    public boolean isParamLocked(String paramName) {
        return lockedParams.contains(paramName);
    }
} 
