package streamSARSA;

import org.deeplearning4j.nn.api.*;
import org.deeplearning4j.nn.gradient.Gradient;
import org.deeplearning4j.nn.workspace.LayerWorkspaceMgr;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.linalg.learning.config.IUpdater;
import org.nd4j.linalg.learning.GradientUpdater;
import org.nd4j.linalg.schedule.ISchedule;
import org.deeplearning4j.optimize.solvers.BaseOptimizer;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;

/**
 * ObGD优化器实现类，实现IUpdater接口以便与DL4J的配置系统集成
 * 采用资格迹和自适应学习率的基于观察的梯度下降
 */
public class ObGDUpdater implements IUpdater {
    /** 学习率：控制参数更新的步长大小 */
    private final double lr;
    
    /** gamma系数：控制SARSA学习算法的折扣因子，决定未来奖励的重要性 */
    private final double gamma;
    
    /** lambda系数：控制资格迹衰减率，影响历史梯度的累积权重 */
    private final double lamda;
    
    /** kappa系数：控制动态步长调整的强度，防止更新步长过大 */
    private final double kappa;
    
    /** 是否使用自适应学习率模式 */
    private final boolean adaptive;
    
    /** beta2系数：控制二阶矩估计的指数衰减率，仅在自适应模式下使用 */
    private final double beta2;
    
    /** epsilon系数：防止除零错误的小常数，确保数值稳定性 */
    private final double eps;

    // 基础构造器，对应Python的ObGD类
    public ObGDUpdater(double lr, double gamma, double lamda, double kappa) {
        this(lr, gamma, lamda, kappa, 0.0, 0.0, false);
    }

    // 自适应构造器，对应Python的AdaptiveObGD类
    public ObGDUpdater(double lr, double gamma, double lamda,
                      double kappa, double beta2, double eps) {
        this(lr, gamma, lamda, kappa, beta2, eps, true);
    }

    private ObGDUpdater(double lr, double gamma, double lamda, double kappa,
                       double beta2, double eps, boolean adaptive) {
        this.lr = lr;
        this.gamma = gamma;
        this.lamda = lamda;
        this.kappa = kappa;
        this.adaptive = adaptive;
        this.beta2 = beta2;
        this.eps = eps;
    }

    /**
     * 创建一个新的GradientUpdater实例，用于实际更新参数
     */
    @Override
    public GradientUpdater instantiate(Map<String, INDArray> paramTable, boolean initializeStates) {
        ObGDGradientUpdater updater = new ObGDGradientUpdater(this);
        if (initializeStates) {
            for (Map.Entry<String, INDArray> entry : paramTable.entrySet()) {
                String key = entry.getKey();
                INDArray param = entry.getValue();
                
                // 获取参数形状，并在主内存中创建资格迹数组
                long[] shape = param.shape();
                
                // 初始化资格迹
                updater.getState().put(key + "_e", Nd4j.zeros(shape).detach());
                
                // 如果是自适应模式，初始化二阶矩估计
                if (adaptive) {
                    updater.getState().put(key + "_v", Nd4j.zeros(shape).detach());
                }
            }
        }
        return updater;
    }

    /**
     * 创建一个新的GradientUpdater实例，使用状态视图数组初始化
     */
    @Override
    public GradientUpdater instantiate(INDArray viewArray, boolean initialize) {
        ObGDGradientUpdater updater = new ObGDGradientUpdater(this);
        
        if (initialize) {
            // 如果需要初始化状态，将视图数组设置为0
            viewArray.assign(0);
        }
        
        updater.setStateViewArray(viewArray, new long[]{viewArray.length()}, 'c', initialize);
        
        return updater;
    }

    /**
     * 计算每个参数需要的状态大小
     * 对于标准模式，每个参数只需要一个状态（e）
     * 对于自适应模式，每个参数需要两个状态（e和v）
     */
    @Override
    public long stateSize(long numParams) {
        // 自适应模式需要两倍的空间（e和v）
        return adaptive ? 2 * numParams : numParams;
    }

    @Override
    public IUpdater clone() {
        return new ObGDUpdater(lr, gamma, lamda, kappa, beta2, eps, adaptive);
    }

    @Override
    public double getLearningRate(int iteration, int epoch) {
        return lr;
    }

    @Override
    public void setLrAndSchedule(double lr, ISchedule lrSchedule) {
        throw new UnsupportedOperationException("ObGDUpdater does not support learning rate schedules");
    }

    @Override
    public boolean hasLearningRate() {
        return true;
    }

    // 获取配置信息
    public Map<String, Object> getConfig() {
        Map<String, Object> config = new HashMap<>();
        config.put("lr", lr);
        config.put("gamma", gamma);
        config.put("lambda", lamda);
        config.put("kappa", kappa);
        config.put("adaptive", adaptive);
        if (adaptive) {
            config.put("beta2", beta2);
            config.put("eps", eps);
        }
        return config;
    }

    @Override
    public String toString() {
        return "ObGDUpdater(lr=" + lr + ", gamma=" + gamma + ", lambda=" + lamda + 
            ", kappa=" + kappa + ", adaptive=" + adaptive + 
            (adaptive ? ", beta2=" + beta2 + ", eps=" + eps : "") + ")";
    }

    /**
     * 内部GradientUpdater实现，处理实际的梯度更新逻辑
     * 设计为避免工作空间问题的实现
     */
    public static class ObGDGradientUpdater implements GradientUpdater<ObGDUpdater> {
        private final ObGDUpdater config;
        private final Map<String, INDArray> states;
        private INDArray stateView;
        private int counter = 0;
        private double globalZSum = 0.0;  // 用于存储全局z_sum
        private double globalStepSize = 1.0;  // 用于存储全局步长

        public ObGDGradientUpdater(ObGDUpdater config) {
            this.config = config;
            this.states = new HashMap<>();
        }

        /**
         * 实现DL4J的applyUpdater接口
         * 使用默认delta=1.0和不重置资格迹
         */
        @Override
        public void applyUpdater(INDArray gradient, int iteration, int epoch) {
            // 使用默认delta=1.0和不重置资格迹
            applyUpdater(gradient, 1.0, false);
        }
        
        /**
         * 计算全局z_sum
         * @param delta TD误差值
         */
        private void computeGlobalZSum(double delta) {
            globalZSum = 0.0;
            
            // 遍历所有状态，计算每个参数的z_sum贡献
            for (Map.Entry<String, INDArray> entry : states.entrySet()) {
                String key = entry.getKey();
                if (key.endsWith("_e")) {  // 只处理资格迹状态
                    INDArray e = entry.getValue();
                    double zSum;
                    
                    if (config.adaptive) {
                        // 获取对应的二阶矩估计
                        String vKey = key.substring(0, key.length() - 2) + "_v";
                        INDArray v = states.get(vKey);
                        
                        // 计算偏差修正
                        INDArray vHat = v.div(1 - Math.pow(config.beta2, counter));
                        INDArray denominator = Transforms.sqrt(vHat.add(config.eps), false);
                        
                        // 计算z_sum贡献
                        zSum = Transforms.abs(e.div(denominator)).sumNumber().doubleValue();
                    } else {
                        // 标准模式计算z_sum贡献
                        zSum = Transforms.abs(e).sumNumber().doubleValue();
                    }
                    
                    // 累加到全局z_sum
                    globalZSum += zSum;
                }
            }
            
            // 计算全局步长
            double deltaBar = Math.max(Math.abs(delta), 1.0);
            double dotProduct = deltaBar * globalZSum * config.kappa;
            globalStepSize = (dotProduct > 1.0) ? 1.0 / dotProduct : 1.0;
        }
        
        /**
         * 实现基于观察的梯度更新，包含delta参数
         * 对应Python的step方法
         * @param gradient 当前参数的梯度
         * @param delta TD误差值
         * @param reset 是否重置资格迹
         */
        public void applyUpdater(INDArray gradient, double delta, boolean reset) {
            counter++;
            
            // 在主内存中复制梯度，以避免工作空间问题
            INDArray gradientCopy = gradient.dup('c').detach();
            
            // 为当前梯度获取或创建资格迹状态
            String gradientKey = "g" + counter;
            INDArray e = getOrCreateState(gradientKey + "_e", gradientCopy.shape());
            
            // 更新资格迹: e = gamma*lambda*e + gradient
            e.muli(config.gamma * config.lamda).addi(gradientCopy);
            
            // 第一阶段：计算全局z_sum（所有参数资格迹的贡献总和）
            globalZSum = 0.0;
            
            // 遍历所有状态，计算每个参数的z_sum贡献
            for (Map.Entry<String, INDArray> entry : states.entrySet()) {
                String key = entry.getKey();
                if (key.endsWith("_e")) {  // 只处理资格迹状态
                    INDArray trace = entry.getValue();
                    double zSum;
                    
                    if (config.adaptive) {
                        // 获取对应的二阶矩估计
                        String vKey = key.substring(0, key.length() - 2) + "_v";
                        INDArray v = states.get(vKey);
                        
                        // 计算偏差修正
                        INDArray vHat = v.div(1 - Math.pow(config.beta2, counter));
                        INDArray denominator = Transforms.sqrt(vHat.add(config.eps), false);
                        
                        // 计算z_sum贡献
                        zSum = Transforms.abs(trace.div(denominator)).sumNumber().doubleValue();
                    } else {
                        // 标准模式计算z_sum贡献
                        zSum = Transforms.abs(trace).sumNumber().doubleValue();
                    }
                    
                    // 累加到全局z_sum
                    globalZSum += zSum;
                }
            }
            
            // 计算全局步长
            double deltaBar = Math.max(Math.abs(delta), 1.0);
            double dotProduct = deltaBar * globalZSum * config.kappa * config.lr;
            globalStepSize = (dotProduct > 1.0) ? config.lr / dotProduct : config.lr;
            
            // 第二阶段：使用计算出的步长更新参数
            // 计算更新值
            INDArray updateValue;
            if (config.adaptive) {
                // 为当前梯度获取或创建二阶矩估计
                String vKey = gradientKey + "_v";
                INDArray v = getOrCreateState(vKey, gradientCopy.shape());
                
                // 更新二阶矩估计：v = beta2*v + (1-beta2)*(delta*e)^2
                INDArray deltaE = e.mul(delta);
                v.muli(config.beta2).addi(deltaE.mul(deltaE).muli(1.0 - config.beta2));
                
                // 计算偏差修正
                INDArray vHat = v.div(1 - Math.pow(config.beta2, counter));
                INDArray denominator = Transforms.sqrt(vHat.add(config.eps), false);
                
                // 计算最终更新值
                updateValue = deltaE.div(denominator).muli(-globalStepSize);
            } else {
                // 标准模式计算更新值
                updateValue = e.mul(-delta * globalStepSize);
            }
            
            // 设置梯度（即为最终的更新值）
            gradient.assign(updateValue);
            
            // 如果需要重置资格迹
            if (reset) {
                e.assign(0);
                if (config.adaptive) {
                    INDArray v = states.get(gradientKey + "_v");
                    if (v != null) {
                        v.assign(0);
                    }
                }
            }
        }

        @Override
        public ObGDUpdater getConfig() {
            return config;
        }
        
        @Override
        public Map<String, INDArray> getState() {
            return states;
        }
        
        @Override
        public void setState(Map<String, INDArray> stateMap, boolean initialize) {
            if(initialize) {
                for(String key : stateMap.keySet()) {
                    INDArray arr = stateMap.get(key);
                    if (arr != null) {
                        arr.assign(0);
                    }
                }
            }
            this.states.clear();
            this.states.putAll(stateMap);
        }
        
        @Override
        public void setStateViewArray(INDArray viewArray, long[] gradientShape, char order, boolean initialize) {
            this.stateView = viewArray;
            if (initialize) {
                viewArray.assign(0);
            }
        }
        
        /**
         * 获取或创建状态变量，确保状态变量与工作空间无关
         */
        private INDArray getOrCreateState(String key, long[] shape) {
            INDArray state = states.get(key);
            if (state == null) {
                // 在主内存中创建状态变量，确保与工作空间无关
                state = Nd4j.zeros(shape).detach();
                states.put(key, state);
            } else if (!Arrays.equals(state.shape(), shape)) {
                // 如果形状不匹配，重新创建状态数组
                state = Nd4j.zeros(shape).detach();
                states.put(key, state);
            }
            return state;
        }
    }

    // 工厂方法用于创建更新器实例
    public static ObGDUpdater create(double lr, double gamma, double lamda, double kappa) {
        return new ObGDUpdater(lr, gamma, lamda, kappa);
    }

    public static ObGDUpdater createAdaptive(double lr, double gamma, double lamda, 
                                           double kappa, double beta2, double eps) {
        return new ObGDUpdater(lr, gamma, lamda, kappa, beta2, eps);
    }
}