package partitioning.dqn.containers;

import java.io.Serializable;

/**
 * 奖励缩放包装器
 * 使用普通数值计算，不依赖INDArray
 */
public class ScaleReward implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private SampleMeanStd rewardStats;
    private double rewardTrace;
    private double gamma;
    private double epsilon;
    
    /**
     * 构造函数
     * @param gamma 折扣因子
     * @param epsilon 用于避免除零错误的小常数
     */
    public ScaleReward(double gamma, double epsilon) {
        this.rewardStats = new SampleMeanStd(1); // 奖励是单一数值
        this.rewardTrace = 0.0;
        this.gamma = gamma;
        this.epsilon = epsilon;
    }
    
    /**
     * 处理并缩放奖励
     * @param reward 原始奖励
     * @param terminated 是否终止
     * @return 缩放后的奖励
     */
    public double process(double reward, boolean terminated) {
        // 更新奖励追踪
        rewardTrace = rewardTrace * gamma * (terminated ? 0 : 1) + reward;
        
        // 更新统计量
        double[] rewardArr = new double[]{rewardTrace};
        rewardStats.update(rewardArr);
        
        // 缩放奖励
        double[] rewardVar = rewardStats.getVar();
        double stdReward = Math.sqrt(rewardVar[0] + epsilon);
        return reward / stdReward;
    }
    
    /**
     * 重置奖励追踪
     */
    public void reset() {
        rewardTrace = 0.0;
    }
    
    /**
     * 获取当前奖励追踪值
     */
    public double getRewardTrace() {
        return rewardTrace;
    }
    
    /**
     * 获取奖励统计信息
     */
    public SampleMeanStd getRewardStats() {
        return rewardStats;
    }
    
    /**
     * 获取当前奖励的标准差
     */
    public double getRewardStd() {
        double[] var = rewardStats.getVar();
        return Math.sqrt(var[0] + epsilon);
    }
    
    /**
     * 获取当前奖励的均值
     */
    public double getRewardMean() {
        double[] mean = rewardStats.getMean();
        return mean[0];
    }
    
    /**
     * 获取处理的奖励样本数量
     */
    public long getRewardCount() {
        return rewardStats.getCount();
    }
} 