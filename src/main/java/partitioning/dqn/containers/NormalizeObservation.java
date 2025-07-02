package partitioning.dqn.containers;

import java.io.Serializable;

/**
 * 观察值归一化包装器
 * 使用普通数组计算，不依赖INDArray
 */
public class NormalizeObservation implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private SampleMeanStd obsStats;
    private double epsilon;
    private int obsShape;
    
    /**
     * 构造函数
     * @param obsShape 观察值的维度
     * @param epsilon 用于避免除零错误的小常数
     */
    public NormalizeObservation(int obsShape, double epsilon) {
        this.obsShape = obsShape;
        this.obsStats = new SampleMeanStd(obsShape);
        this.epsilon = epsilon;
    }
    
    /**
     * 归一化观察值
     * @param obs 原始观察值
     * @return 归一化后的观察值
     */
    public double[] process(double[] obs) {
        if (obs.length != obsShape) {
            throw new IllegalArgumentException("观察值维度不匹配: " + obs.length + " vs " + obsShape);
        }
        
        // 更新统计量并归一化
        obsStats.update(obs);
        return obsStats.normalize(obs, epsilon);
    }
    
    /**
     * 不更新统计量，仅归一化观察值（用于测试或评估阶段）
     * @param obs 原始观察值
     * @return 归一化后的观察值
     */
    public double[] normalizeOnly(double[] obs) {
        if (obs.length != obsShape) {
            throw new IllegalArgumentException("观察值维度不匹配: " + obs.length + " vs " + obsShape);
        }
        
        return obsStats.normalize(obs, epsilon);
    }
    
    /**
     * 获取观察值统计信息
     * @return SampleMeanStd统计对象
     */
    public SampleMeanStd getObsStats() {
        return obsStats;
    }
    
    /**
     * 获取观察值的维度
     */
    public int getObsShape() {
        return obsShape;
    }
    
    /**
     * 获取处理的观察值样本数量
     */
    public long getObsCount() {
        return obsStats.getCount();
    }
    
    /**
     * 获取当前观察值的均值
     */
    public double[] getObsMean() {
        return obsStats.getMean();
    }
    
    /**
     * 获取当前观察值的方差
     */
    public double[] getObsVar() {
        return obsStats.getVar();
    }
    
    /**
     * 获取当前观察值的标准差
     */
    public double[] getObsStd() {
        return obsStats.getStd(epsilon);
    }
    
    /**
     * 重置所有统计量
     */
    public void reset() {
        obsStats.reset();
    }
} 