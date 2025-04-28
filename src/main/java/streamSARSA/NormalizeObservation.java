package streamSARSA;

/**
 * 观察值归一化包装器
 * 对应Python的NormalizeObservation类
 */
public class NormalizeObservation {
    private SampleMeanStd obsStats;
    private double epsilon;
    
    /**
     * 构造函数
     * @param obsShape 观察值的形状
     * @param epsilon 用于避免除零错误的小常数
     */
    public NormalizeObservation(int obsShape, double epsilon) {
        this.obsStats = new SampleMeanStd(obsShape);
        this.epsilon = epsilon;
    }
    
    /**
     * 归一化观察值
     * @param obs 原始观察值
     * @return 归一化后的观察值
     */
    public double[] process(double[] obs) {
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
        return obsStats.normalize(obs, epsilon);
    }
} 