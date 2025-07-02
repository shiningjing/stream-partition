package partitioning.dqn.containers;

import java.io.Serializable;

/**
 * 跟踪并计算样本的均值和方差
 * 使用普通数组实现，不依赖INDArray
 */
public class SampleMeanStd implements Serializable {
    private static final long serialVersionUID = 1L;
    
    private double[] mean; // 均值
    private double[] var;  // 方差
    private double[] m2;   // 平方差的累积和（用于Welford算法）
    private long count;    // 样本数量
    private int size;      // 数组大小
    
    /**
     * 构造函数，初始化为指定大小的数组
     */
    public SampleMeanStd(int size) {
        this.size = size;
        this.mean = new double[size];
        this.var = new double[size];
        this.m2 = new double[size];
        this.count = 0;
        
        // 初始化方差为1
        for (int i = 0; i < size; i++) {
            var[i] = 1.0;
        }
    }
    
    /**
     * 使用新样本更新统计量（使用Welford在线算法）
     */
    public void update(double[] x) {
        if (x.length != size) {
            throw new IllegalArgumentException("输入数组大小不匹配: " + x.length + " vs " + size);
        }
        
        count++;
        
        for (int i = 0; i < size; i++) {
            // Welford在线算法更新均值和方差
            double delta = x[i] - mean[i];
            mean[i] += delta / count;
            double delta2 = x[i] - mean[i];
            m2[i] += delta * delta2;
            
            // 计算方差
            if (count > 1) {
                var[i] = m2[i] / (count - 1);
            } else {
                var[i] = 1.0; // 单个样本时方差设为1
            }
        }
    }
    
    /**
     * 获取均值的副本
     */
    public double[] getMean() {
        return mean.clone();
    }
    
    /**
     * 获取方差的副本
     */
    public double[] getVar() {
        return var.clone();
    }
    
    /**
     * 获取样本数量
     */
    public long getCount() {
        return count;
    }
    
    /**
     * 获取数组大小
     */
    public int getSize() {
        return size;
    }
    
    /**
     * 将输入归一化：(x - mean) / sqrt(var + epsilon)
     */
    public double[] normalize(double[] x, double epsilon) {
        if (x.length != size) {
            throw new IllegalArgumentException("输入数组大小不匹配: " + x.length + " vs " + size);
        }
        
        double[] normalized = new double[size];
        for (int i = 0; i < size; i++) {
            double std = Math.sqrt(var[i] + epsilon);
            normalized[i] = (x[i] - mean[i]) / std;
        }
        return normalized;
    }
    
    /**
     * 重置所有统计量
     */
    public void reset() {
        count = 0;
        for (int i = 0; i < size; i++) {
            mean[i] = 0.0;
            var[i] = 1.0;
            m2[i] = 0.0;
        }
    }
    
    /**
     * 获取标准差
     */
    public double[] getStd(double epsilon) {
        double[] std = new double[size];
        for (int i = 0; i < size; i++) {
            std[i] = Math.sqrt(var[i] + epsilon);
        }
        return std;
    }
} 