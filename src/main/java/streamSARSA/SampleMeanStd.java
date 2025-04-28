package streamSARSA;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * 跟踪并计算样本的均值和方差
 * 对应Python的SampleMeanStd类
 */
public class SampleMeanStd {
    private INDArray mean; // 均值
    private INDArray var;  // 方差
    private INDArray p;    // 中间变量，累积差的平方和
    private long count;    // 样本数量
    
    /**
     * 构造函数，初始化为指定形状的数组
     */
    public SampleMeanStd(int... shape) {
        mean = Nd4j.zeros(shape);
        var = Nd4j.ones(shape);
        p = Nd4j.zeros(shape);
        count = 0;
    }
    
    /**
     * 使用新样本更新统计量
     */
    public void update(INDArray x) {
        if (count == 0) {
            mean = x.dup();
            p = Nd4j.zeros(x.shape());
        }
        
        // 更新统计量
        count++;
        INDArray delta = x.sub(mean);
        INDArray newMean = mean.add(delta.div(count));
        p = p.add(delta.mul(x.sub(newMean)));
        
        // 更新方差
        if (count < 2) {
            var = Nd4j.ones(mean.shape());
        } else {
            var = p.div(count - 1);
        }
        
        // 更新均值
        mean = newMean;
    }
    
    /**
     * 使用新样本更新统计量（数组版本）
     */
    public void update(double[] x) {
        update(Nd4j.create(x));
    }
    
    /**
     * 获取均值
     */
    public INDArray getMean() {
        return mean;
    }
    
    /**
     * 获取方差
     */
    public INDArray getVar() {
        return var;
    }
    
    /**
     * 获取样本数量
     */
    public long getCount() {
        return count;
    }
    
    /**
     * 将输入归一化：(x - mean) / sqrt(var + epsilon)
     */
    public INDArray normalize(INDArray x, double epsilon) {
        return x.sub(mean).div(Transforms.sqrt(var.add(epsilon)));
    }
    
    /**
     * 将输入归一化：(x - mean) / sqrt(var + epsilon)
     */
    public double[] normalize(double[] x, double epsilon) {
        INDArray xArr = Nd4j.create(x);
        INDArray normalized = normalize(xArr, epsilon);
        return normalized.toDoubleVector();
    }
} 