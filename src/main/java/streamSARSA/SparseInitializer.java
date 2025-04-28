package streamSARSA;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.rng.Random;

/**
 * 稀疏初始化器
 * 实现类似PyTorch的稀疏初始化功能
 */
public class SparseInitializer {
    
    /**
     * 对张量进行稀疏初始化
     * @param tensor 要初始化的张量
     * @param sparsity 稀疏度（0-1之间的值）
     * @param type 初始化类型（"uniform" 或 "normal"）
     * @return 初始化后的张量
     */
    public static INDArray sparseInit(INDArray tensor, double sparsity, String type) {
        if (tensor.rank() == 2) {
            // 2D张量（全连接层）
            long fanOut = tensor.size(0);
            long fanIn = tensor.size(1);
            
            long numZeros = (long) Math.ceil(sparsity * fanIn);
            
            // 初始化权重
            if ("uniform".equals(type)) {
                double bound = Math.sqrt(1.0 / fanIn);
                tensor.assign(Nd4j.rand(tensor.shape()).muli(2 * bound).sub(bound));
            } else if ("normal".equals(type)) {
                double std = Math.sqrt(1.0 / fanIn);
                tensor.assign(Nd4j.randn(tensor.shape()).muli(std));
            } else {
                throw new IllegalArgumentException("Unknown initialization type: " + type);
            }
            
            // 对每一列随机选择要置零的位置
            for (int colIdx = 0; colIdx < fanOut; colIdx++) {
                // 创建随机排列的索引
                long[] indices = createRandomPermutation(fanIn);
                // 将前numZeros个位置置零
                for (int i = 0; i < numZeros; i++) {
                    tensor.putScalar(new long[]{colIdx, indices[i]}, 0.0);
                }
            }
            
        } else if (tensor.rank() == 4) {
            // 4D张量（卷积层）
            long channelsOut = tensor.size(0);
            long channelsIn = tensor.size(1);
            long h = tensor.size(2);
            long w = tensor.size(3);
            
            long fanIn = channelsIn * h * w;
            long fanOut = channelsOut * h * w;
            
            long numZeros = (long) Math.ceil(sparsity * fanIn);
            
            // 初始化权重
            if ("uniform".equals(type)) {
                double bound = Math.sqrt(1.0 / fanIn);
                tensor.assign(Nd4j.rand(tensor.shape()).muli(2 * bound).sub(bound));
            } else if ("normal".equals(type)) {
                double std = Math.sqrt(1.0 / fanIn);
                tensor.assign(Nd4j.randn(tensor.shape()).muli(std));
            } else {
                throw new IllegalArgumentException("Unknown initialization type: " + type);
            }
            
            // 对每个输出通道随机选择要置零的位置
            for (int outChannelIdx = 0; outChannelIdx < channelsOut; outChannelIdx++) {
                // 创建随机排列的索引
                long[] indices = createRandomPermutation(fanIn);
                // 将前numZeros个位置置零
                for (int i = 0; i < numZeros; i++) {
                    long idx = indices[i];
                    long c = idx / (h * w);
                    long hw = idx % (h * w);
                    long height = hw / w;
                    long width = hw % w;
                    tensor.putScalar(new long[]{outChannelIdx, c, height, width}, 0.0);
                }
            }
            
        } else {
            throw new IllegalArgumentException("Only tensors with 2 or 4 dimensions are supported");
        }
        
        return tensor;
    }
    
    /**
     * 创建随机排列的索引数组
     * @param size 数组大小
     * @return 随机排列的索引数组
     */
    private static long[] createRandomPermutation(long size) {
        long[] indices = new long[(int)size];
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        
        // Fisher-Yates shuffle
        Random rng = Nd4j.getRandom();
        for (int i = (int)size - 1; i > 0; i--) {
            int j = rng.nextInt(i + 1);
            long temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        return indices;
    }
    
    /**
     * 使用均匀分布进行稀疏初始化
     * @param tensor 要初始化的张量
     * @param sparsity 稀疏度
     * @return 初始化后的张量
     */
    public static INDArray sparseUniformInit(INDArray tensor, double sparsity) {
        return sparseInit(tensor, sparsity, "uniform");
    }
    
    /**
     * 使用正态分布进行稀疏初始化
     * @param tensor 要初始化的张量
     * @param sparsity 稀疏度
     * @return 初始化后的张量
     */
    public static INDArray sparseNormalInit(INDArray tensor, double sparsity) {
        return sparseInit(tensor, sparsity, "normal");
    }
} 