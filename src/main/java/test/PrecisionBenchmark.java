package test;

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
import org.nd4j.linalg.api.buffer.DataType;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.locks.ReentrantReadWriteLock;
import java.util.stream.Collectors;

/**
 * DRL神经网络精度性能基准测试
 * 测试FP16、FP32、FP64三种精度下的推理和训练性能
 * 使用与DRLPartitioner相同的网络架构和参数
 * 兼容Java 8+
 */
public class PrecisionBenchmark {

    // 网络参数 - 与DRLPartitioner保持一致
    private static final int NUM_WORKERS = 128;          // 并行度
    private static final int STATE_SIZE = NUM_WORKERS + NUM_WORKERS; // 状态向量大小：负载 + 分片
    private static final int ACTION_SIZE = NUM_WORKERS; // 动作空间大小
    private static final int HIDDEN_LAYER_SIZE = (STATE_SIZE + ACTION_SIZE); // 隐藏层大小
    private static final ReentrantReadWriteLock networkLock = new ReentrantReadWriteLock();
    // 测试参数
    private static final int WARMUP_ITERATIONS = 1000;   // 预热迭代次数
    private static final int TEST_ITERATIONS = 1000;    // 测试迭代次数
    private static final int BATCH_SIZE = 128;          // 批量大小

    // 精度测试枚举
    enum PrecisionType {
        FP16("FP16", DataType.HALF, "半精度 - 最快速度，较低精度"),
        FP32("FP32", DataType.FLOAT, "单精度 - 平衡性能和精度"),
        FP64("FP64", DataType.DOUBLE, "双精度 - 最高精度，较慢速度");

        final String name;
        final DataType dataType;
        final String description;

        PrecisionType(String name, DataType dataType, String description) {
            this.name = name;
            this.dataType = dataType;
            this.description = description;
        }
    }

    // 测试结果类
    static class BenchmarkResult {
        final PrecisionType precision;
        final double avgInferenceTime;     // 平均推理时间（微秒）
        final double avgTrainingTime;      // 平均训练时间（微秒）
        final double throughputInference;  // 推理吞吐量（次/秒）
        final double throughputTraining;   // 训练吞吐量（次/秒）
        final long memoryUsage;           // 内存使用量（字节）
        final boolean setupSuccess;       // 设置是否成功
        final String errorMessage;        // 错误信息

        BenchmarkResult(PrecisionType precision, double avgInferenceTime, double avgTrainingTime,
                        double throughputInference, double throughputTraining, long memoryUsage,
                        boolean setupSuccess, String errorMessage) {
            this.precision = precision;
            this.avgInferenceTime = avgInferenceTime;
            this.avgTrainingTime = avgTrainingTime;
            this.throughputInference = throughputInference;
            this.throughputTraining = throughputTraining;
            this.memoryUsage = memoryUsage;
            this.setupSuccess = setupSuccess;
            this.errorMessage = errorMessage;
        }
    }

    public static void main(String[] args) {
        System.out.println("==================================================");
        System.out.println("🧠 DRL神经网络精度性能基准测试");
        System.out.println("==================================================");
        System.out.println("网络架构: " + STATE_SIZE + " → " + HIDDEN_LAYER_SIZE + " → " + ACTION_SIZE);
        System.out.println("预热迭代: " + WARMUP_ITERATIONS + " 次");
        System.out.println("测试迭代: " + TEST_ITERATIONS + " 次");
        System.out.println("批量大小: " + BATCH_SIZE);
        System.out.println("Java版本: " + System.getProperty("java.version"));
        System.out.println("==================================================");

        List<BenchmarkResult> results = new ArrayList<>();

        // 测试三种精度
        for (PrecisionType precision : PrecisionType.values()) {
            System.out.println("\n🔬 测试 " + precision.name + " (" + precision.description + ")");
            System.out.println("--------------------------------------------------");

            try {
                BenchmarkResult result = benchmarkPrecision(precision);
                results.add(result);

                if (result.setupSuccess) {
                    System.out.printf("✅ %s 测试完成\n", precision.name);
                    System.out.printf("   推理时间: %.2f微秒 | 吞吐量: %.0f次/秒\n",
                            result.avgInferenceTime, result.throughputInference);
                    System.out.printf("   训练时间: %.2f微秒 | 吞吐量: %.0f次/秒\n",
                            result.avgTrainingTime, result.throughputTraining);
                    System.out.printf("   内存使用: %.2f MB\n", result.memoryUsage / 1024.0 / 1024.0);
                } else {
                    System.out.printf("❌ %s 测试失败: %s\n", precision.name, result.errorMessage);
                }

            } catch (Exception e) {
                System.out.printf("💥 %s 测试异常: %s\n", precision.name, e.getMessage());
                results.add(new BenchmarkResult(precision, 0, 0, 0, 0, 0, false, e.getMessage()));
            }
        }

        // 生成综合报告
        generateComprehensiveReport(results);
    }

    /**
     * 测试指定精度的性能
     */
    private static BenchmarkResult benchmarkPrecision(PrecisionType precision) {
        long startMemory = getUsedMemory();

        try {
            // 1. 创建网络
            System.out.println("📊 创建神经网络...");
            MultiLayerNetwork network = createNetwork(precision.dataType);

            // 2. 预热
            System.out.println("🔥 预热阶段...");
            warmup(network, precision.dataType);

            // 3. 推理性能测试
            System.out.println("⚡ 推理性能测试...");
            double avgInferenceTime = benchmarkInference(network, precision.dataType);
            double throughputInference = 1_000_000.0 / avgInferenceTime; // 转换为次/秒

            // 4. 训练性能测试
            System.out.println("🏋️ 训练性能测试...");
            double avgTrainingTime = benchmarkTraining(network, precision.dataType);
            double throughputTraining = 1_000_000.0 / avgTrainingTime; // 转换为次/秒

            long endMemory = getUsedMemory();
            long memoryUsage = endMemory - startMemory;

            return new BenchmarkResult(precision, avgInferenceTime, avgTrainingTime,
                    throughputInference, throughputTraining, memoryUsage,
                    true, null);

        } catch (Exception e) {
            return new BenchmarkResult(precision, 0, 0, 0, 0, 0, false, e.getMessage());
        }
    }

    /**
     * 创建与DRLPartitioner相同架构的神经网络
     */
    private static MultiLayerNetwork createNetwork(DataType dataType) {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(123)
                .weightInit(WeightInit.XAVIER)
                .updater(new Adam(0.0001))
                .dataType(dataType)  // 设置网络精度
                .list()
                .layer(0, new DenseLayer.Builder()
                        .nIn(STATE_SIZE)
                        .nOut(HIDDEN_LAYER_SIZE)
                        .activation(Activation.RELU)
                        .build())
                .layer(1, new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nIn(HIDDEN_LAYER_SIZE)
                        .nOut(ACTION_SIZE)
                        .activation(Activation.IDENTITY)
                        .build())
                .build();

        MultiLayerNetwork network = new MultiLayerNetwork(conf);
        network.init();

        System.out.println("   网络权重数据类型: " + network.params().dataType());
        System.out.println("   网络参数数量: " + network.numParams());

        return network;
    }

    /**
     * 预热阶段
     */
    private static void warmup(MultiLayerNetwork network, DataType dataType) {
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            // 推理预热
            INDArray input = createRandomInput(1, dataType);
            network.output(input);

            // 训练预热
            if (i % 10 == 0) {
                INDArray batchInput = createRandomInput(BATCH_SIZE, dataType);
                INDArray batchTarget = createRandomTarget(BATCH_SIZE, dataType);
                network.fit(batchInput, batchTarget);
            }
        }
        System.out.println("   预热完成");
    }

    /**
     * 推理性能测试
     */
    private static double benchmarkInference(MultiLayerNetwork network, DataType dataType) {
        List<Double> times = new ArrayList<>();

        for (int i = 0; i < TEST_ITERATIONS; i++) {


            long startTime = System.nanoTime();
            try {
                INDArray input = createRandomInput(1, dataType);
            networkLock.readLock().lock();
            INDArray output = network.output(input);
               int worker = Nd4j.argMax(output, 1).getInt(0);

                if (i == 0) {
                    System.out.println("   输出数据类型: " + output.dataType());
                }
            } finally {
                networkLock.readLock().unlock();
                long endTime = System.nanoTime();

                double timeInMicros = (endTime - startTime) / 1000.0;
                times.add(timeInMicros);
            }
            // 确保输出被使用，防止编译器优化

        }

        // 计算统计信息
        double avgTime = times.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double minTime = times.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double maxTime = times.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);

        System.out.printf("   推理时间 - 平均: %.2f微秒, 最小: %.2f微秒, 最大: %.2f微秒\n",
                avgTime, minTime, maxTime);

        return avgTime;
    }

    /**
     * 训练性能测试
     */
    private static double benchmarkTraining(MultiLayerNetwork network, DataType dataType) {
        List<Double> times = new ArrayList<>();
        int trainIterations = TEST_ITERATIONS / 10; // 训练次数较少，因为训练较慢

        for (int i = 0; i < trainIterations; i++) {
            INDArray batchInput = createRandomInput(BATCH_SIZE, dataType);
            INDArray batchTarget = createRandomTarget(BATCH_SIZE, dataType);

            long startTime = System.nanoTime();
            network.fit(batchInput, batchTarget);
            long endTime = System.nanoTime();

            double timeInMicros = (endTime - startTime) / 1000.0;
            times.add(timeInMicros);
        }

        // 计算统计信息
        double avgTime = times.stream().mapToDouble(Double::doubleValue).average().orElse(0.0);
        double minTime = times.stream().mapToDouble(Double::doubleValue).min().orElse(0.0);
        double maxTime = times.stream().mapToDouble(Double::doubleValue).max().orElse(0.0);

        System.out.printf("   训练时间 - 平均: %.2f微秒, 最小: %.2f微秒, 最大: %.2f微秒\n",
                avgTime, minTime, maxTime);

        return avgTime;
    }

    /**
     * 创建随机输入数据
     */
    private static INDArray createRandomInput(int batchSize, DataType dataType) {
        INDArray input = Nd4j.rand(batchSize, STATE_SIZE);
        return input.castTo(dataType);
    }

    /**
     * 创建随机目标数据
     */
    private static INDArray createRandomTarget(int batchSize, DataType dataType) {
        INDArray target = Nd4j.rand(batchSize, ACTION_SIZE);
        return target.castTo(dataType);
    }

    /**
     * 获取当前内存使用量
     */
    private static long getUsedMemory() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }

    /**
     * 生成综合性能报告
     */
    private static void generateComprehensiveReport(List<BenchmarkResult> results) {
        System.out.println("\n");
        System.out.println("==================================================");
        System.out.println("📈 综合性能报告");
        System.out.println("==================================================");

        // 成功的测试结果 - 使用Java 8兼容语法
        List<BenchmarkResult> successResults = results.stream()
                .filter(r -> r.setupSuccess)
                .collect(Collectors.toList());

        if (successResults.isEmpty()) {
            System.out.println("❌ 所有精度测试都失败了！");
            return;
        }

        // 详细结果表格
        System.out.println("📋 详细结果:");
        System.out.println("┌─────────┬─────────────┬─────────────┬─────────────┬─────────────┬─────────────┐");
        System.out.println("│ 精度    │ 推理时间(μs)│ 训练时间(μs)│ 推理吞吐量  │ 训练吞吐量  │ 内存使用(MB)│");
        System.out.println("├─────────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┤");

        for (BenchmarkResult result : successResults) {
            System.out.printf("│ %-7s │ %11.2f │ %11.2f │ %11.0f │ %11.0f │ %11.2f │\n",
                    result.precision.name,
                    result.avgInferenceTime,
                    result.avgTrainingTime,
                    result.throughputInference,
                    result.throughputTraining,
                    result.memoryUsage / 1024.0 / 1024.0);
        }
        System.out.println("└─────────┴─────────────┴─────────────┴─────────────┴─────────────┴─────────────┘");

        // 性能比较分析
        if (successResults.size() > 1) {
            System.out.println("\n⚡ 性能比较分析:");

            BenchmarkResult baseline = successResults.get(successResults.size() - 1); // 最后一个（通常是FP64）

            for (BenchmarkResult result : successResults) {
                if (result != baseline) {
                    double inferenceSpeedup = baseline.avgInferenceTime / result.avgInferenceTime;
                    double trainingSpeedup = baseline.avgTrainingTime / result.avgTrainingTime;
                    double memoryReduction = (baseline.memoryUsage - result.memoryUsage) / (double)baseline.memoryUsage * 100;

                    System.out.printf("• %s vs %s:\n", result.precision.name, baseline.precision.name);
                    System.out.printf("  - 推理速度提升: %.2fx (%+.1f%%)\n", inferenceSpeedup, (inferenceSpeedup - 1) * 100);
                    System.out.printf("  - 训练速度提升: %.2fx (%+.1f%%)\n", trainingSpeedup, (trainingSpeedup - 1) * 100);
                    System.out.printf("  - 内存节省: %+.1f%%\n", memoryReduction);
                }
            }
        }

        // 推荐建议
        System.out.println("\n💡 推荐建议:");
        BenchmarkResult best = null;
        double bestTime = Double.MAX_VALUE;

        // 找到推理时间最短的结果
        for (BenchmarkResult result : successResults) {
            if (result.avgInferenceTime < bestTime) {
                bestTime = result.avgInferenceTime;
                best = result;
            }
        }

        if (best != null) {
            System.out.println("• 最快推理速度: " + best.precision.name + " (" + best.precision.description + ")");
            System.out.println("• 对于DQN实时推理，推荐使用 " + best.precision.name + " 以获得最佳性能");

            if (best.precision == PrecisionType.FP32) {
                System.out.println("• FP32是性能和精度的最佳平衡点，适合大多数DQN应用");
            } else if (best.precision == PrecisionType.FP16) {
                System.out.println("• FP16提供最快速度，但要注意数值稳定性");
            }
        }

        // 失败的测试 - 使用Java 8兼容语法
        List<BenchmarkResult> failedResults = results.stream()
                .filter(r -> !r.setupSuccess)
                .collect(Collectors.toList());

        if (!failedResults.isEmpty()) {
            System.out.println("\n❌ 失败的测试:");
            for (BenchmarkResult result : failedResults) {
                System.out.println("• " + result.precision.name + ": " + result.errorMessage);
            }
        }

        System.out.println("\n🎯 测试完成！可以根据以上结果选择合适的精度配置。");
        System.out.println("==================================================");
    }
} 