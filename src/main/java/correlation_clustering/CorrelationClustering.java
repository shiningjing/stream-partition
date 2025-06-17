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

package correlation_clustering;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.TaskManagerOptions;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.windowing.assigners.SlidingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.IllegalArgumentException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;

import record.Record;
import correlation_clustering.containers.PartialClusteringOutput;
import partitioning.*;
import sources.*;
import static helperfunctions.PartitionerAssigner.initializePartitioner;

enum CORR_CLUSTERING {
    BEST,
    VOTE
}

/**
 * Correlation Clustering for Streaming data
 * <p>
 */

public class CorrelationClustering {
    // 性能监控类
    public static class PerformanceMonitor {
        private static final PerformanceMonitor INSTANCE = new PerformanceMonitor();
        private String logFilePath;
        private final AtomicLong recordCounter = new AtomicLong(0);
        private final AtomicLong lastTimestamp = new AtomicLong(System.currentTimeMillis());
        private final MemoryMXBean memoryMXBean = ManagementFactory.getMemoryMXBean();
        private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
        private BufferedWriter writer;
        private boolean isRunning = false;

        private PerformanceMonitor() {
            // 初始化时不设置文件路径，等待方法名称
        }

        public static PerformanceMonitor getInstance() {
            return INSTANCE;
        }

        // 设置方法名称并初始化文件
        public void setMethod(String methodName) {
            this.logFilePath = "output/correlation_clustering_metrics_" + methodName + "_" + System.currentTimeMillis() + ".csv";
            try {
                writer = new BufferedWriter(new FileWriter(logFilePath));
                writer.write("timestamp,heapMemoryUsed,nonHeapMemoryUsed,recordsProcessed,throughput\n");
            } catch (IOException e) {
                System.err.println("Error initializing performance monitor: " + e.getMessage());
            }
        }

        public void start() {
            if (!isRunning) {
                isRunning = true;
                scheduler.scheduleAtFixedRate(this::recordMetrics, 0, 1, TimeUnit.SECONDS);
                System.out.println("Performance monitoring started. Metrics will be saved to: " + logFilePath);
            }
        }

        public void stop() {
            isRunning = false;
            scheduler.shutdown();
            try {
                if (writer != null) {
                    writer.close();
                }
                System.out.println("Performance monitoring stopped. Metrics saved to: " + logFilePath);
            } catch (IOException e) {
                System.err.println("Error closing performance monitor: " + e.getMessage());
            }
        }

        public void incrementCounter() {
            recordCounter.incrementAndGet();
        }

        private void recordMetrics() {
            try {
                long currentTime = System.currentTimeMillis();
                long timeElapsed = currentTime - lastTimestamp.get();
                long recordsProcessed = recordCounter.getAndSet(0);
                double throughput = recordsProcessed * 1000.0 / timeElapsed; // records per second
                
                MemoryUsage heapMemory = memoryMXBean.getHeapMemoryUsage();
                MemoryUsage nonHeapMemory = memoryMXBean.getNonHeapMemoryUsage();
                
                writer.write(String.format("%d,%d,%d,%d,%.2f\n", 
                    currentTime, 
                    heapMemory.getUsed(), 
                    nonHeapMemory.getUsed(),
                    recordsProcessed,
                    throughput));
                writer.flush();
                
                lastTimestamp.set(currentTime);
                
                System.out.printf("Memory: Heap=%dMB, NonHeap=%dMB, Records=%d, Throughput=%.2f records/sec\n",
                    heapMemory.getUsed() / (1024 * 1024),
                    nonHeapMemory.getUsed() / (1024 * 1024),
                    recordsProcessed,
                    throughput);
            } catch (IOException e) {
                System.err.println("Error recording metrics: " + e.getMessage());
            }
        }
    }

    // 带监控的处理函数
    public static class MonitoredProcessFunction extends ProcessFunction<Tuple2<Integer, Record>, Tuple2<Integer, Record>> {
        @Override
        public void processElement(Tuple2<Integer, Record> value, ProcessFunction<Tuple2<Integer, Record>, Tuple2<Integer, Record>>.Context ctx, Collector<Tuple2<Integer, Record>> out) {
            PerformanceMonitor.getInstance().incrementCounter();
            out.collect(value);
        }
    }

    /**
     * Main method.
     *
     * @throws Exception which occurs during job execution.
     */
    public static void main(String[] args) throws Exception {
        try {
            final String pathFile = args[0];
            final int combinersParallelism = Integer.parseInt(args[2]);

            final Time WINDOW_SLIDE = Time.milliseconds(Integer.parseInt(args[5]));
            final Time WINDOW_SIZE = Time.milliseconds(Integer.parseInt(args[6]));

            final int estimatedNumOfKeys = Integer.parseInt(args[7]);
            final int reducersParallelism = Integer.parseInt(args[8]);

            CORR_CLUSTERING clustering_alg;
            if (args[3].equals("VOTE"))
                clustering_alg = CORR_CLUSTERING.VOTE;
            else if (args[3].equals("BEST"))
                clustering_alg = CORR_CLUSTERING.BEST;
            else{
                throw new IllegalArgumentException("Unknown Correlation Clustering Algorithm " + args[3]);
            }

            final OutputTag<PartialClusteringOutput> outputTag = new OutputTag<PartialClusteringOutput>("side-output"){};
            Partitioner partitioner = initializePartitioner(args[4], combinersParallelism, Integer.parseInt(args[5]), Integer.parseInt(args[6]), estimatedNumOfKeys);
            
            // 设置性能监控器的方法名称并启动
            PerformanceMonitor.getInstance().setMethod(args[4]);
            PerformanceMonitor.getInstance().start();
            
            Configuration config = new Configuration();
            config.setInteger(TaskManagerOptions.NUM_TASK_SLOTS, 10); // 每个 TaskManager 提供 6 个 Slot

            // 创建环境
            StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment(config);

            // Configure the environment
            //StreamExecutionEnvironment env = StreamExecutionEnvironment.getExecutionEnvironment();
            env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
            env.setMaxParallelism(combinersParallelism);

            // Circular source
            SingleOutputStreamOperator<Tuple2<Integer, Record>> data = env.addSource(new CircularFeed(pathFile), "CircularDataGenerator")
                    .setParallelism(1)
                    .slotSharingGroup("source")
                    .assignTimestampsAndWatermarks(WatermarkStrategy.forMonotonousTimestamps())
                    .setParallelism(1)
                    .slotSharingGroup("source")
                    .flatMap(partitioner)
                    .setParallelism(1)
                    .slotSharingGroup("source")
                    .process(new MonitoredProcessFunction()) // 添加监控处理函数
                    .setParallelism(1)
                    .slotSharingGroup("source");

            if (args[4].equals("HASHING") || args[4].equals("cAM")){ // Hashing-like techniques
                SingleOutputStreamOperator<List<PartialClusteringOutput>> body = data
                        .keyBy(x -> (x.f0)) // keyBy the key specified by the partitioner
                        .window(SlidingEventTimeWindows.of(WINDOW_SIZE, WINDOW_SLIDE))
                        .aggregate(new PartialWindowClustering(clustering_alg)) // correlation clustering in one step
                        .setParallelism(combinersParallelism)
                        .setMaxParallelism(combinersParallelism)
                        .slotSharingGroup("step1");
            }
            else { // key splitting techniques
                SingleOutputStreamOperator<PartialClusteringOutput> body = data
                        .keyBy(x -> (x.f0)) // keyBy the key specified by the partitioner
                        .window(SlidingEventTimeWindows.of(WINDOW_SIZE, WINDOW_SLIDE))
                        .aggregate(new PartialWindowClustering(clustering_alg)) // --------- 1st STEP OF THE PROCESSING: Find most similar cluster per block ---------
                        .setParallelism(combinersParallelism)
                        .setMaxParallelism(combinersParallelism)
                        .slotSharingGroup("step1")
                        .process(new ForwardKeys(outputTag)) // forward non-hot keys to the output and hot keys to the reducers
                        .setParallelism(combinersParallelism)
                        .setMaxParallelism(combinersParallelism)
                        .slotSharingGroup("step1");

                DataStream<List<List<String>>> hotResult = body
                        .keyBy(x -> x.blockId)
                        .window(TumblingEventTimeWindows.of(WINDOW_SLIDE))
                        .aggregate(new IncrementalMerge()) // --------- 2nd STEP OF THE PROCESSING: Merge  ---------
                        .setParallelism(reducersParallelism)
                        .setMaxParallelism(reducersParallelism)
                        .slotSharingGroup("step2");
            }

            env.execute("Flink Stream Java API Skeleton");
        } finally {
            // 确保在程序退出时停止监控
            PerformanceMonitor.getInstance().stop();
        }
    }

    public static class ForwardKeys extends ProcessFunction<List<PartialClusteringOutput>, PartialClusteringOutput> {
        OutputTag<PartialClusteringOutput> outputTag;

        public ForwardKeys(OutputTag<PartialClusteringOutput> outputTag){
            this.outputTag = outputTag;
        }

        @Override
        public void processElement(List<PartialClusteringOutput> input, Context ctx, Collector<PartialClusteringOutput> out) {
            for (PartialClusteringOutput block : input) {
                if (block.isHot) {
                    out.collect(block);
                }
                else{
                    ctx.output(outputTag, block);
                }
            }
        }
    }
}
