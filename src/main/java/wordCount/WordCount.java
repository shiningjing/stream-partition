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

package wordCount;

import org.apache.flink.api.common.eventtime.WatermarkStrategy;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.streaming.api.datastream.DataStream;
import org.apache.flink.streaming.api.datastream.SingleOutputStreamOperator;
import org.apache.flink.streaming.api.environment.StreamExecutionEnvironment;
import org.apache.flink.streaming.api.TimeCharacteristic;
import org.apache.flink.streaming.api.functions.ProcessFunction;
import org.apache.flink.streaming.api.functions.source.SourceFunction;
import org.apache.flink.streaming.api.windowing.assigners.SlidingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.assigners.TumblingEventTimeWindows;
import org.apache.flink.streaming.api.windowing.time.Time;
import org.apache.flink.util.Collector;
import org.apache.flink.util.OutputTag;
import org.apache.flink.metrics.Counter;
import org.apache.flink.api.common.functions.RichFlatMapFunction;
import org.apache.flink.configuration.Configuration;
import org.apache.flink.configuration.TaskManagerOptions;
import org.apache.flink.metrics.Gauge;
import org.apache.flink.api.common.JobExecutionResult;
import org.apache.flink.api.common.functions.FlatMapFunction;

import javax.annotation.concurrent.GuardedBy;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.lang.management.MemoryMXBean;
import java.lang.management.MemoryUsage;
import java.util.*;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicLong;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.io.File;

import partitioning.*;
import record.Record;
import record.RecordStr;
import sources.*;
import wordCount.containers.*;

import static helperfunctions.PartitionerAssigner.initializePartitioner;

/**
 * <p>
 * Implementation of word count for data streams
 */

public class WordCount {
    // 性能监控类
    public static class PerformanceMonitor {
        private static PerformanceMonitor INSTANCE = null;
        private final String logFilePath;
        private final AtomicLong recordCounter = new AtomicLong(0);
        private final AtomicLong lastTimestamp = new AtomicLong(System.currentTimeMillis());
        private final MemoryMXBean memoryMXBean = ManagementFactory.getMemoryMXBean();
        private final ScheduledExecutorService scheduler = Executors.newScheduledThreadPool(1);
        private BufferedWriter writer;
        private boolean isRunning = false;
        private static String methodName = "unknown";

        private PerformanceMonitor() {
            SimpleDateFormat sdf = new SimpleDateFormat("yyyyMMdd_HHmmss");
            // 确保输出目录存在
            new File("output").mkdirs();
            this.logFilePath = "output/performance_metrics_" + methodName + "_" + sdf.format(new Date()) + ".csv";
            System.out.println("Creating performance log with method name: " + methodName);
            try {
                writer = new BufferedWriter(new FileWriter(logFilePath));
                writer.write("timestamp,heapMemoryUsed,nonHeapMemoryUsed,recordsProcessed,throughput\n");
            } catch (IOException e) {
                System.err.println("Error initializing performance monitor: " + e.getMessage());
            }
        }

        public static void setMethodName(String method) {
            methodName = method;
            System.out.println("Setting performance monitor method name to: " + methodName);
        }

        public static synchronized PerformanceMonitor getInstance() {
            if (INSTANCE == null) {
                INSTANCE = new PerformanceMonitor();
            }
            return INSTANCE;
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

    /**
     * Main method.
     *
     * @throws Exception which occurs during job execution.
     */
    public static void main(String[] args) throws Exception {
        System.out.println("Starting WordCount with args: " + Arrays.toString(args));
        
        // 先设置性能监控的方法名，必须在获取实例之前调用
        String method = "unknown";
        if (args.length > 4 && args[4] != null) {
            method = args[4];
            System.out.println("Method name from args[4]: " + method);
        }
        PerformanceMonitor.setMethodName(method);
        
        // 初始化性能监控
        PerformanceMonitor monitor = PerformanceMonitor.getInstance();
        monitor.start(); // 启动性能监控
        
        try {
            String pathFile = args[0];
            int parallelism = Integer.parseInt(args[2]);

            Time WINDOW_SLIDE = Time.milliseconds(Integer.parseInt(args[5]));
            Time WINDOW_SIZE = Time.milliseconds(Integer.parseInt(args[6]));

            int numOfKeys = Integer.parseInt(args[7]);
            int reducerParallelism = Integer.parseInt(args[8]);

            final OutputTag<WordCountState> outputTag = new OutputTag<WordCountState>("side-output"){};

            // Initialize the partitioner
            Partitioner partitioner1 = initializePartitioner(args[4], parallelism, Integer.parseInt(args[5]), Integer.parseInt(args[6]), numOfKeys);
            
            // 创建 Flink 配置
            Configuration config = new Configuration();
            config.setInteger(TaskManagerOptions.NUM_TASK_SLOTS, 10); // 每个 TaskManager 提供 6 个 Slot

            // 创建带 Web UI 的本地环境
            StreamExecutionEnvironment env = StreamExecutionEnvironment.createLocalEnvironmentWithWebUI(config);
            env.setStreamTimeCharacteristic(TimeCharacteristic.EventTime);
            env.setMaxParallelism(parallelism);
            
            // 添加自定义指标
            env.getConfig().setGlobalJobParameters(new Configuration());

            // Circular source - 使用原始实现，同时指定确切的源函数类型
            @SuppressWarnings({"unchecked", "rawtypes"})
            SingleOutputStreamOperator<Tuple2<Integer, Record>> data = (SingleOutputStreamOperator) env
                    .addSource((org.apache.flink.streaming.api.functions.source.SourceFunction<Record>) new CircularFeed(pathFile), "CircularDataGenerator")
                    .setParallelism(parallelism)
                    .slotSharingGroup("source")
                    .assignTimestampsAndWatermarks(WatermarkStrategy.forMonotonousTimestamps())
                    .slotSharingGroup("source")
                    .flatMap((org.apache.flink.api.common.functions.FlatMapFunction<Record, Tuple2<Integer, Record>>) partitioner1)
                    .setParallelism(parallelism)
                    .slotSharingGroup("source");

            // 添加性能监控
            data = (SingleOutputStreamOperator<Tuple2<Integer, Record>>) data
                    .map(new org.apache.flink.api.common.functions.MapFunction<Tuple2<Integer, Record>, Tuple2<Integer, Record>>() {
                        @Override
                        public Tuple2<Integer, Record> map(Tuple2<Integer, Record> tuple) throws Exception {
                            PerformanceMonitor.getInstance().incrementCounter();
                            return tuple;
                        }
                    })
                    .setParallelism(1)
                    .name("Performance Monitoring");

            // 添加显式的类型抑制警告
            @SuppressWarnings({"unchecked", "rawtypes"})
            SingleOutputStreamOperator<WordCountState> body;
            
            if (args[4].equals("HASHING") || args[4].equals("cAM")) { // Hashing-like techniques
                body = data
                        .keyBy(x -> (x.f0)) // keyBy the key specified by the partitioner
                        .window(SlidingEventTimeWindows.of(WINDOW_SIZE, WINDOW_SLIDE))
                        .aggregate((org.apache.flink.api.common.functions.AggregateFunction<Tuple2<Integer, Record>, Map<Integer, WordCountState>, Map<Integer, WordCountState>>) new MapWordCount()) // word count
                        .setParallelism(parallelism)
                        .setMaxParallelism(parallelism)
                        .process(new SplitPerKeyResult())
                        .setParallelism(parallelism)
                        .setMaxParallelism(parallelism);
            }
            else { // key-splitting techniques
                body = data
                        .keyBy(x -> (x.f0)) // keyBy the key specified by the partitioner
                        .window(SlidingEventTimeWindows.of(WINDOW_SIZE, WINDOW_SLIDE))
                        .aggregate((org.apache.flink.api.common.functions.AggregateFunction<Tuple2<Integer, Record>, Map<Integer, WordCountState>, Map<Integer, WordCountState>>) new MapWordCount()) // 1st STEP
                        .setParallelism(parallelism)
                        .setMaxParallelism(parallelism)
                        .slotSharingGroup("step1")
                        .process(new ForwardKeys(outputTag))
                        .setParallelism(parallelism)
                        .setMaxParallelism(parallelism)
                        .slotSharingGroup("step1");

                DataStream<WordCountState> hotResult = body
                        .keyBy(x -> x.getKey())
                        .window(TumblingEventTimeWindows.of(WINDOW_SLIDE))
                        .aggregate(new ReduceWordCount()) // --------- 2nd STEP OF THE PROCESSING ---------
                        .setParallelism(reducerParallelism)
                        .setMaxParallelism(reducerParallelism)
                        .slotSharingGroup("step2");

                DataStream<WordCountState> nonHotResult = body.getSideOutput(outputTag);
            }

            // Run
            JobExecutionResult result = env.execute("Word Counter");
            
            // 输出执行结果信息
            System.out.println("Job completed successfully");
            System.out.println("Execution time: " + result.getNetRuntime(TimeUnit.SECONDS) + " seconds");
        } finally {
            // 关闭性能监控并保存数据
            monitor.stop();
        }
    }

    /**
     * Key-forwarding
     * Sends non-hot keys to the output directly and hot keys to the reducers
     */
    public static class ForwardKeys extends ProcessFunction<Map<Integer, WordCountState>, WordCountState> {
        OutputTag<WordCountState> outputTag;

        public ForwardKeys(OutputTag<WordCountState> outputTag){
            this.outputTag = outputTag;
        }

        @Override
        public void processElement(Map<Integer, WordCountState> input, Context ctx, Collector<WordCountState> out){
            for (Map.Entry<Integer, WordCountState> entry : input.entrySet()) {
                if(entry.getValue().isHot()) {
                    out.collect(entry.getValue());
                }
                else{
                    ctx.output(outputTag, entry.getValue());
                }
            }
        }
    }

    public static class SplitPerKeyResult extends ProcessFunction<Map<Integer, WordCountState>, WordCountState> {
        @Override
        public void processElement(Map<Integer, WordCountState> input, Context ctx, Collector<WordCountState> out){
            for (Map.Entry<Integer, WordCountState> entry : input.entrySet()){
                out.collect(entry.getValue());
            }
        }
    }
}
