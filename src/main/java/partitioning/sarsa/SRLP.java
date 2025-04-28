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

package partitioning.sarsa;

import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;
import org.apache.flink.util.Collector;
import streamSARSA.StreamSARSA;
import partitioning.Partitioner;
import record.Record;

/**
 * Class implementing the SRLP (SARSA Reinforcement Learning Partitioner) for stream processing
 * Uses deep reinforcement learning with SARSA(λ) for adaptive partitioning
 */
public class SRLP extends Partitioner implements CheckpointedFunction {
    
    // 核心SARSA算法组件
    private SARSA sarsa;
    
    // 状态管理
    private ListState<SARSA> sarsa_chk;

    /**
     * 构造函数
     * 
     * @param numWorkers 工作节点数量
     * @param slide 滑动窗口滑动步长
     * @param size 滑动窗口大小
     * @param numOfKeys 预估键数量
     */
    public SRLP(int numWorkers, int slide, int size, int numOfKeys) {
        super(numWorkers);
        sarsa = new SARSA(numWorkers, slide, size, numOfKeys);
    }

    /**
     * 主要处理函数
     * 接收一条记录并决定将其发送到哪个工作节点
     * 
     * @param r 新到达的记录
     * @param out 输出<Worker, Record>
     */
    @Override
    public void flatMap(Record r, Collector<Tuple2<Integer, Record>> out) throws Exception {
        boolean isHot;
        int worker;

        // 检测热点键并进行热点键管理
        isHot = sarsa.isHot(r);
        //isHot=false;
        // 更新状态（每个滑动窗口执行一次）
        sarsa.expireState(r, isHot);
        long startTime = System.nanoTime();;
        // 确定分区（工作节点）
        worker = sarsa.partition(r, isHot);
        r.setHot(true);
        long endTime = System.nanoTime(); // 结束时间
        System.out.println("确定分区运行时间：" + (endTime - startTime) + " 纳秒");
        // 输出到指定工作节点
        out.collect(new Tuple2<>(worker, r));
        long startTime1 = System.nanoTime();
        // 更新状态
        sarsa.updateState(r, worker);
        long endTime1 = System.nanoTime(); // 结束时间
        System.out.println("更新状态运行时间：" + (endTime1 - startTime1) + " 纳秒");
        // 只有热键才更新学习模型
        if (isHot) {
            long startTime2 = System.nanoTime();;
            sarsa.updateQtable(r, isHot, worker);
            long endTime2 = System.nanoTime(); // 结束时间
            System.out.println("更新参数运行时间：" + (endTime2 - startTime2) + " 纳秒");
        }
    }

    /**
     * 检查点快照，保存状态
     */
    @Override
    public void snapshotState(FunctionSnapshotContext functionSnapshotContext) throws Exception {
        // 确保在序列化前保存网络权重
        if (sarsa != null) {
            // 获取StreamSARSA实例
            StreamSARSA model = sarsa.getSarsa();
            if (model != null) {
                model.saveWeights();
            }
        }
        
        sarsa_chk.clear();
        sarsa_chk.add(sarsa);
    }

    /**
     * 初始化状态，从检查点恢复
     */
    @Override
    public void initializeState(FunctionInitializationContext functionInitializationContext) throws Exception {
        sarsa_chk = functionInitializationContext.getOperatorStateStore().getListState(
                new ListStateDescriptor<>("sarsaChk", SARSA.class));
                
        // 从检查点恢复状态
        for (SARSA s : sarsa_chk.get()) {
            sarsa = s;
        }
    }

    /**
     * 获取SARSA算法实例（用于调试）
     * 
     * @return SARSA算法实例
     */
    public SARSA getSarsa() {
        return sarsa;
    }
} 