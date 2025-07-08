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

package partitioning;

import org.apache.flink.api.java.tuple.Tuple2;
import org.apache.flink.util.Collector;
import org.apache.flink.api.common.state.ListState;
import org.apache.flink.api.common.state.ListStateDescriptor;
import org.apache.flink.runtime.state.FunctionInitializationContext;
import org.apache.flink.runtime.state.FunctionSnapshotContext;
import org.apache.flink.streaming.api.checkpoint.CheckpointedFunction;

import java.lang.Math;
import java.util.*;

import record.*;
import partitioning.dalton.state.State;

/**
 * 2 choices Partitioning with State Monitoring
 * <p>
 * Nasir et al.
 * The power of both choices: Practical load balancing for distributed stream processing engines (ICDE'15)
 * 
 * Enhanced with State class for performance monitoring similar to Dalton implementation
 */

public class TwoChoices extends Partitioner implements CheckpointedFunction {
    private final double HASH_C = (Math.sqrt(5) - 1) / 2;
    
    State state;
    private ListState<State> state_chk;

    public TwoChoices(int parallelism, int slide, int size, int numOfKeys) {
        super(parallelism);
        state = new State(size, slide, parallelism, numOfKeys);
    }

    protected int hash1(int n) {
        return n % parallelism;
    }

    // https://www.geeksforgeeks.org/what-are-hash-functions-and-how-to-choose-a-good-hash-function/
    protected int hash2(int n) {
        double a = (n + 1) * HASH_C;
        return (int)Math.floor(parallelism * (a - (int) a));
    }

    @Override
    public void flatMap(Record record, Collector<Tuple2<Integer, Record>> out) throws Exception {
        int recordId = record.getKeyId();
        int worker1 = hash1(recordId);
        int worker2 = hash2(recordId);

        // Update state statistics - expire old state first
        // 设置为true以确保键分配被正确跟踪，用于复制因子计算
        state.updateExpired(record, true); 

        // Choose worker with lower load using State class
        double load1 = state.getLoad(worker1);
        double load2 = state.getLoad(worker2);
        int chosenWorker = (load1 < load2) ? worker1 : worker2;
        
        // Update state with record assignment
        state.update(record, chosenWorker);
        
        out.collect(new Tuple2<>(chosenWorker, record));
    }

    @Override
    public void snapshotState(FunctionSnapshotContext functionSnapshotContext) throws Exception {
        state_chk.clear();
        state_chk.add(state);
    }

    @Override
    public void initializeState(FunctionInitializationContext functionInitializationContext) throws Exception {
        state_chk = functionInitializationContext.getOperatorStateStore()
                .getListState(new ListStateDescriptor<>("twoChoicesStateChk", State.class));
        
        for (State s : state_chk.get()) {
            state = s;
        }
    }

    // Method for debugging - get current state statistics
    public State getState() {
        return state;
    }
}

