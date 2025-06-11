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

package sources;

import org.apache.flink.configuration.Configuration;
import org.apache.flink.streaming.api.functions.source.RichParallelSourceFunction;
import org.apache.flink.core.fs.Path;
import org.apache.flink.core.fs.FileSystem;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.*;

import record.*;

/**
 * Source circular function that continuously feeds the pipeline with data
 * <p>
 */
// https://ci.apache.org/projects/flink/flink-docs-master/api/java/org/apache/flink/streaming/api/functions/source/SourceFunction.html
public class CircularFeed extends RichParallelSourceFunction<Record> {
    private String[] attrs;
    private volatile List<RecordStr> data;
    private volatile boolean cancelled;
    private volatile String filePath;
    private volatile long timestamp;

    public CircularFeed(String argPath) {
        filePath = argPath;
        data = new ArrayList<>();
        cancelled = false;
        timestamp = 0L;
    }

    @Override
    // Called once during initialization.
    public void open(Configuration conf) throws Exception {
        long id = 0L;
        String record;
        final int MAX_RECORDS = 100000; // 最大数据条数限制

        try {
            // Use Flink's FileSystem API to read the file
            Path path = new Path(filePath);
            FileSystem fs = path.getFileSystem();

            BufferedReader myReader = null;

            try {
                myReader = new BufferedReader(new InputStreamReader(fs.open(path)));

                // 跳过第一行标题
                String headerLine = myReader.readLine();

                while ((record = myReader.readLine()) != null && id < MAX_RECORDS) {
                    attrs = record.split(",");
                    try {
                        data.add(new RecordStr(Integer.parseInt(attrs[0]), attrs[1], id));
                    } catch (NumberFormatException e) {
                        e.printStackTrace();
                        continue;
                    }
                    id++;
                }
            } finally {
                if (myReader != null) {
                    try {
                        myReader.close();
                    } catch (Exception e) {
                        System.err.println("ERROR: Failed to close file reader: " + e.getMessage());
                    }
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void sleep(int i){
        final long INTERVAL = i;
        long start = System.nanoTime();
        long end = 0;
        do {
            end  = System.nanoTime();
        }while (start + INTERVAL >= end);
    }

    @Override
    public void run(SourceContext<Record> ctx) throws Exception {
        int cycleCount = 0;

        for (int j = 0; j < data.size() && !cancelled; ) {
            RecordStr record = data.get(j);
            record.setTs(timestamp);
            //sleep(120000);
            ctx.collectWithTimestamp(record, timestamp);
            timestamp += 1;

            // 移动到下一条记录，实现循环播放
            j++;
            if (j >= data.size()) {
                j = 0;  // 重新开始循环
                cycleCount++;
            }
        }
    }

    @Override
    public void cancel() {
        cancelled = true;
    }

    @Override
    public void close() {
        cancelled = true;
    }
}



