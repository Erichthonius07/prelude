package com.prelude.denoise.api;

import com.prelude.denoise.model.FusedFrame;
import com.prelude.denoise.timeout.FakeInferenceRunner;
import com.prelude.denoise.timeout.FakeTimeSource;
import org.junit.Test;
import static org.junit.Assert.*;

public class InferenceApiTest {

    private final FusedFrame dummyInput = new FusedFrame("burst", "naive", new float[1], "fusion_multi");

    private static class DummyThermalProvider implements ThermalStateProvider {
        private int status;
        DummyThermalProvider(int status) { this.status = status; }
        @Override public int getCurrentThermalStatus() { return status; }
    }

    @Test
    public void calibrateLatencyFractionalMsAndNearestRankP95() {
        FakeTimeSource timeSource = new FakeTimeSource() {
            private int callCount = 0;
            @Override
            public long nanoTime() {
                callCount++;
                int measurementCall = callCount;
                int runIndex = (measurementCall - 1) / 2;
                boolean isStart = (measurementCall % 2 == 1);
                
                if (isStart) return 0;
                return (runIndex + 1) * 1000000L + 500000L;
            }
        };
        
        FakeInferenceRunner runner = new FakeInferenceRunner(new float[1]);
        InferenceApi api = new InferenceApi(runner, timeSource);
        
        double p95 = api.calibrateLatency(10, 5, new DummyThermalProvider(0), dummyInput);
        
        assertEquals(15, runner.callCount.get());
        assertEquals(10.5, p95, 0.0001);
    }

    @Test(expected = IllegalStateException.class)
    public void throwsIfThermallyThrottledBefore() {
        InferenceApi api = new InferenceApi(new FakeInferenceRunner(), new FakeTimeSource());
        api.calibrateLatency(10, 5, new DummyThermalProvider(2), dummyInput);
    }
}
