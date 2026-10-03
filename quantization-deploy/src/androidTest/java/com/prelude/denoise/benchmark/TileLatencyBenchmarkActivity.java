package com.prelude.denoise.benchmark;

import android.app.Activity;
import android.os.Bundle;
import android.view.WindowManager;
import android.widget.TextView;

/**
 * Benchmark host activity: keeps the screen on and makes this process the top
 * app while the latency benchmark runs, so it is not treated as a background
 * app by OEM process management (freezer / cpuset).
 */
public class TileLatencyBenchmarkActivity extends Activity {
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        TextView view = new TextView(this);
        view.setText("denoise benchmark running — keep this screen on");
        setContentView(view);
    }
}
