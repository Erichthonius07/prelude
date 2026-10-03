package com.prelude.denoise.tiling;

import com.prelude.denoise.model.LiteRtAdapter;
import org.junit.Test;
import java.nio.ByteBuffer;
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertSame;

/**
 * Task 2: proves the per-tile allocation removal in TileInferenceAdapter.
 *
 * Reuse proof: consecutive run() calls must write into the SAME array instance
 * (no per-tile heap allocation). Correctness proof: the reused array must be
 * fully rewritten on every call — the second call's values must match the
 * second tile exactly, with no stale data from the first tile.
 */
public class TileBufferReuseTest {

    /** Echo adapter: copies the input NCHW buffer to the output (identity model). */
    private static LiteRtAdapter echoAdapter() {
        return new LiteRtAdapter() {
            @Override
            public void run(ByteBuffer in, ByteBuffer out) {
                in.rewind();
                out.clear();
                out.put(in);
            }
            @Override
            public void close() {
            }
        };
    }

    private static float[] tileWithBase(float base) {
        // tileSize=4, channels=1 -> 16 floats; every element differs between the two tiles
        float[] tile = new float[16];
        for (int i = 0; i < tile.length; i++) {
            tile[i] = base + i;
        }
        return tile;
    }

    @Test
    public void tileAdapterReusesTheSameArrayAcrossTiles() {
        TileInferenceAdapter adapter = new TileInferenceAdapter(echoAdapter(), 4, 1);

        float[] tileA = tileWithBase(0f);
        float[] tileB = tileWithBase(100f);

        float[] outA = adapter.run(tileA);
        float[] outB = adapter.run(tileB);

        assertSame("Second tile must be produced into the same array instance "
            + "(no per-tile allocation)", outA, outB);

        // The reused array was fully rewritten: it now holds tile B's values, not tile A's.
        // (Checking outA against tileA here would be invalid usage — the Tiler contract
        // is that the returned tile is consumed before the next run() call.)
        assertArrayEquals("Reused array must contain the second tile's values (no stale data)",
            tileB, outB, 1e-6f);
    }

    @Test
    public void tileAdapterRoundTripIsExactForAnArbitraryTile() {
        TileInferenceAdapter adapter = new TileInferenceAdapter(echoAdapter(), 4, 1);

        float[] tile = tileWithBase(-7.5f);
        float[] out = adapter.run(tile);

        assertArrayEquals(tile, out, 1e-6f);
    }
}
