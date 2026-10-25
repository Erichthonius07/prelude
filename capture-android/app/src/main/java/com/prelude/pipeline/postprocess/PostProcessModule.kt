package com.prelude.pipeline.postprocess

import android.graphics.Bitmap
import android.os.SystemClock
import com.prelude.pipeline.contract.DenoisedFrame
import com.prelude.pipeline.contract.FinalImage
import com.prelude.pipeline.contract.PostProcessParams
import kotlin.math.pow

/**
 * Role 3: tone mapping (gamma lift + global contrast) and unsharp-mask sharpening.
 *
 * [params] must come from the SSIM grid/random search on the fixed test set, not be tuned by eye. The defaults
 * are placeholders; the app replaces them with `postProcessParams` from the Results Service comparison report
 * when one is available.
 */
class PostProcessModule(@Volatile var params: PostProcessParams = PostProcessParams()) {

    fun process(input: DenoisedFrame): FinalImage {
        val start = SystemClock.elapsedRealtime()
        val p = params
        val src = input.bitmap
        val w = src.width; val h = src.height
        val px = IntArray(w * h).also { src.getPixels(it, 0, w, 0, 0, w, h) }

        // Tone curve as a 256-entry lookup table.
        val lut = IntArray(256) { v ->
            val toned = (v / 255.0).pow(p.toneGamma.toDouble())
            (((toned - 0.5) * p.contrast + 0.5) * 255).toInt().coerceIn(0, 255)
        }
        val r = IntArray(w * h); val g = IntArray(w * h); val b = IntArray(w * h)
        for (i in px.indices) {
            val c = px[i]
            r[i] = lut[(c shr 16) and 0xFF]; g[i] = lut[(c shr 8) and 0xFF]; b[i] = lut[c and 0xFF]
        }

        if (p.sharpenAmount > 0f && p.sharpenRadius > 0) {
            sharpen(r, w, h, p); sharpen(g, w, h, p); sharpen(b, w, h, p)
        }
        for (i in px.indices) px[i] = (0xFF shl 24) or (r[i] shl 16) or (g[i] shl 8) or b[i]
        val out = Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
        return FinalImage(input.burstId, out, p, SystemClock.elapsedRealtime() - start)
    }

    /** Unsharp mask: out = v + amount · (v − boxBlur(v)), in place. */
    private fun sharpen(ch: IntArray, w: Int, h: Int, p: PostProcessParams) {
        val blurred = boxBlur(ch, w, h, p.sharpenRadius)
        for (i in ch.indices) ch[i] = (ch[i] + p.sharpenAmount * (ch[i] - blurred[i])).toInt().coerceIn(0, 255)
    }

    /** Separable box blur with clamped edges, O(pixels) per pass. */
    private fun boxBlur(src: IntArray, w: Int, h: Int, radius: Int): IntArray {
        val tmp = IntArray(src.size)
        val out = IntArray(src.size)
        val size = 2 * radius + 1
        for (y in 0 until h) {
            val row = y * w
            var sum = 0
            for (k in -radius..radius) sum += src[row + k.coerceIn(0, w - 1)]
            for (x in 0 until w) {
                tmp[row + x] = sum / size
                sum += src[row + (x + radius + 1).coerceAtMost(w - 1)] - src[row + (x - radius).coerceAtLeast(0)]
            }
        }
        for (x in 0 until w) {
            var sum = 0
            for (k in -radius..radius) sum += tmp[k.coerceIn(0, h - 1) * w + x]
            for (y in 0 until h) {
                out[y * w + x] = sum / size
                sum += tmp[(y + radius + 1).coerceAtMost(h - 1) * w + x] - tmp[(y - radius).coerceAtLeast(0) * w + x]
            }
        }
        return out
    }
}
