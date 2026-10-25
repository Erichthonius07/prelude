package com.prelude.capture.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import android.graphics.Typeface
import android.view.MotionEvent
import android.view.View
import com.prelude.capture.calibration.DeviceThresholds
import com.prelude.capture.model.FrameBurst
import com.prelude.capture.model.FrameSample
import com.prelude.capture.net.ComparisonLoad
import com.prelude.capture.net.ComparisonReport
import com.prelude.pipeline.PipelineModeDecision
import com.prelude.pipeline.PipelineOutput
import com.prelude.pipeline.PipelineRunner
import com.prelude.pipeline.contract.AlignedFrameSet
import com.prelude.pipeline.contract.FusionStrategy
import com.prelude.pipeline.contract.PipelineMode
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import kotlin.math.abs
import kotlin.math.min

/**
 * Full-screen instrument UI drawn on top of the camera TextureView. It renders state only: the activity feeds it
 * capture progress, pipeline stage events, the finished [PipelineOutput] and the scored [ComparisonReport].
 *
 * Every coordinate is in dp and measured from the safe area (system bars + display cutout), and all layout
 * rectangles are computed once per size/inset change in [computeLayout]. Drawing and touch handling read the same
 * rectangles, so what you see is what you can tap.
 *
 * Two kinds of numbers are kept visibly apart: scored PSNR/SSIM/LPIPS come only from the Results Service report
 * (held-out test set), while the live burst is labelled non-scored and shows only its own mode, frames and latency.
 */
class CaptureModule(
    context: Context,
    private val thresholds: DeviceThresholds,
    private val serviceLabel: String,
    private val onOpenCalibration: () -> Unit,
) : View(context) {
    private object C { const val bg = 0xFF1F2D40.toInt(); const val low = 0xFF25354A.toInt(); const val surface = 0xFF2C3E55.toInt(); const val outline = 0xFF435672.toInt(); const val ink = 0xFFF1E6D2.toInt(); const val tan = 0xFFD2B48C.toInt(); const val mustard = 0xFFD8A23B.toInt(); const val clay = 0xFFB35A3C.toInt(); const val olive = 0xFF7F9A3E.toInt(); const val onMustard = 0xFF2A1D05.toInt(); const val scrim = 0xB31F2D40.toInt() }

    /** Shutter tapped with the selected burst length. */
    var onShutter: ((Int) -> Unit)? = null
    /** Sync indicator tapped: re-fetch the comparison report and flush the metrics outbox. */
    var onSyncRequested: (() -> Unit)? = null
    /** Called whenever the camera window moves or resizes (in this view's coordinates). */
    var onViewfinderLayout: ((RectF) -> Unit)? = null

    // Paints and typefaces are created once; onDraw allocates nothing per frame.
    private val sans = Typeface.create("sans-serif", Typeface.NORMAL)
    private val sansBold = Typeface.create("sans-serif", Typeface.BOLD)
    private val monoFace = Typeface.create("monospace", Typeface.NORMAL)
    private val monoBold = Typeface.create("monospace", Typeface.BOLD)
    private val p = Paint(Paint.ANTI_ALIAS_FLAG)
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val bitmapPaint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val srcRect = Rect()
    private val density = resources.displayMetrics.density
    private val clock = SimpleDateFormat("HH:mm", Locale.US)

    private var page = 0
    private var variant = 5
    private var burstLength = 6
    private var variantOffset = 0f
    private var lastTouchX = 0f
    private var touchDownX = 0f
    private var draggingVariants = false

    // Live state, set only through the public on* methods (main thread).
    private var capturing = false
    private var captureExpected = 0
    private val capturedFrames = ArrayList<FrameSample>()
    private var processingBurst: FrameBurst? = null
    private var aligned: AlignedFrameSet? = null
    private var decision: PipelineModeDecision? = null
    private val stageState = IntArray(5)            // 0 pending, 1 running, 2 done
    private val stageDetail = Array(5) { "" }
    private val stageWarn = BooleanArray(5)
    private var output: PipelineOutput? = null
    private var comparison: ComparisonLoad = ComparisonLoad.Unavailable("not synced yet")
    private var syncing = false
    private var errorText: String? = null
    private val busy: Boolean get() = capturing || processingBurst != null

    // Safe-area insets in px, supplied by the activity's window-insets listener.
    private var insetLeft = 0
    private var insetTop = 0
    private var insetRight = 0
    private var insetBottom = 0

    // Layout, recomputed only in computeLayout().
    private val safe = RectF()
    private var contentLeft = 0f
    private var contentRight = 0f
    private var contentTop = 0f
    private var navTop = 0f
    private val viewfinderPanel = RectF()
    private val viewfinderWindow = RectF()
    private val controls = RectF()
    private val shutter = RectF()
    private val chipRects = List(5) { RectF() }
    private val imageRect = RectF()
    private val variantRow = RectF()
    private val calibrateButton = RectF()
    private val syncToggle = RectF()

    private val burstOptions = listOf(4, 5, 6, 7, 8)
    private val navLabels = listOf("capture", "results", "ablation", "device")
    private val stageNames = listOf("capture", "align", "fuse", "denoise", "post-process")
    private val variantLabels = listOf("single raw", "naive mean", "trimmed mean", "weighted", "learned fusion", "full pipeline", "fallback")
    private val variantStrategies = mapOf(1 to FusionStrategy.NAIVE_MEAN, 2 to FusionStrategy.TRIMMED_MEAN, 3 to FusionStrategy.CONFIDENCE_WEIGHTED, 4 to FusionStrategy.LEARNED)

    init { setBackgroundColor(Color.TRANSPARENT); isFocusable = true }

    // ---- state input -------------------------------------------------------------------------------------------

    fun setSafeInsets(left: Int, top: Int, right: Int, bottom: Int) {
        if (left == insetLeft && top == insetTop && right == insetRight && bottom == insetBottom) return
        insetLeft = left; insetTop = top; insetRight = right; insetBottom = bottom
        computeLayout()
        invalidate()
    }

    fun onFrameCaptured(sample: FrameSample, expected: Int) { captureExpected = expected; capturedFrames += sample; invalidate() }

    fun onBurstCaptured(burst: FrameBurst) {
        capturing = false
        processingBurst = burst
        aligned = null; decision = null
        stageState.fill(0); stageWarn.fill(false); stageDetail.fill("")
        val rejected = burst.frames.count { !it.accepted }
        stageState[0] = 2
        stageDetail[0] = "${burst.frames.size} frames  ·  $rejected blur-rejected  ·  ${bucketSummary(burst)}"
        stageWarn[0] = burst.acceptedFrames.size < 2
        invalidate()
    }

    fun onStageStarted(stage: PipelineRunner.Stage) { stageState[stage.row] = 1; invalidate() }

    fun onStageFinished(stage: PipelineRunner.Stage, summary: String, warn: Boolean) {
        stageState[stage.row] = 2; stageDetail[stage.row] = summary; stageWarn[stage.row] = warn; invalidate()
    }

    fun onAligned(set: AlignedFrameSet, decision: PipelineModeDecision) { aligned = set; this.decision = decision; invalidate() }

    fun onPipelineDone(out: PipelineOutput) {
        output = out
        processingBurst = null
        variant = 5; variantOffset = 0f
        page = 1
        invalidate()
    }

    fun onFailed(message: String) {
        capturing = false; processingBurst = null
        errorText = message
        page = 0
        invalidate()
    }

    fun setSyncing(active: Boolean) { syncing = active; invalidate() }

    fun setComparison(load: ComparisonLoad) { comparison = load; syncing = false; invalidate() }

    // ---- layout ------------------------------------------------------------------------------------------------

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) { super.onSizeChanged(w, h, oldw, oldh); computeLayout() }

    private fun computeLayout() {
        if (width == 0 || height == 0) return
        safe.set(insetLeft.toFloat(), insetTop.toFloat(), (width - insetRight).toFloat(), (height - insetBottom).toFloat())
        contentLeft = safe.left + dp(16)
        contentRight = safe.right - dp(16)
        contentTop = safe.top + dp(112)
        navTop = safe.bottom - dp(48)
        syncToggle.set(safe.right - dp(54), safe.top, safe.right, safe.top + dp(56))

        // Capture page: viewfinder panel sits between the title and the status band, controls docked at its bottom.
        viewfinderPanel.set(contentLeft, contentTop, contentRight, safe.bottom - dp(108))
        controls.set(viewfinderPanel.left, viewfinderPanel.bottom - dp(100), viewfinderPanel.right, viewfinderPanel.bottom)
        viewfinderWindow.set(viewfinderPanel.left, viewfinderPanel.top, viewfinderPanel.right, controls.top)
        shutter.set(controls.right - dp(84), controls.top + dp(14), controls.right - dp(12), controls.top + dp(86))
        val chipStep = min(dp(44), (shutter.left - dp(12) - (controls.left + dp(12))) / burstOptions.size)
        chipRects.forEachIndexed { i, r ->
            val x = controls.left + dp(12) + i * chipStep
            r.set(x, controls.top + dp(32), x + chipStep - dp(4), controls.top + dp(66))
        }

        imageRect.set(contentLeft, contentTop, contentRight, contentTop + dp(190))
        variantRow.set(contentLeft, contentTop + dp(332), contentRight, contentTop + dp(378))
        variantOffset = variantOffset.coerceIn(0f, maxVariantOffset())
        calibrateButton.set(contentLeft, safe.bottom - dp(96), contentRight, safe.bottom - dp(56))

        onViewfinderLayout?.invoke(RectF(viewfinderWindow))
    }

    // ---- drawing -----------------------------------------------------------------------------------------------

    override fun onDraw(c: Canvas) {
        super.onDraw(c)
        val processingNow = processingBurst != null
        if (page == 0 && !processingNow) {
            // Opaque chrome everywhere except the camera window; the preview never bleeds under text.
            c.save(); c.clipOutRect(viewfinderWindow); c.drawColor(C.bg); c.restore()
        } else {
            c.drawColor(C.bg)
        }
        topBar(c)
        when {
            processingNow -> processing(c, processingBurst!!)
            page == 0 -> capture(c)
            page == 1 -> results(c)
            page == 2 -> ablation(c)
            else -> device(c)
        }
        bottomNav(c)
    }

    private fun topBar(c: Canvas) {
        // Background extends up under the status bar so it matches on edge-to-edge devices.
        rect(c, 0f, 0f, width.toFloat(), safe.top + dp(56), C.bg)
        pixelCross(c, safe.left + dp(24), safe.top + dp(28), dp(2.5f), C.mustard)
        text(c, "prelude", safe.left + dp(38), safe.top + dp(34), 16f, C.ink, false, true)
        text(c, "primary / calibrated ${thresholds.calibratedAt}", safe.right - dp(50), safe.top + dp(32), 10f, C.tan, true, align = Paint.Align.RIGHT, maxWidth = safe.width() - dp(170))
        val (glyph, color) = when {
            syncing -> "…" to C.tan
            comparison is ComparisonLoad.Fresh -> "↻" to C.olive
            comparison is ComparisonLoad.Cached -> "⌁" to C.mustard
            else -> "⌁" to C.clay
        }
        text(c, glyph, syncToggle.centerX(), safe.top + dp(35), 20f, color, false, align = Paint.Align.CENTER)
        line(c, 0f, safe.top + dp(56), width.toFloat(), safe.top + dp(56), C.outline)
    }

    private fun capture(c: Canvas) {
        title(c, "capture", "viewfinder / low-light")
        val w = viewfinderWindow
        val length = if (capturing) captureExpected.takeIf { it > 0 } ?: burstLength else burstLength

        // Status strip on a scrim so the lock labels stay legible over a bright scene.
        rect(c, w.left, w.top, w.right, w.top + dp(30), C.scrim)
        var x = w.left + dp(12)
        val labelY = w.top + dp(20)
        listOf("ae lock", "awb lock", "af lock").forEach { label ->
            x += text(c, label, x, labelY, 10f, if (capturing) C.mustard else C.tan, true) + dp(12)
        }
        val latest = capturedFrames.lastOrNull()
        val exposure = if (capturing && latest != null) "frame ${capturedFrames.size}  ·  vol ${"%.3f".format(Locale.US, latest.varianceOfLaplacian)}" else "auto  ·  continuous af"
        text(c, exposure, w.right - dp(12), labelY, 10f, C.tan, true, align = Paint.Align.RIGHT, maxWidth = w.right - dp(12) - x)

        val gridTop = w.top + dp(30)
        for (i in 1..2) { val gx = w.left + w.width() * i / 3f; line(c, gx, gridTop, gx, w.bottom, C.outline) }
        for (i in 1..2) { val gy = gridTop + (w.bottom - gridTop) * i / 3f; line(c, w.left, gy, w.right, gy, C.outline) }
        pixelCross(c, w.centerX(), (gridTop + w.bottom) / 2f, dp(3f), if (capturing) C.mustard else C.tan)
        rectStroke(c, w.left, w.top, w.right, w.bottom, C.outline)

        rect(c, controls.left, controls.top, controls.right, controls.bottom, C.surface)
        text(c, "burst length", controls.left + dp(12), controls.top + dp(22), 12f, C.tan, false)
        burstOptions.forEachIndexed { i, n ->
            val r = chipRects[i]; val selected = n == length
            rectStroke(c, r.left, r.top, r.right, r.bottom, if (selected) C.mustard else C.outline)
            text(c, "$n", r.centerX(), r.centerY() + dp(4), 12f, when { selected -> C.mustard; capturing -> C.outline; else -> C.ink }, true, align = Paint.Align.CENTER)
        }

        // Frame strip: empty slots while idle; during capture each real frame lands green (kept) or clay (blur-rejected).
        val stripLabelY = controls.top + dp(88)
        val stripX = controls.left + dp(12) + text(c, "frame strip", controls.left + dp(12), stripLabelY, 11f, C.tan, false) + dp(10)
        repeat(length) { i ->
            val fx = stripX + i * dp(14)
            val frame = if (capturing) capturedFrames.getOrNull(i) else null
            if (frame != null) rect(c, fx, stripLabelY - dp(12), fx + dp(8), stripLabelY + dp(2), if (frame.accepted) C.olive else C.clay)
            else rectStroke(c, fx, stripLabelY - dp(12), fx + dp(8), stripLabelY + dp(2), C.outline)
        }

        rect(c, shutter.left, shutter.top, shutter.right, shutter.bottom, if (capturing) C.outline else C.mustard)
        if (capturing && length > 0) rect(c, shutter.left, shutter.bottom - shutter.height() * capturedFrames.size / length, shutter.right, shutter.bottom, C.mustard)
        p.color = C.onMustard; c.drawCircle(shutter.centerX(), shutter.centerY(), dp(8), p)

        val error = errorText
        when {
            capturing -> band(c, contentLeft, safe.bottom - dp(96), contentRight, safe.bottom - dp(68), "capturing  /  frame ${capturedFrames.size} of $length  ·  ae/awb/af locked", C.mustard)
            error != null -> band(c, contentLeft, safe.bottom - dp(96), contentRight, safe.bottom - dp(68), "error  /  $error", C.clay)
            else -> band(c, contentLeft, safe.bottom - dp(96), contentRight, safe.bottom - dp(68), "ready  /  hold steady for burst", C.mustard)
        }
    }

    private fun processing(c: Canvas, burst: FrameBurst) {
        title(c, "processing", "${burst.burstId} / ${decision?.mode ?: "deciding mode"}")
        val done = stageState.count { it == 2 }
        rect(c, contentLeft, contentTop, contentRight, contentTop + dp(2), C.outline)
        rect(c, contentLeft, contentTop, contentLeft + (contentRight - contentLeft) * done / stageNames.size, contentTop + dp(2), C.mustard)

        stageNames.forEachIndexed { i, name ->
            val y = contentTop + dp(34 + i * 58)
            val state = stageState[i]
            line(c, safe.left, y + dp(40), safe.right, y + dp(40), C.outline)
            text(c, "0${i + 1}", contentLeft, y + dp(8), 11f, if (state == 0) C.outline else C.tan, true)
            if (state == 1) pixelCross(c, safe.left + dp(48), y + dp(4), dp(1.5f), C.mustard)
            text(c, name, safe.left + dp(66), y + dp(8), 16f, if (state == 0) C.tan else C.ink, false, true)
            val detail = when (state) { 0 -> "pending  ·  —"; 1 -> "running"; else -> stageDetail[i] }
            val color = when { state == 0 -> C.outline; state == 1 -> C.mustard; stageWarn[i] -> C.mustard; else -> C.tan }
            text(c, detail, safe.left + dp(66), y + dp(26), 10f, color, true, maxWidth = contentRight - safe.left - dp(90))
            if (state == 2) text(c, if (stageWarn[i]) "!" else "✓", contentRight, y + dp(8), 12f, if (stageWarn[i]) C.mustard else C.olive, false, align = Paint.Align.RIGHT)
        }

        // Per-frame view: sharpness and blur rejection, then RANSAC acceptance once alignment has reported.
        val framesTop = contentTop + dp(336)
        text(c, "frames  /  align floor %.2f  ·  ref = sharpest".format(Locale.US, thresholds.alignmentConfidenceFloor), contentLeft, framesTop, 10f, C.tan, true, maxWidth = contentRight - contentLeft)
        val n = burst.frames.size
        val cell = min(dp(40), (contentRight - contentLeft - (n - 1) * dp(6)) / n)
        val set = aligned
        burst.frames.forEachIndexed { i, f ->
            val fx = contentLeft + i * (cell + dp(6)); val top = framesTop + dp(12); val bottom = top + dp(30)
            val a = set?.frames?.firstOrNull { it.sourceIndex == f.index }
            when {
                !f.accepted -> rect(c, fx, top, fx + cell, bottom, C.clay)
                set == null -> rectStroke(c, fx, top, fx + cell, bottom, C.ink)
                a != null && a.accepted -> rect(c, fx, top, fx + cell, bottom, C.olive)
                else -> rectStroke(c, fx, top, fx + cell, bottom, C.mustard)
            }
            if (f.index == burst.referenceIndex) text(c, "ref", fx + cell / 2f, top + dp(19), 9f, if (f.accepted && set != null) C.onMustard else C.ink, true, align = Paint.Align.CENTER)
            text(c, ".%03d".format(Locale.US, (f.varianceOfLaplacian * 1000).toInt().coerceIn(0, 999)), fx + cell / 2f, bottom + dp(14), 9f, C.tan, true, align = Paint.Align.CENTER, maxWidth = cell + dp(4))
        }

        band(c, contentLeft, safe.bottom - dp(96), contentRight, safe.bottom - dp(66), "denoise budget  /  ${thresholds.inferenceTimeoutMs} ms  ·  discard-race in denoise module", C.mustard)
    }

    private fun results(c: Canvas) {
        val report = comparison.reportOrNull
        val out = output
        title(c, "results", out?.let { "${it.burst.burstId} / ${it.decision.mode}" } ?: "no live burst yet")

        // Live burst, split view: the raw reference frame on the left, the selected variant's output on the right.
        val top = imageRect.top; val imageBottom = imageRect.bottom; val mid = imageRect.centerX()
        rect(c, imageRect.left, top, imageRect.right, imageBottom, C.low)
        val raw = out?.burst?.frames?.firstOrNull { it.index == out.burst.referenceIndex }?.bitmap
        val processed = out?.let { liveImageFor(it, variant) }
        if (raw != null) drawCover(c, raw, imageRect, imageRect.left, mid)
        if (processed != null) drawCover(c, processed, imageRect, mid, imageRect.right)
        rect(c, imageRect.left, top, imageRect.right, top + dp(34), C.scrim)
        text(c, "single frame", contentLeft + dp(12), top + dp(22), 10f, C.tan, true)
        text(c, variantLabels[variant], mid + dp(12), top + dp(22), 10f, C.ink, true, maxWidth = imageRect.right - mid - dp(20))
        if (out == null) text(c, "capture a burst to compare live", imageRect.centerX(), imageRect.centerY() + dp(4), 11f, C.tan, true, align = Paint.Align.CENTER)
        else if (processed == null) text(c, "not produced for this burst", (mid + imageRect.right) / 2f, imageRect.centerY() + dp(4), 9f, C.tan, true, align = Paint.Align.CENTER, maxWidth = imageRect.right - mid - dp(16))
        line(c, mid, top, mid, imageBottom, C.mustard)
        rect(c, imageRect.left, imageBottom - dp(26), mid, imageBottom, C.scrim)
        text(c, "live burst  ·  non-scored", contentLeft + dp(12), imageBottom - dp(9), 9f, C.tan, true, maxWidth = mid - contentLeft - dp(16))

        // Scored numbers: held-out test set only, from the Results Service.
        val v = report?.variant(ComparisonReport.VARIANT_IDS[variant])
        text(c, if (report != null) "test-set psnr delta" else "scored results", contentLeft, top + dp(226), 12f, C.tan, false)
        val headline = when {
            v?.deltaPsnr != null -> "%+.2f dB".format(Locale.US, v.deltaPsnr)
            v != null && v.significance == "baseline" -> "baseline"
            else -> "—"
        }
        text(c, headline, contentLeft, top + dp(262), 28f, if (v?.deltaPsnr != null) C.mustard else C.outline, true)
        val sub = when {
            report == null -> "no data  ·  tap ⌁ to sync"
            v == null -> "not in report"
            v.ci95 != null -> "%s  ·  95%% ci [%.2f, %.2f]".format(Locale.US, v.significance, v.ci95.first, v.ci95.second)
            else -> v.significance
        }
        text(c, sub, contentLeft, top + dp(282), 10f, if (v?.significance == "significant") C.olive else C.tan, true, maxWidth = mid - contentLeft)
        val metrics = if (v != null) "ssim   ${fmt(v.ssim, "%.3f")}\nlpips  ${fmt(v.lpips, "%.3f")}\np95    ${v.p95LatencyMs?.let { "$it ms" } ?: "—"}" else "ssim   —\nlpips  —\np95    —"
        text(c, metrics, contentRight, top + dp(226), 11f, C.ink, true, align = Paint.Align.RIGHT)

        if (out != null) {
            val usable = out.aligned.usableFrames.size
            text(c, "this burst  ·  ${out.decision.mode}  ·  $usable of ${out.burst.frames.size} frames  ·  ${out.processingMs} ms", contentLeft, top + dp(303), 9f, C.tan, true, maxWidth = contentRight - contentLeft)
        }

        text(c, "evaluation variant" + (report?.let { "  ·  N = ${it.testSetN}" } ?: ""), contentLeft, top + dp(322), 12f, C.tan, false)
        c.save(); c.clipRect(variantRow)
        variantLabels.forEachIndexed { i, label ->
            val x = variantRow.left + i * dp(90) - variantOffset
            val selected = i == variant
            val hasData = report?.variant(ComparisonReport.VARIANT_IDS[i]) != null || (out != null && liveImageFor(out, i) != null)
            rectStroke(c, x, variantRow.top + dp(6), x + dp(84), variantRow.bottom - dp(8), if (selected) C.mustard else C.outline)
            text(c, label, x + dp(42), variantRow.top + dp(27), 9f, when { selected -> C.mustard; hasData -> C.ink; else -> C.outline }, false, align = Paint.Align.CENTER, maxWidth = dp(76))
        }
        c.restore()
        syncBand(c)
    }

    /** Live image for an evaluation variant, or null when this burst didn't produce one. */
    private fun liveImageFor(out: PipelineOutput, index: Int): Bitmap? = when (index) {
        0 -> out.burst.frames.firstOrNull { it.index == out.burst.referenceIndex }?.bitmap
        in 1..4 -> out.fusion.outputs[variantStrategies[index]]?.bitmap
        5 -> out.final.bitmap
        else -> if (out.decision.pipelineMode == PipelineMode.FUSION_SINGLE) out.final.bitmap else null
    }

    private fun ablation(c: Canvas) {
        val report = comparison.reportOrNull
        title(c, "ablation", report?.let { "held-out test set / N = ${it.testSetN}" } ?: "held-out test set")
        if (report == null) {
            text(c, "no scored comparison yet", contentLeft, contentTop + dp(40), 14f, C.ink, false)
            text(c, "sync with the results service (top right)", contentLeft, contentTop + dp(62), 10f, C.tan, true, maxWidth = contentRight - contentLeft)
            syncBand(c)
            return
        }
        val tableWidth = contentRight - contentLeft - dp(24)
        text(c, "variant          psnr   ssim     Δ    p95", contentLeft, contentTop + dp(12), 10f, C.tan, true, maxWidth = tableWidth)
        var previous: Double? = null
        ComparisonReport.VARIANT_IDS.forEachIndexed { i, id ->
            val v = report.variant(id)
            val y = contentTop + dp(36 + i * 32)
            val step = if (v?.psnr != null && previous != null && i in 1..5) "%+.2f".format(Locale.US, v.psnr - previous!!) else "—"
            val row = "%-15s %5s  %5s  %5s  %4s".format(Locale.US, variantLabels[i], fmt(v?.psnr, "%.2f"), fmt(v?.ssim, "%.3f").removePrefix("0"), step, v?.p95LatencyMs?.toString() ?: "—")
            text(c, row, contentLeft, y, 10f, if (v == null) C.outline else C.ink, true, maxWidth = tableWidth)
            line(c, contentLeft, y + dp(10), contentRight, y + dp(10), C.outline)
            if (v != null && v.significance != "baseline") {
                val sig = v.significance == "significant"
                text(c, if (sig) "✓" else "!", contentRight, y, 12f, if (sig) C.olive else C.mustard, false, align = Paint.Align.RIGHT)
            }
            if (v?.psnr != null && i <= 4) previous = v.psnr
        }

        val gainsTop = contentTop + dp(270)
        text(c, "per-stage gain / 95% ci", contentLeft, gainsTop, 12f, C.tan, false)
        val maxGain = report.stageGains.maxOfOrNull { abs(it.deltaPsnr) }?.takeIf { it > 0 } ?: 1.0
        val barLeft = contentLeft + dp(96); val barMax = contentRight - barLeft - dp(56)
        report.stageGains.take(4).forEachIndexed { i, g ->
            val y = gainsTop + dp(28 + i * 26)
            text(c, g.stage, contentLeft, y, 11f, C.ink, false, maxWidth = dp(90))
            rect(c, barLeft, y - dp(10), barLeft + barMax * (abs(g.deltaPsnr) / maxGain).toFloat(), y, if (g.stage.contains("denoise")) C.mustard else C.tan)
            text(c, "%+.2f".format(Locale.US, g.deltaPsnr), contentRight, y, 10f, C.ink, true, align = Paint.Align.RIGHT)
        }
        val gap = if (report.learnedFusionTrainSsim != null && report.learnedFusionValSsim != null) "learned fusion train−val ssim %.3f".format(Locale.US, report.learnedFusionTrainSsim - report.learnedFusionValSsim) else null
        val post = report.postProcessSsimContribution?.let { "post-process alone %+.3f ssim".format(Locale.US, it) }
        text(c, listOfNotNull(post, gap).joinToString("  ·  ").ifEmpty { "full pipeline delta = ai denoise + post-process" }, contentLeft, safe.bottom - dp(108), 9f, C.tan, true, maxWidth = contentRight - contentLeft)
        syncBand(c)
    }

    private fun device(c: Canvas) {
        title(c, "device", "calibration / profile")
        rect(c, contentLeft, contentTop, contentRight, contentTop + dp(62), C.surface)
        text(c, android.os.Build.MODEL.lowercase(Locale.US), contentLeft + dp(12), contentTop + dp(28), 16f, C.ink, false, true, maxWidth = contentRight - contentLeft - dp(24))
        text(c, "calibrated ${thresholds.calibratedAt}", contentLeft + dp(12), contentTop + dp(48), 10f, C.tan, true, maxWidth = contentRight - contentLeft - dp(24))
        text(c, "thresholds", contentLeft, contentTop + dp(92), 12f, C.tan, false)
        val colWidth = (contentRight - contentLeft - dp(8)) / 2f
        listOf(
            "blur / low" to "${thresholds.lowTextureBlurFloor}",
            "blur / med" to "${thresholds.mediumTextureBlurFloor}",
            "alignment floor" to "${thresholds.alignmentConfidenceFloor}",
            "usable frames" to "${thresholds.minimumUsableFrames} / 8",
        ).forEachIndexed { i, item ->
            val x = contentLeft + (i % 2) * (colWidth + dp(8)); val y = contentTop + dp(110 + (i / 2) * 90)
            rect(c, x, y, x + colWidth, y + dp(82), C.surface)
            text(c, item.first, x + dp(10), y + dp(22), 10f, C.tan, false)
            text(c, item.second, x + dp(10), y + dp(56), 20f, C.mustard, true)
        }
        text(c, "results service  /  $serviceLabel", contentLeft, contentTop + dp(300), 10f, C.tan, true, maxWidth = contentRight - contentLeft)
        band(c, contentLeft, safe.bottom - dp(142), contentRight, safe.bottom - dp(112), "inference timeout  /  ${thresholds.inferenceTimeoutMs} ms cap", C.mustard)
        val b = calibrateButton
        rect(c, b.left, b.top, b.right, b.bottom, C.mustard)
        text(c, "run calibration", b.left + dp(12), b.centerY() + dp(5), 13f, C.onMustard, false, true)
    }

    private fun syncBand(c: Canvas) {
        val (label, color) = when (val load = comparison) {
            is ComparisonLoad.Fresh -> "online  /  results fetched ${clock.format(Date(load.fetchedAt))}" to C.olive
            is ComparisonLoad.Cached -> "offline  /  cached results from ${if (load.fetchedAt > 0) clock.format(Date(load.fetchedAt)) else "earlier"}" to C.mustard
            is ComparisonLoad.Unavailable -> "no results  /  ${load.reason}" to C.clay
        }
        band(c, contentLeft, safe.bottom - dp(96), contentRight, safe.bottom - dp(68), if (syncing) "syncing  /  contacting results service" else label, color)
    }

    private fun bottomNav(c: Canvas) {
        // Background extends down under the gesture/nav bar.
        rect(c, 0f, navTop, width.toFloat(), height.toFloat(), C.low)
        val itemWidth = safe.width() / navLabels.size
        navLabels.forEachIndexed { i, label ->
            val x = safe.left + i * itemWidth; val active = page == i && !busy
            if (active) rect(c, x + dp(6), navTop + dp(6), x + itemWidth - dp(6), navTop + dp(42), C.mustard)
            text(c, label, x + itemWidth / 2f, navTop + dp(28), 11f, when { active -> C.onMustard; busy -> C.outline; else -> C.ink }, false, align = Paint.Align.CENTER, maxWidth = itemWidth - dp(16))
        }
    }

    // ---- input -------------------------------------------------------------------------------------------------

    override fun onTouchEvent(e: MotionEvent): Boolean {
        if (busy) return true
        when (e.actionMasked) {
            MotionEvent.ACTION_DOWN -> {
                touchDownX = e.x; lastTouchX = e.x
                draggingVariants = page == 1 && variantRow.contains(e.x, e.y)
                if (draggingVariants) parent?.requestDisallowInterceptTouchEvent(true)
                return true
            }
            MotionEvent.ACTION_MOVE -> {
                if (draggingVariants) {
                    variantOffset = (variantOffset + lastTouchX - e.x).coerceIn(0f, maxVariantOffset())
                    lastTouchX = e.x; invalidate()
                }
                return true
            }
            MotionEvent.ACTION_CANCEL -> { draggingVariants = false; return true }
            MotionEvent.ACTION_UP -> Unit
            else -> return true
        }

        if (draggingVariants) {
            if (abs(touchDownX - e.x) < dp(12)) {
                val selected = ((e.x - variantRow.left + variantOffset) / dp(90)).toInt()
                if (selected in variantLabels.indices) variant = selected
            }
            draggingVariants = false; invalidate(); return true
        }
        if (e.y >= navTop) {
            page = ((e.x - safe.left) / (safe.width() / navLabels.size)).toInt().coerceIn(0, navLabels.size - 1)
            invalidate(); return true
        }
        if (syncToggle.contains(e.x, e.y)) { if (!syncing) { syncing = true; invalidate(); onSyncRequested?.invoke() }; return true }
        if (page == 0) {
            chipRects.forEachIndexed { i, r -> if (r.contains(e.x, e.y)) { burstLength = burstOptions[i]; invalidate(); return true } }
            if (shutter.contains(e.x, e.y)) {
                capturing = true; captureExpected = burstLength; capturedFrames.clear(); errorText = null
                invalidate()
                onShutter?.invoke(burstLength)
                return true
            }
        }
        if (page == 3 && calibrateButton.contains(e.x, e.y)) onOpenCalibration()
        return true
    }

    // ---- helpers -----------------------------------------------------------------------------------------------

    private fun bucketSummary(burst: FrameBurst): String {
        val buckets = burst.frames.mapNotNull { it.textureBucket?.name?.lowercase(Locale.US) }
        return buckets.groupingBy { it }.eachCount().maxByOrNull { it.value }?.key?.let { "$it texture" } ?: "texture n/a"
    }

    private fun fmt(value: Double?, pattern: String): String = value?.let { pattern.format(Locale.US, it) } ?: "—"

    private fun maxVariantOffset(): Float = (dp(variantLabels.size * 90) - variantRow.width()).coerceAtLeast(0f)

    /** Draws [bitmap] centre-cropped to fill [dst], but only the horizontal slice [clipLeft]..[clipRight]. */
    private fun drawCover(c: Canvas, bitmap: Bitmap, dst: RectF, clipLeft: Float, clipRight: Float) {
        val dstAspect = dst.width() / dst.height()
        val bw = bitmap.width.toFloat(); val bh = bitmap.height.toFloat()
        if (bw / bh > dstAspect) { val sw = bh * dstAspect; srcRect.set(((bw - sw) / 2f).toInt(), 0, ((bw + sw) / 2f).toInt(), bh.toInt()) }
        else { val sh = bw / dstAspect; srcRect.set(0, ((bh - sh) / 2f).toInt(), bw.toInt(), ((bh + sh) / 2f).toInt()) }
        c.save(); c.clipRect(clipLeft, dst.top, clipRight, dst.bottom); c.drawBitmap(bitmap, srcRect, dst, bitmapPaint); c.restore()
    }

    private fun title(c: Canvas, name: String, detail: String) {
        text(c, name, contentLeft, safe.top + dp(86), 24f, C.ink, false, true)
        text(c, detail, contentLeft, safe.top + dp(104), 10f, C.tan, true, maxWidth = contentRight - contentLeft)
    }
    private fun band(c: Canvas, l: Float, t: Float, r: Float, b: Float, label: String, color: Int) { rectStroke(c, l, t, r, b, color); text(c, label, l + dp(10), t + (b - t) / 2f + dp(3.5f), 10f, color, true, maxWidth = r - l - dp(20)) }
    /** Five-cell "+" glyph centred on (cx, cy); [cell] is the side of one cell in px. */
    private fun pixelCross(c: Canvas, cx: Float, cy: Float, cell: Float, color: Int) { val step = cell * 1.4f; listOf(0 to -2, -2 to 0, 0 to 0, 2 to 0, 0 to 2).forEach { (dx, dy) -> val x = cx + dx * step - cell / 2f; val y = cy + dy * step - cell / 2f; rect(c, x, y, x + cell, y + cell, color) } }
    private fun rect(c: Canvas, l: Float, t: Float, r: Float, b: Float, color: Int) { p.color = color; p.style = Paint.Style.FILL; c.drawRect(l, t, r, b, p) }
    private fun rectStroke(c: Canvas, l: Float, t: Float, r: Float, b: Float, color: Int) { p.color = color; p.style = Paint.Style.STROKE; p.strokeWidth = dp(1); c.drawRect(l, t, r, b, p); p.style = Paint.Style.FILL }
    private fun line(c: Canvas, x1: Float, y1: Float, x2: Float, y2: Float, color: Int) { p.color = color; p.strokeWidth = dp(1); c.drawLine(x1, y1, x2, y2, p) }

    /**
     * Draws (possibly multi-line) text at a baseline and returns the widest line's width in px.
     * When [maxWidth] is set, the text is scaled down to fit instead of running into its neighbours.
     */
    private fun text(c: Canvas, value: String, x: Float, y: Float, size: Float, color: Int, isMono: Boolean, bold: Boolean = false, align: Paint.Align = Paint.Align.LEFT, maxWidth: Float = 0f): Float {
        val paint = textPaint
        paint.color = color
        paint.typeface = if (isMono) (if (bold) monoBold else monoFace) else (if (bold) sansBold else sans)
        paint.textAlign = align
        paint.textSize = dp(size)
        val rows = if (value.indexOf('\n') >= 0) value.split('\n') else null
        var widest = if (rows == null) paint.measureText(value) else rows.maxOf { paint.measureText(it) }
        if (maxWidth > 0f && widest > maxWidth) { paint.textSize *= maxWidth / widest; widest = maxWidth }
        if (rows == null) c.drawText(value, x, y, paint)
        else rows.forEachIndexed { i, row -> c.drawText(row, x, y + i * (paint.textSize + dp(5)), paint) }
        return widest
    }
    private fun dp(value: Int): Float = value * density
    private fun dp(value: Float): Float = value * density
}
