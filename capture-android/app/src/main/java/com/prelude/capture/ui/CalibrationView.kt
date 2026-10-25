package com.prelude.capture.ui

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.view.MotionEvent
import com.prelude.capture.calibration.DeviceThresholds

/** Calibration screen. All coordinates are dp; the hosting activity pads it to the safe area. */
class CalibrationView(context: Context, private var profile: DeviceThresholds, private val onCalibrate: () -> DeviceThresholds) : android.view.View(context) {
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val sans = Typeface.create("sans-serif", Typeface.NORMAL)
    private val mono = Typeface.create("monospace", Typeface.NORMAL)
    private val density = resources.displayMetrics.density
    private val button = RectF()
    private val bg = 0xFF1F2D40.toInt(); private val outline = 0xFF435672.toInt(); private val ink = 0xFFF1E6D2.toInt(); private val tan = 0xFFD2B48C.toInt(); private val mustard = 0xFFD8A23B.toInt()

    override fun onSizeChanged(w: Int, h: Int, oldw: Int, oldh: Int) { super.onSizeChanged(w, h, oldw, oldh); button.set(dp(20), h - dp(64), w - dp(20), h - dp(20)) }

    override fun onDraw(c: Canvas) {
        c.drawColor(bg)
        val right = width - dp(20)
        text(c, "‹  calibration", dp(16), dp(36), 24f, ink)
        line(c, 0f, dp(56), width.toFloat(), dp(56), outline)
        text(c, "pixel 8 pro", dp(20), dp(96), 18f, ink)
        text(c, "primary device  ·  ${profile.calibratedAt}", dp(20), dp(118), 11f, tan, true)
        listOf(
            "blur threshold / low" to "${profile.lowTextureBlurFloor}",
            "blur threshold / med" to "${profile.mediumTextureBlurFloor}",
            "alignment confidence floor" to "${profile.alignmentConfidenceFloor}",
            "minimum usable frames" to "${profile.minimumUsableFrames} / 8",
        ).forEachIndexed { i, (label, value) ->
            val y = dp(172 + i * 50)
            text(c, label, dp(20), y, 12f, tan)
            text(c, value, right, y, 18f, mustard, true, Paint.Align.RIGHT)
        }
        paint.color = mustard; c.drawRect(button, paint)
        text(c, "run calibration", button.left + dp(12), button.centerY() + dp(5), 14f, 0xFF2A1D05.toInt())
    }

    override fun onTouchEvent(e: MotionEvent): Boolean { if (e.action == MotionEvent.ACTION_UP && button.contains(e.x, e.y)) { profile = onCalibrate(); invalidate() }; return true }
    private fun text(c: Canvas, s: String, x: Float, y: Float, size: Float, color: Int, isMono: Boolean = false, align: Paint.Align = Paint.Align.LEFT) { paint.color = color; paint.textSize = dp(size); paint.typeface = if (isMono) mono else sans; paint.textAlign = align; c.drawText(s, x, y, paint) }
    private fun line(c: Canvas, x: Float, y: Float, r: Float, b: Float, color: Int) { paint.color = color; paint.strokeWidth = dp(1); c.drawLine(x, y, r, b, paint) }
    private fun dp(value: Int): Float = value * density
    private fun dp(value: Float): Float = value * density
}
