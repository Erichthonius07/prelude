package com.prelude.capture.ui

import android.os.Bundle
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import com.prelude.capture.calibration.CalibrationRepository

class CalibrationActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        WindowCompat.setDecorFitsSystemWindows(window, false)
        val repository = CalibrationRepository()
        // No camera here, so plain padding is safe: the root's background fills the bar areas, the view stays inside the safe area.
        val root = FrameLayout(this).apply {
            setBackgroundColor(0xFF1F2D40.toInt())
            addView(CalibrationView(this@CalibrationActivity, repository.load()) { repository.runCalibration() }, FrameLayout.LayoutParams(-1, -1))
        }
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val safe = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            view.setPadding(safe.left, safe.top, safe.right, safe.bottom)
            WindowInsetsCompat.CONSUMED
        }
        setContentView(root)
    }
}
