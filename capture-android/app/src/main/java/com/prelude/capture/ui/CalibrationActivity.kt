package com.prelude.capture.ui

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.prelude.capture.calibration.*
import com.prelude.capture.capture.BurstCaptureManager
import com.prelude.capture.capture.SharpnessEstimator
import com.prelude.capture.databinding.ActivityCalibrationBinding
import kotlinx.coroutines.launch

/** In-app calibration debug mode + runtime settings (Results Service base URL). */
class CalibrationActivity : AppCompatActivity() {

    private lateinit var binding: ActivityCalibrationBinding
    private lateinit var repo: CalibrationRepository
    private lateinit var calculators: List<ThresholdCalculator>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityCalibrationBinding.inflate(layoutInflater)
        setContentView(binding.root)
        repo = CalibrationRepository(this)
        calculators = listOf(
            BlurTextureThresholdCalculator(repo),
            AlignmentConfidenceFloorCalculator(),
            MinFrameCountCalculator(),
            InferenceTimeoutCalculator(),
        )

        binding.baseUrlInput.setText(repo.loadBaseUrl())
        binding.saveUrlButton.setOnClickListener {
            repo.saveBaseUrl(binding.baseUrlInput.text.toString())
            binding.statusText.text = "Base URL saved: ${repo.loadBaseUrl()}"
        }
        binding.runBlurButton.setOnClickListener { runBlurCalibration() }
        refreshBlockedStatus()
    }

    private fun refreshBlockedStatus() {
        val sb = StringBuilder()
        calculators.filter { it.isBlocked }.forEach {
            sb.appendLine("[${it.type}] BLOCKED — ${it.blockedReason}").appendLine()
        }
        binding.blockedText.text = sb.ifEmpty { "No blocked calculators." }
    }

    private fun runBlurCalibration() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1); return
        }
        binding.statusText.text = "Capturing focus sweep…"
        lifecycleScope.launch {
            val cam = BurstCaptureManager(this@CalibrationActivity)
            cam.open()
            val raw = cam.captureBurst(BlurTextureThresholdCalculator.MIN_SAMPLES)
            val samples = raw.map { SharpnessEstimator.features(it.first) }
            cam.close()
            val outcome = calculators.first { it.type == ThresholdType.BLUR_TEXTURE }.calibrate(samples)
            binding.statusText.text = when (outcome) {
                is CalibrationOutcome.Success -> "Blur/texture calibrated: ${outcome.summary}"
                is CalibrationOutcome.Blocked -> "Failed: ${outcome.reason}"
            }
        }
    }
}