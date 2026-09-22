package com.prelude.capture.ui

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.Settings
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.prelude.capture.CaptureModule
import com.prelude.capture.calibration.CalibrationRepository
import com.prelude.capture.databinding.ActivityMainBinding
import kotlinx.coroutines.launch

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var repo: CalibrationRepository

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        repo = CalibrationRepository(this)

        // Standing run for casual/field capture; formal benchmarks use their own IDs.
        val deviceId = Settings.Secure.getString(contentResolver, Settings.Secure.ANDROID_ID) ?: Build.MODEL
        val runId = "dev-capture-$deviceId"

        binding.calibrateButton.setOnClickListener {
            startActivity(Intent(this, CalibrationActivity::class.java))
        }
        binding.captureButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
                != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), 1)
                return@setOnClickListener
            }
            val baseUrl = repo.loadBaseUrl() // runtime-editable via the calibration screen
            lifecycleScope.launch {
                binding.statusText.text = "Capturing…"
                val module = CaptureModule(this@MainActivity, baseUrl, runId)
                val burst = module.captureAndSubmit()
                binding.statusText.text =
                    "Burst ${burst.burstId}: ${burst.frames.size} frames, fallback=${burst.emergencyFallbackActive}"
            }
        }
    }
}