package com.prelude.capture.ui

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.RectF
import android.graphics.SurfaceTexture
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraManager
import android.os.Build
import android.os.Bundle
import android.util.Size
import android.view.Surface
import android.view.TextureView
import android.widget.FrameLayout
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import com.prelude.PreludeConfig
import com.prelude.capture.calibration.CalibrationRepository
import com.prelude.capture.capture.BurstCaptureManager
import com.prelude.capture.model.FrameBurst
import com.prelude.capture.model.FrameSample
import com.prelude.capture.net.ComparisonLoad
import com.prelude.capture.net.MetricsOutbox
import com.prelude.capture.net.ResultsServiceClient
import com.prelude.pipeline.PipelineFactory
import com.prelude.pipeline.PipelineModeDecision
import com.prelude.pipeline.PipelineOutput
import com.prelude.pipeline.PipelineRunner
import com.prelude.pipeline.contract.AlignedFrameSet
import com.prelude.pipeline.contract.FusionStrategy
import java.io.File
import java.util.concurrent.Executors

/**
 * Composition root for the demo app: camera → [BurstCaptureManager] → [PipelineRunner] → [CaptureModule], plus
 * the Results Service sync (on open and on tap, never polling).
 */
class MainActivity : ComponentActivity() {
    private lateinit var cameraPreview: TextureView
    private lateinit var captureModule: CaptureModule
    private lateinit var cameraManager: BurstCaptureManager
    private lateinit var pipeline: PipelineRunner
    private lateinit var resultsClient: ResultsServiceClient
    private lateinit var outbox: MetricsOutbox
    private val io = Executors.newSingleThreadExecutor()
    private val previewSize: Size by lazy { choosePreviewSize() }
    private val viewfinder = RectF()
    private val cameraPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        if (granted && cameraPreview.isAvailable) cameraPreview.surfaceTexture?.let { openCamera(it) }
        else if (!granted) captureModule.onFailed("camera permission denied")
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        // Draw edge-to-edge on every API level (Android 15 enforces it anyway) and let CaptureModule
        // paint behind the bars, so layout is identical on all devices.
        WindowCompat.setDecorFitsSystemWindows(window, false)
        window.statusBarColor = Color.TRANSPARENT
        window.navigationBarColor = Color.TRANSPARENT
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) window.isNavigationBarContrastEnforced = false

        val thresholds = CalibrationRepository().load()
        outbox = MetricsOutbox(File(filesDir, "metrics_outbox.jsonl"))
        resultsClient = ResultsServiceClient(PreludeConfig.RESULTS_SERVICE_BASE_URL, PreludeConfig.DEVICE_ID, filesDir, PreludeConfig.RUN_ID)
        pipeline = PipelineFactory.create(thresholds, outbox, PreludeConfig.DEVICE_ID)

        cameraPreview = TextureView(this)
        captureModule = CaptureModule(this, thresholds, PreludeConfig.RESULTS_SERVICE_BASE_URL) { startActivity(android.content.Intent(this, CalibrationActivity::class.java)) }
        captureModule.onViewfinderLayout = { rect -> viewfinder.set(rect); updatePreviewTransform() }
        captureModule.onShutter = { length -> cameraManager.captureBurst(length) }
        captureModule.onSyncRequested = { sync() }

        // Both children stay full-screen and are never re-laid out for insets; resizing the TextureView
        // after the camera session starts is what made the preview jump and stretch.
        val root = FrameLayout(this).apply {
            setBackgroundColor(0xFF1F2D40.toInt())
            addView(cameraPreview, FrameLayout.LayoutParams(-1, -1))
            addView(captureModule, FrameLayout.LayoutParams(-1, -1))
        }
        ViewCompat.setOnApplyWindowInsetsListener(captureModule) { _, insets ->
            val safe = insets.getInsets(WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.displayCutout())
            captureModule.setSafeInsets(safe.left, safe.top, safe.right, safe.bottom)
            WindowInsetsCompat.CONSUMED
        }
        setContentView(root)

        cameraManager = BurstCaptureManager(this, thresholds, object : BurstCaptureManager.Listener {
            override fun onFrameCaptured(sample: FrameSample, expected: Int) = captureModule.onFrameCaptured(sample, expected)
            override fun onBurstReady(burst: FrameBurst) { captureModule.onBurstCaptured(burst); pipeline.process(burst, pipelineListener) }
            override fun onCaptureError(error: Throwable) = captureModule.onFailed(error.message ?: "camera error")
        })
        cameraPreview.surfaceTextureListener = object : TextureView.SurfaceTextureListener {
            override fun onSurfaceTextureAvailable(surface: SurfaceTexture, width: Int, height: Int) { updatePreviewTransform(); openCamera(surface) }
            override fun onSurfaceTextureSizeChanged(surface: SurfaceTexture, width: Int, height: Int) { surface.setDefaultBufferSize(previewSize.width, previewSize.height); updatePreviewTransform() }
            override fun onSurfaceTextureDestroyed(surface: SurfaceTexture): Boolean { cameraManager.closeCamera(); return true }
            override fun onSurfaceTextureUpdated(surface: SurfaceTexture) = Unit
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
        }

        // Show the cached report instantly, then fetch once (spec 3.4: request-triggered on open, cached for offline).
        captureModule.setComparison(resultsClient.cachedComparison())
        applyReport(resultsClient.cachedComparison())
        sync()
    }

    private val pipelineListener = object : PipelineRunner.Listener {
        override fun onStageStarted(stage: PipelineRunner.Stage) = captureModule.onStageStarted(stage)
        override fun onStageFinished(stage: PipelineRunner.Stage, summary: String, warn: Boolean) = captureModule.onStageFinished(stage, summary, warn)
        override fun onAligned(aligned: AlignedFrameSet, decision: PipelineModeDecision) = captureModule.onAligned(aligned, decision)
        override fun onPipelineDone(output: PipelineOutput) { captureModule.onPipelineDone(output); io.execute { resultsClient.uploadOutbox(outbox) } }
        override fun onPipelineFailed(stage: PipelineRunner.Stage, error: Throwable) = captureModule.onFailed("${stage.name.lowercase()} failed: ${error.message}")
    }

    /** Fetches the comparison report and flushes queued live-burst metrics, off the main thread. */
    private fun sync() {
        captureModule.setSyncing(true)
        io.execute {
            val load = resultsClient.loadComparison()
            resultsClient.uploadOutbox(outbox)
            runOnUiThread { applyReport(load); captureModule.setComparison(load) }
        }
    }

    /** Scored results drive two live settings: which fusion output is denoised, and the tuned post-process params. */
    private fun applyReport(load: ComparisonLoad) {
        val report = load.reportOrNull ?: return
        FusionStrategy.values().firstOrNull { it.wire == report.bestFusionStrategy }?.let { pipeline.preferredStrategy = it }
        report.postProcessParams?.let { pipeline.postProcess.params = it }
    }

    private fun openCamera(texture: SurfaceTexture) {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) return
        // Fix the buffer to a real camera size; otherwise it follows the view size and the camera picks an arbitrary, stretched one.
        texture.setDefaultBufferSize(previewSize.width, previewSize.height)
        cameraManager.startPreview(Surface(texture))
    }

    /**
     * Maps the full-screen preview into the viewfinder window with centre-crop, keeping the camera's
     * aspect ratio. Only a matrix changes here, never the view's layout, so this is cheap and jitter-free.
     */
    private fun updatePreviewTransform() {
        val viewW = cameraPreview.width.toFloat(); val viewH = cameraPreview.height.toFloat()
        if (viewW == 0f || viewH == 0f || viewfinder.isEmpty) return
        // Sensor buffers are landscape; the activity is portrait-locked, so the displayed content is the rotated size.
        val contentAspect = previewSize.height.toFloat() / previewSize.width
        val (shownW, shownH) = if (viewfinder.width() / viewfinder.height() > contentAspect) {
            viewfinder.width() to viewfinder.width() / contentAspect
        } else {
            viewfinder.height() * contentAspect to viewfinder.height()
        }
        cameraPreview.setTransform(Matrix().apply {
            setScale(shownW / viewW, shownH / viewH)
            postTranslate(viewfinder.centerX() - shownW / 2f, viewfinder.centerY() - shownH / 2f)
        })
    }

    /** Largest 16:9 preview size up to 1080p supported by the back camera, falling back to 1280x720. */
    private fun choosePreviewSize(): Size = runCatching {
        val manager = getSystemService(CameraManager::class.java)
        val id = manager.cameraIdList.first { manager.getCameraCharacteristics(it).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK }
        val map = manager.getCameraCharacteristics(id).get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        map.getOutputSizes(SurfaceTexture::class.java)
            .filter { it.width <= 1920 && it.height <= 1080 && it.width * 9 == it.height * 16 }
            .maxByOrNull { it.width * it.height }
    }.getOrNull() ?: Size(1280, 720)

    override fun onDestroy() { cameraManager.release(); pipeline.shutdown(); io.shutdown(); super.onDestroy() }
}
