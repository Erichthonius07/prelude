package com.prelude.capture.capture

import android.annotation.SuppressLint
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import android.hardware.camera2.CameraCaptureSession
import android.hardware.camera2.CameraCharacteristics
import android.hardware.camera2.CameraDevice
import android.hardware.camera2.CameraManager
import android.hardware.camera2.CaptureRequest
import android.hardware.camera2.TotalCaptureResult
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.os.Looper
import android.util.Size
import android.view.Surface
import com.prelude.capture.calibration.BlurTextureThresholdCalculator
import com.prelude.capture.calibration.DeviceThresholds
import com.prelude.capture.model.FrameBurst
import com.prelude.capture.model.FrameSample
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Camera2 preview + locked-exposure burst capture with texture-aware blur rejection (CaptureModule → FrameBurst).
 *
 * Burst sequence: lock AE and AWB on the preview stream and trigger an AF lock, let it settle, fire N still
 * requests with the locks held, decode each JPEG to an upright bitmap, score sharpness against the threshold for
 * that frame's texture bucket, then release the locks. If every frame is rejected, the sharpest is kept anyway
 * (emergency fallback). All callbacks are delivered on the main thread.
 */
class BurstCaptureManager(
    context: Context,
    private val thresholds: DeviceThresholds,
    private val listener: Listener,
) {
    interface Listener {
        fun onFrameCaptured(sample: FrameSample, expected: Int)
        fun onBurstReady(burst: FrameBurst)
        fun onCaptureError(error: Throwable)
    }

    private val cameraManager = context.getSystemService(CameraManager::class.java)
    private val worker = HandlerThread("prelude-camera").apply { start() }
    private val handler = Handler(worker.looper)
    private val main = Handler(Looper.getMainLooper())
    private val capturing = AtomicBoolean(false)
    private val sharpness = SharpnessEstimator()
    private val textures = TextureBucketEstimator(thresholds)
    private val blurThresholds = BlurTextureThresholdCalculator(thresholds)

    private var camera: CameraDevice? = null
    private var session: CameraCaptureSession? = null
    private var reader: ImageReader? = null
    private var previewSurface: Surface? = null
    private var sensorOrientation = 90
    private var requestedFrames = 6

    // Per-burst state, touched only on the camera thread.
    private var expected = 0
    private val samples = mutableListOf<FrameSample>()
    private val results = HashMap<Long, TotalCaptureResult>()
    private var watchdog: Runnable? = null

    @SuppressLint("MissingPermission")
    fun startPreview(surface: Surface) {
        runCatching {
            val id = cameraManager.cameraIdList.first { id ->
                cameraManager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_BACK
            }
            sensorOrientation = cameraManager.getCameraCharacteristics(id).get(CameraCharacteristics.SENSOR_ORIENTATION) ?: 90
            previewSurface = surface
            cameraManager.openCamera(id, object : CameraDevice.StateCallback() {
                override fun onOpened(device: CameraDevice) { camera = device; createSession(id, surface) }
                override fun onDisconnected(device: CameraDevice) { device.close(); camera = null }
                override fun onError(device: CameraDevice, error: Int) { device.close(); camera = null; fail(IllegalStateException("Camera2 error $error")) }
            }, handler)
        }.onFailure(::fail)
    }

    @Suppress("DEPRECATION")
    private fun createSession(cameraId: String, preview: Surface) {
        val device = camera ?: return
        val size = chooseStillSize(cameraId)
        reader = ImageReader.newInstance(size.width, size.height, ImageFormat.JPEG, MAX_FRAMES).also { imageReader ->
            imageReader.setOnImageAvailableListener({ source -> onImage(source) }, handler)
        }
        device.createCaptureSession(listOf(preview, reader!!.surface), object : CameraCaptureSession.StateCallback() {
            override fun onConfigured(cameraSession: CameraCaptureSession) {
                session = cameraSession
                runCatching { cameraSession.setRepeatingRequest(previewRequest(locked = false).build(), null, handler) }.onFailure(::fail)
            }
            override fun onConfigureFailed(session: CameraCaptureSession) = fail(IllegalStateException("Capture session failed"))
        }, handler)
    }

    fun setBurstLength(count: Int) { requestedFrames = count.coerceIn(4, MAX_FRAMES) }

    /** Starts a burst of [count] frames (4–8). Ignored while a burst is already running. */
    fun captureBurst(count: Int = requestedFrames) {
        if (!capturing.compareAndSet(false, true)) return
        handler.post {
            val cameraSession = session
            if (cameraSession == null || camera == null) { capturing.set(false); fail(IllegalStateException("camera not ready")); return@post }
            expected = count.coerceIn(4, MAX_FRAMES)
            samples.clear(); results.clear()
            runCatching {
                // 1. Lock AE/AWB on the live stream and trigger an AF lock.
                cameraSession.setRepeatingRequest(previewRequest(locked = true).build(), null, handler)
                cameraSession.capture(previewRequest(locked = true).apply { set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_START) }.build(), null, handler)
                // 2. Give the locks a moment to settle, then fire the burst with the same locks held.
                handler.postDelayed({ fireBurst() }, LOCK_SETTLE_MS)
            }.onFailure { finishBurst(it) }
        }
    }

    private fun fireBurst() {
        val device = camera ?: return finishBurst(IllegalStateException("camera closed"))
        val cameraSession = session ?: return finishBurst(IllegalStateException("session closed"))
        val target = reader?.surface ?: return finishBurst(IllegalStateException("reader closed"))
        runCatching {
            val requests = List(expected) {
                device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
                    addTarget(target)
                    set(CaptureRequest.CONTROL_AE_LOCK, true)
                    set(CaptureRequest.CONTROL_AWB_LOCK, true)
                    set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
                    set(CaptureRequest.JPEG_ORIENTATION, 0) // rotated in code, so decoding never depends on EXIF
                    set(CaptureRequest.JPEG_QUALITY, 95.toByte())
                }.build()
            }
            cameraSession.captureBurst(requests, object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(session: CameraCaptureSession, request: CaptureRequest, result: TotalCaptureResult) {
                    result.get(android.hardware.camera2.CaptureResult.SENSOR_TIMESTAMP)?.let { results[it] = result }
                }
            }, handler)
            watchdog = Runnable { finishBurst(if (samples.isEmpty()) IllegalStateException("burst timed out with no frames") else null) }.also { handler.postDelayed(it, BURST_TIMEOUT_MS) }
        }.onFailure { finishBurst(it) }
    }

    private fun onImage(source: ImageReader) {
        val image = source.acquireNextImage() ?: return
        if (!capturing.get() || samples.size >= expected) { image.close(); return }
        val timestamp = image.timestamp
        val bytes = image.planes[0].buffer.let { buf -> ByteArray(buf.remaining()).also { buf.get(it) } }
        image.close()
        val bitmap = decodeUpright(bytes) ?: return
        val bucket = textures.bucket(bitmap)
        val threshold = blurThresholds.blurThreshold(bucket)
        val score = sharpness.varianceOfLaplacian(bitmap)
        val sample = FrameSample(samples.size, score, score >= threshold, bitmap, timestampNs = timestamp, textureBucket = bucket, blurThreshold = threshold)
        samples += sample
        main.post { listener.onFrameCaptured(sample, expected) }
        if (samples.size == expected) finishBurst(null)
    }

    private fun finishBurst(error: Throwable?) {
        watchdog?.let(handler::removeCallbacks); watchdog = null
        // Release AE/AWB/AF locks and go back to the free-running preview.
        runCatching {
            session?.capture(previewRequest(locked = false).apply { set(CaptureRequest.CONTROL_AF_TRIGGER, CaptureRequest.CONTROL_AF_TRIGGER_CANCEL) }.build(), null, handler)
            session?.setRepeatingRequest(previewRequest(locked = false).build(), null, handler)
        }
        if (error != null || samples.isEmpty()) {
            capturing.set(false)
            fail(error ?: IllegalStateException("no frames captured"))
            return
        }
        // Attach exposure metadata, then apply the emergency fallback if every frame was rejected.
        var frames = samples.map { s ->
            val r = s.timestampNs?.let(results::get)
            s.copy(exposureTimeNs = r?.get(android.hardware.camera2.CaptureResult.SENSOR_EXPOSURE_TIME), iso = r?.get(android.hardware.camera2.CaptureResult.SENSOR_SENSITIVITY))
        }
        if (frames.none { it.accepted }) {
            val sharpest = frames.maxBy { it.varianceOfLaplacian }.index
            frames = frames.map { if (it.index == sharpest) it.copy(accepted = true) else it }
        }
        val burst = FrameBurst(expected, frames)
        capturing.set(false)
        main.post { listener.onBurstReady(burst) }
    }

    private fun previewRequest(locked: Boolean): CaptureRequest.Builder {
        val device = camera ?: throw IllegalStateException("camera closed")
        return device.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW).apply {
            previewSurface?.let(::addTarget)
            set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE)
            set(CaptureRequest.CONTROL_AE_LOCK, locked)
            set(CaptureRequest.CONTROL_AWB_LOCK, locked)
        }
    }

    /** Largest JPEG size up to ~2 MP, so a burst of 8 decoded frames stays well within memory. */
    private fun chooseStillSize(cameraId: String): Size {
        val map = cameraManager.getCameraCharacteristics(cameraId).get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)
        return map?.getOutputSizes(ImageFormat.JPEG)?.filter { it.width * it.height <= 1920 * 1080 }?.maxByOrNull { it.width * it.height } ?: Size(1280, 720)
    }

    private fun decodeUpright(bytes: ByteArray): Bitmap? {
        val decoded = BitmapFactory.decodeByteArray(bytes, 0, bytes.size) ?: return null
        if (sensorOrientation % 360 == 0) return decoded
        val rotated = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, Matrix().apply { postRotate(sensorOrientation.toFloat()) }, true)
        if (rotated !== decoded) decoded.recycle()
        return rotated
    }

    private fun fail(error: Throwable) { main.post { listener.onCaptureError(error) } }

    /** Closes the camera but keeps the worker thread, so [startPreview] works again when the surface comes back. */
    fun closeCamera() {
        handler.post {
            runCatching { session?.close(); camera?.close(); reader?.close() }
            session = null; camera = null; reader = null
            capturing.set(false)
        }
    }

    /** Final teardown; the instance can't be reused afterwards. */
    fun release() { closeCamera(); worker.quitSafely() }

    companion object {
        const val MAX_FRAMES = 8
        private const val LOCK_SETTLE_MS = 350L
        private const val BURST_TIMEOUT_MS = 6000L
    }
}
