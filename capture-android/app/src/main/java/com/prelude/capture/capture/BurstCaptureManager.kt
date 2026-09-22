package com.prelude.capture.capture

import android.content.Context
import android.graphics.ImageFormat
import android.hardware.camera2.*
import android.media.Image
import android.media.ImageReader
import android.os.Handler
import android.os.HandlerThread
import android.util.Size
import com.prelude.capture.model.YuvFrame
import kotlinx.coroutines.withTimeout
import kotlin.coroutines.resume
import kotlin.coroutines.resumeWithException
import kotlin.coroutines.suspendCancellableCoroutine

/**
 * Camera2 burst capture using real captureBurst() to minimize inter-frame gap
 * (critical for Role 2 alignment). AE/AWB/focus locked across the burst.
 * NOTE: statically reviewed only — not compiled/run (no Android SDK in this env).
 */
class BurstCaptureManager(private val context: Context) {

    private val cameraManager = context.getSystemService(Context.CAMERA_SERVICE) as CameraManager

    private lateinit var device: CameraCharacteristics
    private lateinit var cameraId: String
    private lateinit var imageReader: ImageReader
    private lateinit var session: CameraCaptureSession
    private lateinit var bgThread: HandlerThread
    private lateinit var bgHandler: Handler

    suspend fun open(): Size = suspendCancellableCoroutine { cont ->
        bgThread = HandlerThread("prelude-cam").also { it.start() }
        bgHandler = Handler(bgThread.looper)
        cameraId = cameraManager.cameraIdList.first { id ->
            cameraManager.getCameraCharacteristics(id).get(CameraCharacteristics.LENS_FACING) ==
                CameraCharacteristics.LENS_FACING_BACK
        }
        device = cameraManager.getCameraCharacteristics(cameraId)
        val map = device.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP)!!
        val size = map.getOutputSizes(ImageFormat.YUV_420_888)
            .minByOrNull { it.width * it.height } ?: Size(1280, 720)
        imageReader = ImageReader.newInstance(size.width, size.height, ImageFormat.YUV_420_888, numFramesCapacity())

        cameraManager.openCamera(cameraId, object : CameraDevice.StateCallback() {
            override fun onOpened(cam: CameraDevice) {
                cam.createCaptureSession(listOf(imageReader.surface),
                    object : CameraCaptureSession.StateCallback() {
                        override fun onConfigured(s: CameraCaptureSession) { session = s; cont.resume(size) }
                        override fun onConfigureFailed(s: CameraCaptureSession) =
                            cont.resumeWithException(IllegalStateException("session config failed"))
                    }, bgHandler)
            }
            override fun onDisconnected(cam: CameraDevice) =
                cont.resumeWithException(IllegalStateException("disconnected"))
            override fun onError(cam: CameraDevice, error: Int) =
                cont.resumeWithException(IllegalStateException("cam error $error"))
        }, bgHandler)
    }

    private fun numFramesCapacity() = 16 // ImageReader buffer: >= max burst (8) with headroom

    /** Lock AE/AWB/focus once, then fire a hardware burst. */
    suspend fun captureBurst(numFrames: Int): List<Pair<YuvFrame, CaptureResult>> {
        lock3A()
        val requests = List(numFrames) { buildLockedStillRequest() }
        val (rawImages, rawResults) = submitBurst(requests)

        val resultByTs = rawResults.filterNotNull()
            .associateBy { it.get(CaptureResult.SENSOR_TIMESTAMP) ?: Long.MIN_VALUE }
        val out = mutableListOf<Pair<YuvFrame, CaptureResult>>()
        for (img in rawImages) {
            val ts = img.timestamp
            val yuv = toYuvFrame(img)
            img.close()
            resultByTs[ts]?.let { out.add(yuv to it) }
        }
        return out
    }

    private suspend fun lock3A() = withTimeout(6000) {
        // Fire AF trigger once.
        val trigger = session.device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(imageReader.surface)
            set(CaptureRequest.CONTROL_MODE, CameraMetadata.CONTROL_MODE_AUTO)
            set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
            set(CaptureRequest.CONTROL_AF_TRIGGER, CameraMetadata.CONTROL_AF_TRIGGER_START)
        }.build()
        session.capture(trigger, null, bgHandler)

        // Repeat until focus locked and exposure converged, then stop.
        val observe = session.device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(imageReader.surface)
            set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
        }.build()
        suspendCancellableCoroutine<Unit> { c ->
            val cb = object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(s: CameraCaptureSession, r: CaptureRequest, m: TotalCaptureResult) {
                    val af = m.get(CaptureResult.CONTROL_AF_STATE)
                    val ae = m.get(CaptureResult.CONTROL_AE_STATE)
                    val focused = af == CameraMetadata.CONTROL_AF_STATE_FOCUSED_LOCKED ||
                                  af == CameraMetadata.CONTROL_AF_STATE_NOT_FOCUSED_LOCKED
                    val exposed = ae == CameraMetadata.CONTROL_AE_STATE_CONVERGED ||
                                  ae == CameraMetadata.CONTROL_AE_STATE_FLASH_REQUIRED ||
                                  ae == CameraMetadata.CONTROL_AE_STATE_LOCKED
                    if (focused && exposed && !c.isCompleted) c.resume(Unit)
                }
            }
            session.setRepeatingRequest(observe, cb, bgHandler)
        }
        session.stopRepeating()
    }

    private fun buildLockedStillRequest(): CaptureRequest =
        session.device.createCaptureRequest(CameraDevice.TEMPLATE_STILL_CAPTURE).apply {
            addTarget(imageReader.surface)
            set(CaptureRequest.CONTROL_AE_LOCK, true)
            set(CaptureRequest.CONTROL_AWB_LOCK, true)
            set(CaptureRequest.CONTROL_AF_MODE, CameraMetadata.CONTROL_AF_MODE_AUTO)
        }.build()

    private suspend fun submitBurst(requests: List<CaptureRequest>): Pair<List<Image>, List<CaptureResult?>> {
        val n = requests.size
        val lock = Any()
        val images = mutableListOf<Image>()
        val results = mutableListOf<CaptureResult?>()

        return suspendCancellableCoroutine { cont ->
            fun checkDone() {
                val i: List<Image>; val r: List<CaptureResult?>
                synchronized(lock) {
                    if (cont.isCompleted || images.size < n || results.size < n) return
                    i = images.toList(); r = results.toList()
                }
                cont.resume(i to r)
            }
            imageReader.setOnImageAvailableListener({ reader ->
                val img = reader.acquireNextImage() ?: return@setOnImageAvailableListener
                synchronized(lock) { images.add(img) }
                checkDone()
            }, bgHandler)

            val cb = object : CameraCaptureSession.CaptureCallback() {
                override fun onCaptureCompleted(s: CameraCaptureSession, r: CaptureRequest, m: TotalCaptureResult) {
                    synchronized(lock) { results.add(m) }; checkDone()
                }
                override fun onCaptureFailed(s: CameraCaptureSession, r: CaptureRequest, m: CaptureFailure) {
                    synchronized(lock) { results.add(null) }; checkDone() // avoid deadlock; frame dropped on join
                }
            }
            try {
                session.captureBurst(requests, cb, bgHandler)
            } catch (e: CameraAccessException) {
                if (!cont.isCompleted) cont.resumeWithException(e)
            }
            cont.invokeOnCancellation {
                synchronized(lock) { images.forEach { runCatching { it.close() } } }
            }
        }
    }

    private fun toYuvFrame(image: Image): YuvFrame {
        val w = image.width; val h = image.height
        fun copy(plane: Image.Plane): ByteArray {
            val buf = plane.buffer
            val arr = ByteArray(buf.remaining())
            buf.get(arr); return arr
        }
        return YuvFrame(w, h, copy(image.planes[0]), copy(image.planes[1]), copy(image.planes[2]))
    }

    fun metadata(result: CaptureResult): Triple<Int, Long, Long> {
        val iso = result.get(CaptureResult.SENSOR_SENSITIVITY) ?: 0
        val exp = result.get(CaptureResult.SENSOR_EXPOSURE_TIME) ?: 0L
        val ts = result.get(CaptureResult.SENSOR_TIMESTAMP) ?: 0L
        return Triple(iso, exp, ts)
    }

    fun close() {
        runCatching { session.stopRepeating() }
        runCatching { session.close() }
        runCatching { imageReader.close() }
        runCatching { bgThread.quitSafely() }
    }
}