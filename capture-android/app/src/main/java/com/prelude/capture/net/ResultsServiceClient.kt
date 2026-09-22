package com.prelude.capture.net

import com.prelude.capture.model.FrameBurst
import okhttp3.*
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONObject
import java.time.Instant
import java.util.UUID

class ResultsServiceClient(private val baseUrl: String) {
    private val http = OkHttpClient()
    private val JSON = "application/json; charset=utf-8".toMediaType()

    /** POST /api/v1/metrics/capture — real endpoint, not mocked. */
    fun submitCapture(runId: String, deviceId: String, appVersion: String, bursts: List<FrameBurst>): Boolean {
        val payload = JSONObject().put("bursts", JSONArray().apply {
            bursts.forEach { b ->
                put(JSONObject()
                    .put("imageId", b.burstId)
                    .put("frames", JSONArray().apply {
                        b.frames.forEach { f ->
                            put(JSONObject()
                                .put("frameIndex", f.frameIndex)
                                .put("iso", f.iso)
                                .put("exposureTimeNs", f.exposureTimeNs)
                                .put("capturedAtEpochMs", System.currentTimeMillis())
                                .put("sharpnessScore", f.sharpnessScore.toDouble())
                                .put("blurRejected", f.blurRejected)
                                .put("emergencyFallback", b.emergencyFallbackActive))
                        }
                    }))
            }
        })
        val envelope = JSONObject()
            .put("submissionId", UUID.randomUUID().toString())
            .put("runId", runId)
            .put("deviceId", deviceId)
            .put("appVersion", appVersion)
            .put("occurredAt", Instant.now().toString())
            .put("payload", payload)

        val req = Request.Builder()
            .url("$baseUrl/api/v1/metrics/capture")
            .post(envelope.toString().toRequestBody(JSON))
            .build()
        return runCatching {
            http.newCall(req).execute().use { it.isSuccessful }
        }.getOrElse { false }
    }
}