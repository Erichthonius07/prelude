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
        val req = Request.Builder()
            .url("$baseUrl/api/v1/metrics/capture")
            .post(buildEnvelope(runId, deviceId, appVersion, bursts).toString().toRequestBody(JSON))
            .build()
        return runCatching {
            http.newCall(req).execute().use { it.isSuccessful }
        }.getOrElse { false }
    }

    /**
     * Wire shape per docs/data-contract.md §5.1 (capture stage) and the IngestionEnvelope DTO:
     * bursts[].imageId carries the FrameBurst's burstId (§2.3); every frame row is per-frame
     * (aggregate-only submissions are rejected, §2.2). Internal so unit tests can validate
     * the exact payload against the contract.
     */
    internal fun buildEnvelope(
        runId: String,
        deviceId: String,
        appVersion: String,
        bursts: List<FrameBurst>,
    ): JSONObject {
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
        return JSONObject()
            .put("submissionId", UUID.randomUUID().toString())
            .put("runId", runId)
            .put("deviceId", deviceId)
            .put("appVersion", appVersion)
            .put("occurredAt", Instant.now().toString())
            .put("payload", payload)
    }
}
