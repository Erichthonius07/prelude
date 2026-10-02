package com.prelude.capture.net

import com.prelude.capture.model.CapturedFrame
import com.prelude.capture.model.FrameBurst
import com.prelude.capture.model.YuvFrame
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Assume.assumeTrue
import org.junit.Test
import java.time.Instant
import java.util.UUID

/**
 * Validates the wire payload against docs/data-contract.md §5.1 (capture stage schema)
 * and, when PRELUDE_LIVE_URL points at a running Results Service, round-trips a real
 * envelope through run registration + POST /api/v1/metrics/capture.
 */
class ResultsServiceClientTest {

    private val client = ResultsServiceClient("http://localhost:8080")
    private val JSON = "application/json; charset=utf-8".toMediaType()

    private fun sampleBurst() = FrameBurst(
        burstId = "it-device-1:1000",
        frames = listOf(
            CapturedFrame(0, YuvFrame(4, 4, ByteArray(16), ByteArray(8), ByteArray(8)),
                sensorTimestampNs = 100L, iso = 3200, exposureTimeNs = 83_000_000L,
                sharpnessScore = 0.42f, blurRejected = false),
            CapturedFrame(1, YuvFrame(4, 4, ByteArray(16), ByteArray(8), ByteArray(8)),
                sensorTimestampNs = 200L, iso = 3200, exposureTimeNs = 83_000_000L,
                sharpnessScore = 0.11f, blurRejected = true),
        ),
        emergencyFallbackActive = false,
    )

    @Test
    fun `envelope matches capture stage contract`() {
        val env = client.buildEnvelope("run-x", "dev-1", "0.1.0", listOf(sampleBurst()))

        // Envelope fields (server IngestionEnvelope DTO)
        UUID.fromString(env.getString("submissionId"))
        assertEquals("run-x", env.getString("runId"))
        assertEquals("dev-1", env.getString("deviceId"))
        assertEquals("0.1.0", env.getString("appVersion"))
        Instant.parse(env.getString("occurredAt"))

        val burst = env.getJSONObject("payload").getJSONArray("bursts").getJSONObject(0)
        assertEquals("it-device-1:1000", burst.getString("imageId"))
        val frames = burst.getJSONArray("frames")
        assertEquals(2, frames.length())

        for (i in 0 until frames.length()) {
            val f = frames.getJSONObject(i)
            // Field names and types per §5.1 — all required, per-frame rows.
            assertTrue(f.getInt("frameIndex") >= 0)
            assertTrue(f.has("iso") && f.get("iso") is Int)
            assertTrue(f.has("exposureTimeNs") && f.get("exposureTimeNs") is Long)
            assertTrue(f.has("capturedAtEpochMs") && f.get("capturedAtEpochMs") is Long)
            assertTrue(f.has("sharpnessScore") && f.get("sharpnessScore") is Double)
            assertTrue(f.get("blurRejected") is Boolean)
            assertTrue(f.get("emergencyFallback") is Boolean)
        }
        assertTrue(frames.getJSONObject(1).getBoolean("blurRejected"))
        assertTrue(frames.getJSONObject(0).getBoolean("blurRejected").not())
    }

    @Test
    fun `live round trip registers run and accepts capture payload`() {
        val base = System.getenv("PRELUDE_LIVE_URL")
        assumeTrue("PRELUDE_LIVE_URL not set; skipping live service round-trip", base != null)
        val http = OkHttpClient()

        val runId = "it-capture-" + UUID.randomUUID()
        val registration = JSONObject()
            .put("runId", runId)
            .put("description", "automated capture-contract round-trip")
            .put("plannedTestN", 2)
            .put("plannedComparisons", 1)
            .put("createdBy", "ResultsServiceClientTest")

        http.newCall(
            Request.Builder().url("$base/api/v1/runs")
                .post(registration.toString().toRequestBody(JSON))
                .build()
        ).execute().use { resp ->
            val body = resp.body?.string()
            assertTrue("run registration failed: ${resp.code} $body", resp.isSuccessful)
            println("REGISTER run=$runId -> HTTP ${resp.code} $body")
        }

        // The exact JSON the serializer produces for a sample burst.
        val envelope = client.buildEnvelope(runId, "it-device-1", "0.1.0", listOf(sampleBurst()))
        http.newCall(
            Request.Builder().url("$base/api/v1/metrics/capture")
                .post(envelope.toString().toRequestBody(JSON))
                .build()
        ).execute().use { resp ->
            val body = resp.body?.string()
            println("CAPTURE -> HTTP ${resp.code} $body")
            assertTrue("capture submission failed: ${resp.code} $body", resp.isSuccessful)
            val out = JSONObject(body!!)
            assertEquals(1, out.getInt("acceptedImageCount"))
            assertEquals(false, out.getBoolean("duplicate"))
        }
    }
}
