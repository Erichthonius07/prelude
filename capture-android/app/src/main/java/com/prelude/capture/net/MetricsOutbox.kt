package com.prelude.capture.net

import org.json.JSONObject
import java.io.File
import java.time.Instant
import java.util.UUID

/**
 * Local-first metrics queue (spec 4.1: "each stage caches results locally on-device first, then uploads in bulk").
 * One JSON submission per line. Live demo bursts are always tagged `scored: false` / `source: live_demo`, because
 * they have no tripod ground truth and must never mix with held-out test-set records.
 */
class MetricsOutbox(private val file: File) {

    @Synchronized
    fun record(deviceId: String, burstId: String, stage: String, metrics: JSONObject) {
        val submission = JSONObject()
            .put("submissionId", UUID.randomUUID().toString())
            .put("schemaVersion", SCHEMA_VERSION)
            .put("source", "live_demo")
            .put("scored", false)
            .put("deviceId", deviceId)
            .put("burstId", burstId)
            .put("stage", stage)
            .put("recordedAt", Instant.now().toString())
            .put("metrics", metrics)
        file.appendText(submission.toString() + "\n")
    }

    @Synchronized
    fun pending(): List<String> = if (file.exists()) file.readLines().filter { it.isNotBlank() } else emptyList()

    /** Drops the first [count] submissions after a successful upload; anything queued meanwhile is kept. */
    @Synchronized
    fun remove(count: Int) {
        val rest = pending().drop(count)
        if (rest.isEmpty()) file.delete() else file.writeText(rest.joinToString("\n", postfix = "\n"))
    }

    companion object { const val SCHEMA_VERSION = "0.1-draft" }
}
