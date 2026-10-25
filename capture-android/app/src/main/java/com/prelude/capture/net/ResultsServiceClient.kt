package com.prelude.capture.net

import java.io.File
import java.net.HttpURLConnection
import java.net.URL
import java.util.UUID

/**
 * Results Service client for the demo app (Role 3 side of Role 1's REST API, docs/data-contract.md §11-13).
 *
 *  - GET  /                                        reachability check, {"status":"up"}
 *  - GET  /api/v1/runs/{runId}/snapshot            comparison report, fetched when the app opens or the user taps sync
 *  - POST /api/v1/metrics/batch                    bulk upload of locally queued live-burst metrics (idempotent)
 *
 * Request-triggered only, never polling. The last good report is cached on disk so the demo still shows
 * before/after numbers when the laptop or Wi-Fi is unreachable (spec 3.4).
 */
class ResultsServiceClient(
    private val baseUrl: String = "http://192.168.0.14:8080",
    private val deviceId: String = "",
    private val storageDir: File? = null,
    private val runId: String = "",
) {
    private val cacheFile: File? get() = storageDir?.let { File(it, "comparison_cache.json") }
    private val cacheMetaFile: File? get() = storageDir?.let { File(it, "comparison_cache.meta") }

    fun refresh(): SyncResult = runCatching {
        val connection = open("/", "GET")
        connection.connect()
        val code = connection.responseCode
        connection.disconnect()
        if (code in 200..299) SyncResult.Connected else SyncResult.Offline("service returned $code")
    }.getOrElse { SyncResult.Offline(it.message ?: "connection unavailable") }

    /** Network first; falls back to the on-disk cache. Call off the main thread. */
    fun loadComparison(): ComparisonLoad {
        val fetched = runCatching {
            val connection = open("/api/v1/runs/$runId/snapshot", "GET")
            try {
                val code = connection.responseCode
                if (code !in 200..299) error("service returned $code")
                connection.inputStream.bufferedReader().use { it.readText() }
            } finally { connection.disconnect() }
        }
        fetched.onSuccess { body ->
            // Parse before caching, so a malformed response never replaces a good cache.
            val report = runCatching { ComparisonReport.parse(body) }.getOrElse { return cachedComparison("bad response: ${it.message}") }
            val now = System.currentTimeMillis()
            runCatching { cacheFile?.writeText(body); cacheMetaFile?.writeText(now.toString()) }
            return ComparisonLoad.Fresh(report, now)
        }
        return cachedComparison(fetched.exceptionOrNull()?.message ?: "connection unavailable")
    }

    /** Cache only, no network; used for the instant first paint. */
    fun cachedComparison(reason: String = "not synced yet"): ComparisonLoad {
        val body = runCatching { cacheFile?.takeIf { it.exists() }?.readText() }.getOrNull() ?: return ComparisonLoad.Unavailable(reason)
        val fetchedAt = runCatching { cacheMetaFile?.readText()?.trim()?.toLong() }.getOrNull() ?: 0L
        return runCatching { ComparisonLoad.Cached(ComparisonReport.parse(body), fetchedAt, reason) }.getOrElse { ComparisonLoad.Unavailable("cache unreadable") }
    }

    /**
     * Uploads everything queued in [outbox] as one batch. Each submission carries its own submissionId, and the
     * batch an Idempotency-Key, so a retry after a dropped response can't double-count. Returns the number sent.
     */
    fun uploadOutbox(outbox: MetricsOutbox): Int {
        val pending = outbox.pending()
        if (pending.isEmpty()) return 0
        val body = "{\"schemaVersion\":\"${MetricsOutbox.SCHEMA_VERSION}\",\"deviceId\":\"$deviceId\",\"submissions\":[${pending.joinToString(",")}]}"
        val ok = runCatching {
            val connection = open("/api/v1/metrics/batch", "POST").apply {
                doOutput = true
                setRequestProperty("Content-Type", "application/json")
                setRequestProperty("Idempotency-Key", UUID.nameUUIDFromBytes(body.toByteArray()).toString())
            }
            try {
                connection.outputStream.use { it.write(body.toByteArray()) }
                connection.responseCode in 200..299
            } finally { connection.disconnect() }
        }.getOrDefault(false)
        if (ok) outbox.remove(pending.size)
        return if (ok) pending.size else 0
    }

    private fun open(path: String, method: String): HttpURLConnection =
        (URL(baseUrl.trimEnd('/') + path).openConnection() as HttpURLConnection).apply {
            requestMethod = method
            connectTimeout = 1_500
            readTimeout = 3_000
            setRequestProperty("Accept", "application/json")
        }
}

sealed interface SyncResult {
    data object Connected : SyncResult
    data class Offline(val reason: String) : SyncResult
}

sealed interface ComparisonLoad {
    data class Fresh(val report: ComparisonReport, val fetchedAt: Long) : ComparisonLoad
    data class Cached(val report: ComparisonReport, val fetchedAt: Long, val reason: String) : ComparisonLoad
    data class Unavailable(val reason: String) : ComparisonLoad

    val reportOrNull: ComparisonReport? get() = when (this) { is Fresh -> report; is Cached -> report; is Unavailable -> null }
}
