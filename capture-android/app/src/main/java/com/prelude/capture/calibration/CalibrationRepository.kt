package com.prelude.capture.calibration

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/** Persists calibrated thresholds + runtime settings in SharedPreferences. */
class CalibrationRepository(context: Context) {
    private val prefs = context.applicationContext
        .getSharedPreferences("prelude_calibration", Context.MODE_PRIVATE)

    fun saveBlurTexture(t: BlurTextureThreshold) {
        val o = JSONObject()
            .put("edges", JSONArray(t.bucketEdges))
            .put("floors", JSONArray(t.bucketSharpnessMin))
        prefs.edit().putString(KEY_BLUR, o.toString()).apply()
    }

    // NEW: Persistence path for the alignment confidence floor.
    // Called by the AlignmentConfidenceFloorCalculator once Role 2 unblocks it.
    fun saveAlignmentConfidenceFloor(floor: Float) {
        prefs.edit().putFloat(KEY_ALIGNMENT_FLOOR, floor).apply()
    }

    // UPDATED: Now loads both the blur threshold and the alignment floor.
    fun load(): DeviceThresholds {
        val blurRaw = prefs.getString(KEY_BLUR, null)
        val blur = if (blurRaw != null) {
            val o = JSONObject(blurRaw)
            val edges = (0 until o.getJSONArray("edges").length()).map { o.getJSONArray("edges").getDouble(it).toFloat() }
            val floors = (0 until o.getJSONArray("floors").length()).map { o.getJSONArray("floors").getDouble(it).toFloat() }
            BlurTextureThreshold(edges, floors)
        } else null

        val alignmentFloor = if (prefs.contains(KEY_ALIGNMENT_FLOOR)) {
            prefs.getFloat(KEY_ALIGNMENT_FLOOR, 0f)
        } else null

        return DeviceThresholds(
            blurTexture = blur,
            alignmentConfidenceFloor = alignmentFloor
        )
    }

    // Runtime-editable Results Service base URL (debug screen).
    fun saveBaseUrl(url: String) = prefs.edit().putString(KEY_BASE_URL, url.trim()).apply()
    fun loadBaseUrl(): String = prefs.getString(KEY_BASE_URL, DEFAULT_BASE_URL) ?: DEFAULT_BASE_URL

    // Device-profile upload was the old spec idea, but the contract has no calibration-profile
    // submission type; the orchestrator resolved the floor's delivery path as local-only (no
    // server endpoint), so this repository is the single delivery mechanism.


    private companion object {
        const val KEY_BLUR = "blur_texture"
        const val KEY_ALIGNMENT_FLOOR = "alignment_confidence_floor" // NEW
        const val KEY_BASE_URL = "base_url"
        const val DEFAULT_BASE_URL = "http://10.0.0.2:8080"
    }
}