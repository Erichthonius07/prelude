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

    fun load(): DeviceThresholds {
        val blurRaw = prefs.getString(KEY_BLUR, null) ?: return DeviceThresholds()
        val o = JSONObject(blurRaw)
        val edges = (0 until o.getJSONArray("edges").length()).map { o.getJSONArray("edges").getDouble(it).toFloat() }
        val floors = (0 until o.getJSONArray("floors").length()).map { o.getJSONArray("floors").getDouble(it).toFloat() }
        return DeviceThresholds(blurTexture = BlurTextureThreshold(edges, floors))
    }

    // Runtime-editable Results Service base URL (debug screen).
    fun saveBaseUrl(url: String) = prefs.edit().putString(KEY_BASE_URL, url.trim()).apply()
    fun loadBaseUrl(): String = prefs.getString(KEY_BASE_URL, DEFAULT_BASE_URL) ?: DEFAULT_BASE_URL

    // TODO(data-contract addendum): spec wants the device profile auto-uploaded to the Results
    // Service, but the contract has no calibration-profile submission type yet. Local-only for now.

    private companion object {
        const val KEY_BLUR = "blur_texture"
        const val KEY_BASE_URL = "base_url"
        const val DEFAULT_BASE_URL = "http://10.0.0.2:8080"
    }
}