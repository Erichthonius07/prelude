package com.prelude.capture.net

import com.prelude.pipeline.contract.PostProcessParams
import org.json.JSONArray
import org.json.JSONObject

/**
 * Scored comparison data for the demo, parsed from the Results Service snapshot
 * (GET /api/v1/runs/{runId}/snapshot, data-contract addendum §13.7). Everything here is computed on the fixed held-out test set by the batch
 * runner. The app only displays it and never computes or edits scores.
 */
class ComparisonReport(
    val schemaVersion: String,
    val generatedAt: String,
    val deviceId: String,
    val testSetN: Int,
    val variants: List<VariantResult>,
    val stageGains: List<StageGain>,
    /** Post-processing's standalone SSIM contribution, reported separately from the full-pipeline delta. */
    val postProcessSsimContribution: Double?,
    val learnedFusionTrainSsim: Double?,
    val learnedFusionValSsim: Double?,
    /** Wire id of the best-scoring fusion strategy; the live pipeline hands this one to the denoiser. */
    val bestFusionStrategy: String?,
    val postProcessParams: PostProcessParams?,
) {
    fun variant(id: String): VariantResult? = variants.firstOrNull { it.id == id }

    companion object {
        /** The seven evaluation variants, in the order the proposal defines them. */
        val VARIANT_IDS = listOf("single_raw_frame", "fusion_naive", "fusion_trimmed", "fusion_confidence_weighted", "fusion_learned", "full_pipeline", "single_frame_fallback")

        fun parse(json: String): ComparisonReport {
            val o = JSONObject(json)
            val bootstrap = o.optJSONArray("bootstrap")?.objects().orEmpty()
            val variants = o.getJSONArray("ablation").objects().map { VariantResult.parse(it, bootstrap) }
            return ComparisonReport(
                schemaVersion = o.getString("snapshotVersion"),
                generatedAt = o.getJSONObject("run").optString("createdAt"),
                deviceId = o.optString("deviceId"),
                testSetN = o.getJSONObject("run").getInt("plannedTestN"),
                variants = variants,
                stageGains = o.optJSONArray("stageGains")?.objects()?.map(StageGain::parse).orEmpty(),
                postProcessSsimContribution = o.optDoubleOrNull("postProcessSsimContribution"),
                learnedFusionTrainSsim = o.optJSONObject("learnedFusionDelta")?.optDoubleOrNull("trainSsim"),
                learnedFusionValSsim = o.optJSONObject("learnedFusionDelta")?.optDoubleOrNull("valSsim"),
                bestFusionStrategy = variants.filter { it.id.startsWith("fusion_") && it.ssim != null }.maxByOrNull { it.ssim!! }?.id
                    ?.let { mapOf("fusion_naive" to "naive_mean", "fusion_trimmed" to "trimmed_mean", "fusion_confidence_weighted" to "confidence_weighted", "fusion_learned" to "learned")[it] },
                postProcessParams = o.optJSONObject("postProcessParams")?.let { p ->
                    PostProcessParams(
                        toneGamma = p.getDouble("toneGamma").toFloat(),
                        contrast = p.getDouble("contrast").toFloat(),
                        sharpenAmount = p.getDouble("sharpenAmount").toFloat(),
                        sharpenRadius = p.getInt("sharpenRadius"),
                        source = p.optString("source", "results service"),
                    )
                },
            )
        }
    }
}

class VariantResult(
    val id: String,
    val label: String,
    val psnr: Double?,
    val ssim: Double?,
    val lpips: Double?,
    val p95LatencyMs: Long?,
    /** PSNR delta against the single-raw-frame baseline. */
    val deltaPsnr: Double?,
    val ci95: Pair<Double, Double>?,
    /** "baseline", "significant" or "directional" (CI too wide after correction). */
    val significance: String,
    /** Number of per-image scores behind this aggregate (should equal testSetN). */
    val perImageCount: Int,
) {
    companion object {
        /** [bootstrap] rows are a − b, so a "single_raw_frame__vs__<id>" row is flipped to read as <id> − baseline. */
        fun parse(o: JSONObject, bootstrap: List<JSONObject>): VariantResult {
            val id = o.getString("variant")
            val b = bootstrap.firstOrNull { it.getString("metric") == "psnr" && setOf(it.getString("variantA"), it.getString("variantB")) == setOf(id, "single_raw_frame") }
            val sign = if (b?.getString("variantA") == id) 1.0 else -1.0
            return VariantResult(
                id = id,
                label = id,
                psnr = o.optDoubleOrNull("meanPsnr"),
                ssim = o.optDoubleOrNull("meanSsim"),
                lpips = null,
                p95LatencyMs = if (o.has("p95LatencyMs") && !o.isNull("p95LatencyMs")) o.getLong("p95LatencyMs") else null,
                deltaPsnr = b?.let { sign * it.getDouble("meanDelta") },
                ci95 = b?.let { if (sign > 0) it.getDouble("ciLower") to it.getDouble("ciUpper") else -it.getDouble("ciUpper") to -it.getDouble("ciLower") },
                significance = if (id == "single_raw_frame") "baseline" else b?.getString("significanceStatus") ?: "directional",
                perImageCount = o.optInt("n"),
            )
        }
    }
}

class StageGain(val stage: String, val deltaPsnr: Double, val ci95: Pair<Double, Double>?) {
    companion object {
        fun parse(o: JSONObject) = StageGain(o.getString("stage"), o.getDouble("deltaPsnr"), o.optJSONArray("ci95")?.let { it.getDouble(0) to it.getDouble(1) })
    }
}

private fun JSONArray.objects(): List<JSONObject> = List(length()) { getJSONObject(it) }
private fun JSONObject.optDoubleOrNull(key: String): Double? = if (has(key) && !isNull(key)) getDouble(key) else null
