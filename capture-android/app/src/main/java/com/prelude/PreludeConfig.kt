package com.prelude

/**
 * Deployment settings for the demo app. Edit these values for your setup.
 *
 * The Results Service runs on a team laptop over local Wi-Fi (spec 4.1), so the base URL is that laptop's LAN
 * address and Spring Boot port. The device id must match the id the calibration tool uploaded the profile under.
 */
object PreludeConfig {
    const val RESULTS_SERVICE_BASE_URL = "http://192.168.0.14:8080"
    const val DEVICE_ID = "pixel-8-pro"
    /** Evaluation run registered on the Results Service (POST /api/v1/runs); the demo shows its snapshot. */
    const val RUN_ID = "phase2-fusion-ablation"
}
