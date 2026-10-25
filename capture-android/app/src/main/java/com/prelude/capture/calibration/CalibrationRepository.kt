package com.prelude.capture.calibration

class CalibrationRepository {
    @Volatile private var current = DeviceThresholds()

    fun load(): DeviceThresholds = current

    fun save(profile: DeviceThresholds) {
        current = profile
    }

    fun runCalibration(): DeviceThresholds {
        val calibrated = current.copy(calibratedAt = java.time.LocalDate.now().toString())
        current = calibrated
        return calibrated
    }
}
