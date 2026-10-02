package com.prelude.denoise.api;

/**
 * Hook for reading the device's thermal state before/during calibration.
 * Role 1 provides the implementation (e.g. mapping to Android's PowerManager).
 */
public interface ThermalStateProvider {
    int getCurrentThermalStatus();
}
