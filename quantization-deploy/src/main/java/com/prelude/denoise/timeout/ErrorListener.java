package com.prelude.denoise.timeout;

/**
 * Injectable error/info listener to avoid framework logging inside business logic.
 */
public interface ErrorListener {
    void onError(String message, Throwable t);

    /** For informational messages (e.g. XNNPACK requested, thread count). Not an error. */
    default void onInfo(String message) {
        // default no-op so existing implementations don't break
    }
}
