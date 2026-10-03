package com.prelude.denoise.timeout;

/**
 * Injectable error listener to avoid android.util.Log inside business logic.
 */
public interface ErrorListener {
    void onError(String message, Throwable t);
}
