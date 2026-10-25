package com.prelude.capture.net

import org.junit.Assert.assertTrue
import org.junit.Test

class ResultsServiceClientTest {
    @Test fun unreachableServiceReturnsOfflineState() {
        val result = ResultsServiceClient("http://127.0.0.1:1").refresh()
        assertTrue(result is SyncResult.Offline)
    }
}
