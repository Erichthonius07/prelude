package com.prelude.resultsservice.api;

import java.util.UUID;

public abstract class PreludeApiException extends RuntimeException {

    private final String code;

    protected PreludeApiException(String code, String message) {
        super(message);
        this.code = code;
    }

    protected PreludeApiException(String code, String message, Throwable cause) {
        super(message, cause);
        this.code = code;
    }

    public String code() {
        return code;
    }

    public static class RunNotRegistered extends PreludeApiException {
        public RunNotRegistered(String runId) {
            super("RUN_NOT_REGISTERED", "Run '" + runId + "' is not registered. Register it first via "
                    + "POST /api/v1/runs (plannedTestN and plannedComparisons are fixed at registration).");
        }
    }

    public static class ContractViolation extends PreludeApiException {
        public ContractViolation(String code, String message) {
            super(code, message);
        }
    }

    public static class IdempotencyKeyReused extends PreludeApiException {
        public IdempotencyKeyReused(UUID submissionId) {
            super("IDEMPOTENCY_KEY_REUSED", "submissionId " + submissionId + " was already used for a different "
                    + "payload. A submissionId is an idempotency key: reuse it only when retrying the exact same "
                    + "submission; otherwise generate a fresh UUID.");
        }
    }

    /** Flush-time constraint violation; the exception advice classifies it (duplicate retry vs
     *  concurrent-supersede race) using a fresh read after rollback. */
    public static class MetricConflictSignal extends PreludeApiException {
        private final UUID submissionId;
        private final String payloadHash;

        public MetricConflictSignal(UUID submissionId, String payloadHash, Throwable cause) {
            super("CONFLICT_SIGNAL", "Database constraint violation during ingestion.", cause);
            this.submissionId = submissionId;
            this.payloadHash = payloadHash;
        }

        public UUID submissionId() {
            return submissionId;
        }

        public String payloadHash() {
            return payloadHash;
        }
    }

    public static class AdminAuthFailed extends PreludeApiException {
        public AdminAuthFailed(String message) {
            super("ADMIN_AUTH_FAILED", message);
        }
    }
}