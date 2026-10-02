package com.prelude.resultsservice.read;

/** Exceptions specific to the read & query API, handled by {@link ReadApiExceptionHandler}. */
public abstract class ReadApiException extends RuntimeException {

    private final String code;

    protected ReadApiException(String code, String message) {
        super(message);
        this.code = code;
    }

    public String code() {
        return code;
    }

    public static class RunNotFound extends ReadApiException {
        public RunNotFound(String runId) {
            super("RUN_NOT_FOUND", "Run '" + runId + "' does not exist.");
        }
    }

    public static class InvalidReadRequest extends ReadApiException {
        public InvalidReadRequest(String code, String message) {
            super(code, message);
        }
    }

    public static class BootstrapNotComputable extends ReadApiException {
        public BootstrapNotComputable(String code, String message) {
            super(code, message);
        }
    }
}