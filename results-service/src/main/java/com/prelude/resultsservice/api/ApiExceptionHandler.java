package com.prelude.resultsservice.api;

import com.prelude.resultsservice.dto.IngestionResponse;
import com.prelude.resultsservice.entity.SubmissionEntity;
import com.prelude.resultsservice.repository.SubmissionRepository;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.converter.HttpMessageNotReadableException;
import org.springframework.web.bind.MethodArgumentNotValidException;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

import java.util.List;
import java.util.Map;

@RestControllerAdvice
public class ApiExceptionHandler {

    private static final Logger log = LoggerFactory.getLogger(ApiExceptionHandler.class);

    public record ApiError(String code, String message, Object details) {}

    private final SubmissionRepository submissions;

    public ApiExceptionHandler(SubmissionRepository submissions) {
        this.submissions = submissions;
    }

    @ExceptionHandler(PreludeApiException.MetricConflictSignal.class)
    public ResponseEntity<?> conflictSignal(PreludeApiException.MetricConflictSignal e) {
        SubmissionEntity existing = submissions.findById(e.submissionId()).orElse(null);
        if (existing != null) {
            if (existing.getPayloadHash().equals(e.payloadHash())) {
                // Concurrent retry of the exact same submission: idempotent replay.
                return ResponseEntity.ok(new IngestionResponse(existing.getSubmissionId(), existing.getRunId(),
                        existing.getStage(), true, existing.getImageCount(), List.of()));
            }
            return conflict("IDEMPOTENCY_KEY_REUSED", "submissionId " + e.submissionId()
                    + " was already used for a different payload. Generate a fresh submissionId.");
        }
        // Versions are server-assigned, so the only remaining cause is two submissions racing on
        // the same grouping key. The loser's version was never consumed; retrying is safe.
        return conflict("CONCURRENT_SUPERSEDE",
                "Another submission superseded this grouping key at the same moment. Supersede versions are "
                        + "assigned server-side; simply retry this submission to receive the next version. Cause: "
                        + rootMessage(e));
    }

    @ExceptionHandler(PreludeApiException.IdempotencyKeyReused.class)
    public ResponseEntity<?> keyReused(PreludeApiException.IdempotencyKeyReused e) {
        return conflict("IDEMPOTENCY_KEY_REUSED", e.getMessage());
    }

    @ExceptionHandler(PreludeApiException.RunNotRegistered.class)
    public ResponseEntity<?> runNotRegistered(PreludeApiException.RunNotRegistered e) {
        return ResponseEntity.badRequest().body(new ApiError(e.code(), e.getMessage(), null));
    }

    @ExceptionHandler(PreludeApiException.ContractViolation.class)
    public ResponseEntity<?> contractViolation(PreludeApiException.ContractViolation e) {
        return ResponseEntity.badRequest().body(new ApiError(e.code(), e.getMessage(), null));
    }

    @ExceptionHandler(PreludeApiException.AdminAuthFailed.class)
    public ResponseEntity<?> adminAuth(PreludeApiException.AdminAuthFailed e) {
        return ResponseEntity.status(HttpStatus.FORBIDDEN).body(new ApiError(e.code(), e.getMessage(), null));
    }

    @ExceptionHandler(MethodArgumentNotValidException.class)
    public ResponseEntity<?> invalid(MethodArgumentNotValidException e) {
        List<Map<String, String>> fieldErrors = e.getBindingResult().getFieldErrors().stream()
                .map(fe -> Map.of(
                        "field", fe.getField(),
                        "error", fe.getDefaultMessage() == null ? "invalid" : fe.getDefaultMessage()))
                .toList();
        return ResponseEntity.badRequest().body(new ApiError("VALIDATION_ERROR",
                "Request failed contract validation. Aggregate-only submissions are rejected: per-image rows "
                        + "are mandatory (docs/data-contract.md §2.4).", fieldErrors));
    }

    @ExceptionHandler(HttpMessageNotReadableException.class)
    public ResponseEntity<?> unreadable(HttpMessageNotReadableException e) {
        return ResponseEntity.badRequest().body(new ApiError("MALFORMED_REQUEST",
                "Request body could not be parsed as a valid ingestion envelope.", null));
    }

    @ExceptionHandler(Exception.class)
    public ResponseEntity<?> unexpected(Exception e) {
        log.error("Unexpected error in results-service", e);
        return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                .body(new ApiError("INTERNAL", "Unexpected server error.", null));
    }

    private ResponseEntity<?> conflict(String code, String message) {
        return ResponseEntity.status(HttpStatus.CONFLICT).body(new ApiError(code, message, null));
    }

    private String rootMessage(Throwable t) {
        Throwable c = t;
        while (c.getCause() != null && c.getCause() != c) {
            c = c.getCause();
        }
        return c.getMessage() == null ? c.getClass().getSimpleName() : c.getMessage();
    }
}