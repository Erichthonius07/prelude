package com.prelude.resultsservice.dto;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotNull;

import java.time.Instant;
import java.util.UUID;

/**
 * Ingestion envelope. Note: resultVersion was removed from the wire contract at ratification —
 * supersede versions are server-assigned, monotonic per grouping key (data-contract.md §3.2).
 * If a client still sends a resultVersion field, it is ignored.
 */
public record IngestionEnvelope<T>(
        @NotNull UUID submissionId,
        @NotBlank String runId,
        String deviceId,
        String appVersion,
        Instant occurredAt,
        @NotNull @Valid T payload) {
}