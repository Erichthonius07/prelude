package com.prelude.resultsservice.dto;

import java.util.List;
import java.util.UUID;

public record IngestionResponse(
        UUID submissionId,
        String runId,
        String stage,
        boolean duplicate,
        int acceptedImageCount,
        List<String> warnings) {
}