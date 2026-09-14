package com.prelude.resultsservice.dto;

import java.util.List;

import jakarta.validation.Valid;
import jakarta.validation.constraints.NotBlank;
import jakarta.validation.constraints.NotEmpty;

public final class TestSetDtos {

    private TestSetDtos() {}

    public record ManifestImage(
            @NotBlank String imageId,
            @NotBlank String sha256,
            @NotBlank String source) {}

    public record ManifestResponse(int imageCount, List<Image> images) {
        public record Image(String imageId, String sha256, String source, int manifestVersion) {}
    }

    public record Candidate(String imageId, String sha256) {}

    public record CheckRequest(@NotEmpty List<@Valid Candidate> candidates) {}

    public record CheckResponse(List<Match> heldout, List<Candidate> notInHeldout) {
        public record Match(Candidate candidate, String matchedImageId, String matchedBy) {}
    }

    public record AddImagesRequest(@NotEmpty List<@Valid ManifestImage> images) {}
}
