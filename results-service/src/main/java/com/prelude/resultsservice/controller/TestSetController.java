package com.prelude.resultsservice.controller;

import com.prelude.resultsservice.api.PreludeApiException;
import com.prelude.resultsservice.dto.TestSetDtos.AddImagesRequest;
import com.prelude.resultsservice.dto.TestSetDtos.CheckRequest;
import com.prelude.resultsservice.dto.TestSetDtos.CheckResponse;
import com.prelude.resultsservice.dto.TestSetDtos.ManifestResponse;
import com.prelude.resultsservice.service.TestSetService;
import jakarta.validation.Valid;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestHeader;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/testset")
public class TestSetController {

    private final TestSetService testSet;
    private final String adminToken;

    public TestSetController(TestSetService testSet, @Value("${prelude.admin-token:}") String adminToken) {
        this.testSet = testSet;
        this.adminToken = adminToken;
    }

    @GetMapping("/manifest")
    public ManifestResponse manifest() {
        return testSet.manifest();
    }

    @PostMapping("/check")
    public CheckResponse check(@Valid @RequestBody CheckRequest request) {
        return testSet.check(request);
    }

    @PostMapping("/images")
    public ManifestResponse addImages(
            @RequestHeader(value = "X-Prelude-Admin-Token", required = false) String token,
            @Valid @RequestBody AddImagesRequest request) {
        if (adminToken == null || adminToken.isBlank() || !adminToken.equals(token)) {
            throw new PreludeApiException.AdminAuthFailed(
                    "Manifest mutation requires a valid X-Prelude-Admin-Token. This is placeholder auth — "
                            + "see open questions in docs/data-contract.md §8.");
        }
        return testSet.addImages(request);
    }
}