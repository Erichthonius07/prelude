package com.prelude.resultsservice.controller;

import com.prelude.resultsservice.dto.RunRegistration;
import com.prelude.resultsservice.service.IngestionService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1/runs")
public class RunController {

    private final IngestionService ingestion;

    public RunController(IngestionService ingestion) {
        this.ingestion = ingestion;
    }

    @PostMapping
    public RunRegistration.Response register(@Valid @RequestBody RunRegistration request) {
        return ingestion.registerRun(request);
    }
}