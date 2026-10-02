package com.prelude.resultsservice.controller;

import com.prelude.resultsservice.dto.RunRegistration;
import com.prelude.resultsservice.service.IngestionService;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
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
    public ResponseEntity<RunRegistration.Response> register(@Valid @RequestBody RunRegistration request) {
        RunRegistration.Response response = ingestion.registerRun(request);
        // Contract §10: 201 on first acceptance; 200 with duplicate:true on idempotent replay.
        HttpStatus status = response.duplicate() ? HttpStatus.OK : HttpStatus.CREATED;
        return ResponseEntity.status(status).body(response);
    }
}
