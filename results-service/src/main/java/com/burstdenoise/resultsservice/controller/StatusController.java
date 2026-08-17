package com.burstdenoise.resultsservice.controller;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.RestController;

import java.util.Map;

/**
 * Placeholder root endpoint. Real ingestion endpoints (POST /api/v1/metrics, etc.)
 * land once the data contract (docs/data-contract.md) is finalized — this just
 * proves the container, Postgres connection, and Spring Boot app are wired up.
 */
@RestController
public class StatusController {

    @GetMapping("/")
    public Map<String, String> status() {
        return Map.of(
                "service", "results-service",
                "status", "up",
                "note", "Ingestion API not yet implemented — see docs/data-contract.md"
        );
    }
}
