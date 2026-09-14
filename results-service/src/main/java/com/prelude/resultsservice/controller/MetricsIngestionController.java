package com.prelude.resultsservice.controller;

import com.prelude.resultsservice.dto.AlignmentPayload;
import com.prelude.resultsservice.dto.BatchPayload;
import com.prelude.resultsservice.dto.BootstrapPayload;
import com.prelude.resultsservice.dto.CapturePayload;
import com.prelude.resultsservice.dto.DenoisePayload;
import com.prelude.resultsservice.dto.FusionPayload;
import com.prelude.resultsservice.dto.IngestionEnvelope;
import com.prelude.resultsservice.dto.IngestionResponse;
import com.prelude.resultsservice.dto.PostProcessPayload;
import com.prelude.resultsservice.dto.TrainingPayload;
import com.prelude.resultsservice.service.IngestionService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
@RequestMapping("/api/v1")
public class MetricsIngestionController {

    private final IngestionService ingestion;

    public MetricsIngestionController(IngestionService ingestion) {
        this.ingestion = ingestion;
    }

    @PostMapping("/metrics/capture")
    public IngestionResponse capture(@Valid @RequestBody IngestionEnvelope<CapturePayload> envelope) {
        return ingestion.ingestCapture(envelope);
    }

    @PostMapping("/metrics/alignment")
    public IngestionResponse alignment(@Valid @RequestBody IngestionEnvelope<AlignmentPayload> envelope) {
        return ingestion.ingestAlignment(envelope);
    }

    @PostMapping("/metrics/fusion")
    public IngestionResponse fusion(@Valid @RequestBody IngestionEnvelope<FusionPayload> envelope) {
        return ingestion.ingestFusion(envelope);
    }

    @PostMapping("/metrics/denoise")
    public IngestionResponse denoise(@Valid @RequestBody IngestionEnvelope<DenoisePayload> envelope) {
        return ingestion.ingestDenoise(envelope);
    }

    @PostMapping("/metrics/postprocess")
    public IngestionResponse postprocess(@Valid @RequestBody IngestionEnvelope<PostProcessPayload> envelope) {
        return ingestion.ingestPostProcess(envelope);
    }

    @PostMapping("/metrics/training")
    public IngestionResponse training(@Valid @RequestBody IngestionEnvelope<TrainingPayload> envelope) {
        return ingestion.ingestTraining(envelope);
    }

    @PostMapping("/metrics/batch-variant")
    public IngestionResponse batchVariant(@Valid @RequestBody IngestionEnvelope<BatchPayload> envelope) {
        return ingestion.ingestBatchVariant(envelope);
    }

    @PostMapping("/metrics/bootstrap")
    public IngestionResponse bootstrap(@Valid @RequestBody IngestionEnvelope<BootstrapPayload> envelope) {
        return ingestion.ingestBootstrap(envelope);
    }
}