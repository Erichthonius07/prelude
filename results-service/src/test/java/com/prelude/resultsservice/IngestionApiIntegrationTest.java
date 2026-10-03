package com.prelude.resultsservice;

import com.prelude.resultsservice.dto.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.boot.testcontainers.service.connection.ServiceConnection;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.http.client.ClientHttpResponse;
import org.springframework.test.context.TestPropertySource;
import org.springframework.web.client.ResponseErrorHandler;
import org.springframework.web.client.RestTemplate;
import org.testcontainers.containers.PostgreSQLContainer;
import org.testcontainers.junit.jupiter.Container;
import org.testcontainers.junit.jupiter.Testcontainers;

import java.time.Instant;
import java.util.List;
import java.util.UUID;

import static org.assertj.core.api.Assertions.assertThat;

@Testcontainers
@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
@TestPropertySource(properties = "prelude.admin-token=ci-test-token")
class IngestionApiIntegrationTest {

    @Container
    @ServiceConnection
    static PostgreSQLContainer<?> postgres = new PostgreSQLContainer<>("postgres:18-alpine");

    @LocalServerPort
    int port;

    RestTemplate restTemplate;

    @BeforeEach
    void setUp() {
        restTemplate = new RestTemplate();
        // TestRestTemplate was removed in Boot 4 and never threw on error statuses; replicate
        // that so the 400/409 assertions can inspect the response body instead of catching.
        restTemplate.setErrorHandler(new ResponseErrorHandler() {
            @Override
            public boolean hasError(ClientHttpResponse response) {
                return false;
            }
        });
    }

    @Test
    void happyPath_captureIngestSucceeds() {
        String runId = "test-run-" + UUID.randomUUID();
        registerRun(runId);
        IngestionEnvelope<CapturePayload> env = buildCaptureEnvelope(runId, UUID.randomUUID(), "img-1");
        ResponseEntity<IngestionResponse> resp = restTemplate.postForEntity(
            uri("/api/v1/metrics/capture"), env, IngestionResponse.class);
        assertThat(resp.getStatusCode()).isEqualTo(HttpStatus.CREATED);
        assertThat(resp.getBody()).isNotNull();
        assertThat(resp.getBody().duplicate()).isFalse();
        assertThat(resp.getBody().acceptedImageCount()).isEqualTo(1);
    }

    @Test
    void idempotency_safeRetry_returnsDuplicate() {
        String runId = "test-run-" + UUID.randomUUID();
        registerRun(runId);
        IngestionEnvelope<CapturePayload> env = buildCaptureEnvelope(runId, UUID.randomUUID(), "img-1");
        restTemplate.postForEntity(uri("/api/v1/metrics/capture"), env, IngestionResponse.class);
        ResponseEntity<IngestionResponse> resp2 = restTemplate.postForEntity(
            uri("/api/v1/metrics/capture"), env, IngestionResponse.class);
        assertThat(resp2.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(resp2.getBody()).isNotNull();
        assertThat(resp2.getBody().duplicate()).isTrue();
    }

    @Test
    void idempotency_keyReused_returns409() {
        String runId = "test-run-" + UUID.randomUUID();
        registerRun(runId);
        UUID subId = UUID.randomUUID();
        restTemplate.postForEntity(uri("/api/v1/metrics/capture"),
            buildCaptureEnvelope(runId, subId, "img-1"), IngestionResponse.class);
        IngestionEnvelope<CapturePayload> env2 = buildCaptureEnvelope(runId, subId, "img-2");
        ResponseEntity<String> resp2 = restTemplate.postForEntity(
            uri("/api/v1/metrics/capture"), env2, String.class);
        assertThat(resp2.getStatusCode()).isEqualTo(HttpStatus.CONFLICT);
        assertThat(resp2.getBody()).contains("IDEMPOTENCY_KEY_REUSED");
    }

    @Test
    void supersede_serverAssignsMonotonicVersions() {
        String runId = "test-run-" + UUID.randomUUID();
        registerRun(runId);
        ResponseEntity<IngestionResponse> resp1 = restTemplate.postForEntity(
            uri("/api/v1/metrics/capture"), buildCaptureEnvelope(runId, UUID.randomUUID(), "img-1"),
            IngestionResponse.class);
        assertThat(resp1.getStatusCode()).isEqualTo(HttpStatus.CREATED);
        ResponseEntity<IngestionResponse> resp2 = restTemplate.postForEntity(
            uri("/api/v1/metrics/capture"), buildCaptureEnvelope(runId, UUID.randomUUID(), "img-1"),
            IngestionResponse.class);
        assertThat(resp2.getStatusCode()).isEqualTo(HttpStatus.CREATED);
        assertThat(resp2.getBody().duplicate()).isFalse();
    }

    @Test
    void leakageGuard_rejectsTrainingOnHeldOutImage() {
        String runId = "test-run-" + UUID.randomUUID();
        registerRun(runId);

        TestSetDtos.AddImagesRequest addReq = new TestSetDtos.AddImagesRequest(
            List.of(new TestSetDtos.ManifestImage("leaked-img", "abc123", "sidd")));
        HttpHeaders headers = new HttpHeaders();
        headers.set("X-Prelude-Admin-Token", "ci-test-token");
        restTemplate.postForEntity(uri("/api/v1/testset/images"),
            new HttpEntity<>(addReq, headers), TestSetDtos.ManifestResponse.class);

        TrainingPayload payload = new TrainingPayload("denoise_cnn",
            List.of(new TrainingPayload.ImageScore("leaked-img", 1, 100, "psnr", 30.0, "train")),
            null);
        IngestionEnvelope<TrainingPayload> env = new IngestionEnvelope<>(
            UUID.randomUUID(), runId, "dev", "1.0", Instant.now(), payload);
        ResponseEntity<String> resp = restTemplate.postForEntity(
            uri("/api/v1/metrics/training"), env, String.class);
        assertThat(resp.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(resp.getBody()).contains("TEST_SET_LEAKAGE");
    }

    // --- helpers ---

    private String uri(String path) {
        return "http://localhost:" + port + path;
    }

    private void registerRun(String runId) {
        RunRegistration reg = new RunRegistration(runId, "test", "phase1", 10, 1, 10000, 0.95, "test");
        restTemplate.postForEntity(uri("/api/v1/runs"), reg, RunRegistration.Response.class);
    }

    private IngestionEnvelope<CapturePayload> buildCaptureEnvelope(String runId, UUID subId, String imageId) {
        CapturePayload.Frame frame = new CapturePayload.Frame(
            0, 3200, 83000000L, System.currentTimeMillis(), 0.42, false, false);
        CapturePayload.Burst burst = new CapturePayload.Burst(imageId, List.of(frame));
        CapturePayload payload = new CapturePayload(List.of(burst));
        return new IngestionEnvelope<>(subId, runId, "test-device", "1.0.0", Instant.now(), payload);
    }
}
