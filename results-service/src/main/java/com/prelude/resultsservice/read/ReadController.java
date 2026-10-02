package com.prelude.resultsservice.read;

import java.util.Map;

import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;

import com.prelude.resultsservice.read.dto.ReadDtos.AblationResponse;
import com.prelude.resultsservice.read.dto.ReadDtos.BootstrapComputeRequest;
import com.prelude.resultsservice.read.dto.ReadDtos.BootstrapComputeResponse;
import com.prelude.resultsservice.read.dto.ReadDtos.ImageScoreRow;
import com.prelude.resultsservice.read.dto.ReadDtos.Paged;
import com.prelude.resultsservice.read.dto.ReadDtos.RunDetail;
import com.prelude.resultsservice.read.dto.ReadDtos.RunSummary;

/**
 * Read & query API (addendum §13). Reads are open on the LAN (no token). Base path is /api/v1;
 * no handler here collides with the existing POST-only RunController or ingestion controllers.
 */
@RestController
@RequestMapping("/api/v1")
public class ReadController {

    private final ReadService reads;

    public ReadController(ReadService reads) {
        this.reads = reads;
    }

    @GetMapping("/runs")
    public Paged<RunSummary> listRuns(@RequestParam(defaultValue = "0") int page,
                                      @RequestParam(defaultValue = "50") int size) {
        return reads.listRuns(clampPage(page), clampSize(size));
    }

    @GetMapping("/runs/{runId}")
    public RunDetail getRun(@PathVariable String runId) {
        return reads.getRun(runId);
    }

    @GetMapping("/runs/{runId}/scores")
    public Paged<ImageScoreRow> scores(@PathVariable String runId,
                                       @RequestParam String stage,
                                       @RequestParam(required = false) String variant,
                                       @RequestParam(required = false) String strategy,
                                       @RequestParam(defaultValue = "false") boolean history,
                                       @RequestParam(defaultValue = "0") int page,
                                       @RequestParam(defaultValue = "50") int size) {
        return reads.imageScores(runId, stage, variant, strategy, history, clampPage(page), clampSize(size));
    }

    @GetMapping("/runs/{runId}/ablation")
    public AblationResponse ablation(@PathVariable String runId) {
        return reads.ablation(runId);
    }

    @GetMapping("/runs/{runId}/snapshot")
    public Map<String, Object> snapshot(@PathVariable String runId) {
        return reads.snapshot(runId);
    }

    @PostMapping("/runs/{runId}/bootstrap")
    public BootstrapComputeResponse computeBootstrap(@PathVariable String runId,
                                                     @RequestBody BootstrapComputeRequest request) {
        return reads.computeBootstrap(runId, request);
    }

    private static int clampPage(int page) {
        return Math.max(0, page);
    }

    private static int clampSize(int size) {
        return Math.max(1, Math.min(500, size));
    }
}