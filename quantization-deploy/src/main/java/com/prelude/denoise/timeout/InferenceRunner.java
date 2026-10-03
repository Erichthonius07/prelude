package com.prelude.denoise.timeout;

import com.prelude.denoise.model.FusedFrame;

/**
 * Abstraction over the synchronous, non-cancellable LiteRT inference call (T1).
 *
 * Implementations:
 *   Production: wraps Interpreter.run() or CompiledModel.run() (added later)
 *   Test: FakeInferenceRunner with configurable latency/exceptions
 *
 * A single instance must NEVER be called concurrently (rule D4).
 */
public interface InferenceRunner {

    /**
     * Run inference on the given input. This call is synchronous and blocking.
     * It CANNOT be cancelled mid-run (T1).
     *
     * @param input the fused frame to denoise
     * @return denoised image pixel data as a float array
     */
    float[] run(FusedFrame input);
}
