package com.prelude.denoise.timeout

import com.prelude.denoise.model.FusedFrame

/**
 * Abstraction over the synchronous, non-cancellable LiteRT inference call (T1).
 *
 * Implementations:
 * - Production: wraps `Interpreter.run()` or `CompiledModel.run()` (added later)
 * - Test: `FakeInferenceRunner` with configurable latency/exceptions
 *
 * A single instance must NEVER be called concurrently (rule D4).
 */
interface InferenceRunner {
    /**
     * Run inference on [input]. This call is synchronous and blocking.
     * It CANNOT be cancelled mid-run (T1).
     *
     * @return Denoised image pixel data as a float array.
     */
    fun run(input: FusedFrame): FloatArray
}
