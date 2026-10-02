package com.prelude.resultsservice.read;

import org.springframework.core.Ordered;
import org.springframework.core.annotation.Order;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.RestControllerAdvice;

/**
 * Separate advice for the read API so no existing file is edited. Ordered with highest
 * precedence so its specific handlers win over the existing catch-all in ApiExceptionHandler;
 * exceptions it does not handle still fall through to the existing advice unchanged.
 */
@RestControllerAdvice
@Order(Ordered.HIGHEST_PRECEDENCE)
public class ReadApiExceptionHandler {

    public record ApiError(String code, String message) {
    }

    @ExceptionHandler(ReadApiException.RunNotFound.class)
    public ResponseEntity<ApiError> notFound(ReadApiException.RunNotFound e) {
        return ResponseEntity.status(HttpStatus.NOT_FOUND).body(new ApiError(e.code(), e.getMessage()));
    }

    @ExceptionHandler(ReadApiException.InvalidReadRequest.class)
    public ResponseEntity<ApiError> badRequest(ReadApiException.InvalidReadRequest e) {
        return ResponseEntity.status(HttpStatus.BAD_REQUEST).body(new ApiError(e.code(), e.getMessage()));
    }

    @ExceptionHandler(ReadApiException.BootstrapNotComputable.class)
    public ResponseEntity<ApiError> notComputable(ReadApiException.BootstrapNotComputable e) {
        return ResponseEntity.status(HttpStatus.CONFLICT).body(new ApiError(e.code(), e.getMessage()));
    }
}