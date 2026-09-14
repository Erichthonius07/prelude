package com.prelude.resultsservice.service;

import java.lang.reflect.RecordComponent;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.time.Instant;
import java.util.Collection;
import java.util.Map;
import java.util.TreeMap;

/**
 * Dependency-free canonical serializer used for the idempotency payload fingerprint.
 * Deliberately avoids any JSON-library types so retry-equality does not depend on
 * which JSON provider the framework ships with.
 */
public final class Canonicalizer {

    private Canonicalizer() {
    }

    public static String write(Object value) {
        StringBuilder sb = new StringBuilder();
        writeValue(value, sb);
        return sb.toString();
    }

       /** Fingerprint over the semantically meaningful envelope parts. occurredAt/deviceId/appVersion
     *  are excluded: they may legitimately differ between retries of the same submission.
     *  resultVersion is not part of the fingerprint: it no longer exists on the wire (server-assigned). */
    public static String fingerprint(String stage, String runId, String canonicalPayloadJson) {
        return stage + '\u0000' + runId + '\u0000' + canonicalPayloadJson;
    }

    public static String sha256Hex(String input) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            byte[] digest = md.digest(input.getBytes(StandardCharsets.UTF_8));
            StringBuilder hex = new StringBuilder(digest.length * 2);
            for (byte b : digest) {
                hex.append(Character.forDigit((b >> 4) & 0xF, 16)).append(Character.forDigit(b & 0xF, 16));
            }
            return hex.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new IllegalStateException("SHA-256 unavailable", e);
        }
    }

    private static void writeValue(Object v, StringBuilder sb) {
        if (v == null) {
            sb.append("null");
            return;
        }
        if (v instanceof String s) {
            writeString(s, sb);
            return;
        }
        if (v instanceof Boolean b) {
            sb.append(b.booleanValue());
            return;
        }
        if (v instanceof Integer || v instanceof Long || v instanceof Short || v instanceof Byte) {
            sb.append(((Number) v).longValue());
            return;
        }
        if (v instanceof Double d) {
            writeDouble(d, sb);
            return;
        }
        if (v instanceof Float f) {
            writeDouble(f.doubleValue(), sb);
            return;
        }
        if (v instanceof Number n) {
            sb.append(n);
            return;
        }
        if (v instanceof Instant i) {
            writeString(i.toString(), sb);
            return;
        }
        if (v instanceof Enum<?> e) {
            writeString(e.name(), sb);
            return;
        }
        if (v instanceof Map<?, ?> m) {
            TreeMap<String, Object> sorted = new TreeMap<>();
            for (Map.Entry<?, ?> e : m.entrySet()) {
                if (!(e.getKey() instanceof String key)) {
                    throw new IllegalArgumentException("Non-string map key in canonical JSON");
                }
                sorted.put(key, e.getValue());
            }
            writeSortedMap(sorted, sb);
            return;
        }
        if (v instanceof Collection<?> c) {
            sb.append('[');
            boolean first = true;
            for (Object o : c) {
                if (!first) {
                    sb.append(',');
                }
                first = false;
                writeValue(o, sb);
            }
            sb.append(']');
            return;
        }
        if (v.getClass().isRecord()) {
            writeRecord(v, sb);
            return;
        }
        throw new IllegalArgumentException("Cannot canonicalize type " + v.getClass().getName());
    }

    private static void writeRecord(Object record, StringBuilder sb) {
        TreeMap<String, Object> values = new TreeMap<>();
        for (RecordComponent component : record.getClass().getRecordComponents()) {
            try {
                values.put(component.getName(), component.getAccessor().invoke(record));
            } catch (ReflectiveOperationException e) {
                throw new IllegalStateException("Failed to read record component " + component.getName(), e);
            }
        }
        writeSortedMap(values, sb);
    }

    private static void writeSortedMap(TreeMap<String, Object> sorted, StringBuilder sb) {
        sb.append('{');
        boolean first = true;
        for (Map.Entry<String, Object> e : sorted.entrySet()) {
            if (!first) {
                sb.append(',');
            }
            first = false;
            writeString(e.getKey(), sb);
            sb.append(':');
            writeValue(e.getValue(), sb);
        }
        sb.append('}');
    }

    private static void writeDouble(double d, StringBuilder sb) {
        if (Double.isNaN(d) || Double.isInfinite(d)) {
            throw new IllegalArgumentException("Non-finite double in payload");
        }
        sb.append(Double.toString(d));
    }

    private static void writeString(String s, StringBuilder sb) {
        sb.append('"');
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"' -> sb.append("\\\"");
                case '\\' -> sb.append("\\\\");
                case '\n' -> sb.append("\\n");
                case '\r' -> sb.append("\\r");
                case '\t' -> sb.append("\\t");
                case '\b' -> sb.append("\\b");
                case '\f' -> sb.append("\\f");
                default -> {
                    if (c < 0x20) {
                        sb.append(String.format("\\u%04x", (int) c));
                    } else {
                        sb.append(c);
                    }
                }
            }
        }
        sb.append('"');
    }
}