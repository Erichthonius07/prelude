# Minimal stand-in for the Results Service, for testing the app against the DRAFT contract before Role 1's
# Spring Boot service is live. Serves docs/sample_comparison_report.json and accepts metric batches.
# Usage (on the laptop): python mock_results_service.py ../sample_comparison_report.json
import http.server, sys, json
SAMPLE = open(sys.argv[1]).read()
class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith('/api/v1/comparisons/latest') or self.path == '/api/v1/health':
            body = SAMPLE.encode() if 'comparisons' in self.path else b'{"status":"ok"}'
            self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(body)
        else: self.send_response(404); self.end_headers()
    def do_POST(self):
        n = int(self.headers.get('Content-Length', 0)); data = json.loads(self.rfile.read(n))
        print("POST", self.path, "key", self.headers.get('Idempotency-Key'), "subs", len(data['submissions']), file=sys.stderr, flush=True)
        self.send_response(202); self.end_headers()
http.server.HTTPServer(('0.0.0.0', 8080), H).serve_forever()
