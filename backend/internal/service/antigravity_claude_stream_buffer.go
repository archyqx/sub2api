package service

import (
	"bytes"
	"net/http"
	"strings"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/pkg/antigravity"
	"github.com/Wei-Shaw/sub2api/internal/pkg/logger"
	"github.com/tidwall/gjson"
)

const antigravityClaudePreContentKeepaliveGrace = 25 * time.Second

// antigravityClaudeStreamBuffer buffers pre-content Claude SSE events until
// substantive output arrives so empty upstream responses can trigger failover retry
// instead of committing HTTP 200 end_turn with 0 output tokens.
type antigravityClaudeStreamBuffer struct {
	pending                  bytes.Buffer
	committed                bool
	preContentKeepaliveSent  bool
	preContentKeepaliveGrace time.Duration

	seenLines            int
	lastFinishReason     string
	lastBlockReason      string
	lastPromptTokens     int64
	lastCandidatesTokens int64
	lastThoughtsTokens   int64
	reqBodyBytes         int
}

func newAntigravityClaudeStreamBuffer() *antigravityClaudeStreamBuffer {
	return newAntigravityClaudeStreamBufferWithBytes(0)
}

func newAntigravityClaudeStreamBufferWithBytes(reqBodyBytes int) *antigravityClaudeStreamBuffer {
	return &antigravityClaudeStreamBuffer{
		preContentKeepaliveGrace: antigravityClaudePreContentKeepaliveGrace,
		reqBodyBytes:             reqBodyBytes,
	}
}

// extractAntigravitySSELineRoot unwraps a raw SSE data line into its inner response gjson object.
func extractAntigravitySSELineRoot(line string) (gjson.Result, bool) {
	trimmed := strings.TrimSpace(line)
	if !strings.HasPrefix(trimmed, "data:") {
		return gjson.Result{}, false
	}
	payload := strings.TrimSpace(strings.TrimPrefix(trimmed, "data:"))
	if payload == "" || payload == "[DONE]" {
		return gjson.Result{}, false
	}
	parsed := gjson.Parse(payload)
	if resp := parsed.Get("response"); resp.Exists() && resp.IsObject() {
		return resp, true
	}
	return parsed, true
}

// ObserveRawLine records upstream finishReason, blockReason, and token counts for diagnostics.
func (b *antigravityClaudeStreamBuffer) ObserveRawLine(line string) {
	root, ok := extractAntigravitySSELineRoot(line)
	if !ok {
		return
	}
	b.seenLines++
	if fr := strings.TrimSpace(root.Get("candidates.0.finishReason").String()); fr != "" {
		b.lastFinishReason = fr
	}
	if br := strings.TrimSpace(root.Get("promptFeedback.blockReason").String()); br != "" {
		b.lastBlockReason = br
	}
	if um := root.Get("usageMetadata"); um.Exists() && um.IsObject() {
		b.lastPromptTokens = um.Get("promptTokenCount").Int()
		b.lastCandidatesTokens = um.Get("candidatesTokenCount").Int()
		b.lastThoughtsTokens = um.Get("thoughtsTokenCount").Int()
	}
}

func classifyAntigravityEmptyStreamReason(finishReason, blockReason string, candidatesTokens, thoughtsTokens int64) string {
	if br := strings.TrimSpace(blockReason); br != "" {
		return "blocked:" + br
	}
	fr := strings.ToUpper(strings.TrimSpace(finishReason))
	switch fr {
	case "STOP":
		if thoughtsTokens > 0 && candidatesTokens == 0 {
			return "thought_only_stop"
		}
		return "empty_stop"
	case "MAX_TOKENS":
		if thoughtsTokens > 0 && candidatesTokens == 0 {
			return "thought_only_max_tokens"
		}
		return "empty_max_tokens"
	case "":
		return "silent_eof"
	default:
		return strings.ToLower(fr)
	}
}

// isAntigravityContentFilterReason reports whether upstream terminated the stream
// due to a deterministic safety or policy filter rather than a transient empty response.
func isAntigravityContentFilterReason(finishReason, blockReason string) bool {
	if strings.TrimSpace(blockReason) != "" {
		return true
	}
	switch strings.ToUpper(strings.TrimSpace(finishReason)) {
	case "SAFETY", "PROHIBITED_CONTENT", "RECITATION", "BLOCKLIST", "SPII", "IMAGE_SAFETY":
		return true
	default:
		return false
	}
}

// newAntigravityEmptyStreamFailoverError builds an UpstreamFailoverError for empty
// Antigravity streams. Transient empty streams (including stochastic MAX_TOKENS thinking
// spikes and empty chunks) keep the full same-account retry budget without cooling the
// account, while deterministic safety/policy blocks stop immediately without retrying
// or rotating across accounts.
func newAntigravityEmptyStreamFailoverError(finishReason, blockReason string) *UpstreamFailoverError {
	if isAntigravityContentFilterReason(finishReason, blockReason) {
		return &UpstreamFailoverError{
			StatusCode:             http.StatusBadGateway,
			ResponseBody:           []byte(`{"error":"empty stream response from upstream"}`),
			RetryableOnSameAccount: false,
			RequestScopedTransient: true,
			NextAccountAction:      NextAccountStop,
		}
	}
	return &UpstreamFailoverError{
		StatusCode:             http.StatusBadGateway,
		ResponseBody:           []byte(`{"error":"empty stream response from upstream"}`),
		RetryableOnSameAccount: true,
		RequestScopedTransient: true,
	}
}

const (
	antigravityLongContextPromptTokensThreshold = 100_000
	antigravityLongContextReqBytesThreshold     = 400 * 1024
	antigravityLongContextElapsedThreshold      = 15 * time.Second
)

// newAntigravityEmptyStreamFailoverErrorWithContext caps same-account retries to 1 on
// deterministic long-context silent EOFs (finishReason=="", seenLines<=1, and
// promptTokens>=100k || reqBodyBytes>=400KB || elapsed>=15s) while leaving MAX_TOKENS
// and short/medium-context empty streams at the full retry budget.
func newAntigravityEmptyStreamFailoverErrorWithContext(
	finishReason, blockReason string,
	seenLines int,
	promptTokens int64,
	reqBodyBytes int,
	elapsed time.Duration,
) *UpstreamFailoverError {
	err := newAntigravityEmptyStreamFailoverError(finishReason, blockReason)
	if err.RetryableOnSameAccount &&
		strings.TrimSpace(finishReason) == "" &&
		seenLines <= 1 &&
		(promptTokens >= antigravityLongContextPromptTokensThreshold ||
			reqBodyBytes >= antigravityLongContextReqBytesThreshold ||
			elapsed >= antigravityLongContextElapsedThreshold) {
		err.SameAccountRetryMax = 1
	}
	return err
}

func (b *antigravityClaudeStreamBuffer) hasMeaningfulContent(processor *antigravity.StreamingProcessor, events []byte) bool {
	if processor != nil && processor.HasContent() {
		return true
	}
	// Grounding metadata is emitted as text_delta inside emitFinish() without flipping p.hasContent.
	return bytes.Contains(events, []byte(`"type":"text_delta"`))
}

// WriteEvents writes or buffers converted Claude SSE events until meaningful content arrives.
func (b *antigravityClaudeStreamBuffer) WriteEvents(
	cw *antigravityClientWriter,
	processor *antigravity.StreamingProcessor,
	events []byte,
	onFirstContent func(),
) {
	if len(events) == 0 {
		return
	}
	if b.committed {
		cw.Write(events)
		return
	}
	_, _ = b.pending.Write(events)
	if b.hasMeaningfulContent(processor, events) {
		if onFirstContent != nil {
			onFirstContent()
		}
		b.committed = true
		cw.Write(b.pending.Bytes())
		b.pending.Reset()
	}
}

// ShouldSendKeepalive delays pre-content pings past preContentKeepaliveGrace so slow empty streams remain retryable.
func (b *antigravityClaudeStreamBuffer) ShouldSendKeepalive(now, startTime, lastDataAt time.Time, keepaliveInterval time.Duration) bool {
	if keepaliveInterval <= 0 || now.Sub(lastDataAt) < keepaliveInterval {
		return false
	}
	if b.committed {
		return true
	}
	grace := b.preContentKeepaliveGrace
	if keepaliveInterval > grace {
		grace = keepaliveInterval
	}
	return now.Sub(startTime) >= grace
}

// SendKeepalive emits an SSE ping and tracks whether HTTP headers were committed before content arrived.
func (b *antigravityClaudeStreamBuffer) SendKeepalive(cw *antigravityClientWriter) bool {
	ok := cw.Fprintf("event: ping\ndata: {\"type\": \"ping\"}\n\n")
	if ok && !b.committed {
		b.preContentKeepaliveSent = true
	}
	return ok
}

// Finish completes the buffered stream on upstream EOF, returning a retryable failover error when no content was produced.
func (b *antigravityClaudeStreamBuffer) Finish(
	cw *antigravityClientWriter,
	processor *antigravity.StreamingProcessor,
	startTime time.Time,
	firstTokenMs **int,
	convertUsage func(*antigravity.ClaudeUsage) *ClaudeUsage,
	sendError func(string),
) (*antigravityStreamResult, error) {
	finalEvents, agUsage := processor.Finish()
	if b.committed {
		if len(finalEvents) > 0 {
			cw.Write(finalEvents)
		}
		return &antigravityStreamResult{
			usage:            convertUsage(agUsage),
			firstTokenMs:     *firstTokenMs,
			clientDisconnect: cw.Disconnected(),
		}, nil
	}

	if len(finalEvents) > 0 {
		_, _ = b.pending.Write(finalEvents)
	}
	if b.hasMeaningfulContent(processor, finalEvents) {
		if *firstTokenMs == nil {
			ms := int(time.Since(startTime).Milliseconds())
			*firstTokenMs = &ms
		}
		b.committed = true
		if b.pending.Len() > 0 {
			cw.Write(b.pending.Bytes())
			b.pending.Reset()
		}
		return &antigravityStreamResult{
			usage:            convertUsage(agUsage),
			firstTokenMs:     *firstTokenMs,
			clientDisconnect: cw.Disconnected(),
		}, nil
	}

	if cw.Disconnected() {
		return &antigravityStreamResult{
			usage:            convertUsage(agUsage),
			firstTokenMs:     *firstTokenMs,
			clientDisconnect: true,
		}, nil
	}

	emptyReason := classifyAntigravityEmptyStreamReason(b.lastFinishReason, b.lastBlockReason, b.lastCandidatesTokens, b.lastThoughtsTokens)
	logger.LegacyPrintf(
		"service.antigravity_gateway",
		"[antigravity-Claude-Stream] empty content response (empty_reason=%s message_start=%v keepalive_sent=%v lines=%d finish_reason=%q block_reason=%q prompt_tokens=%d candidates_tokens=%d thoughts_tokens=%d elapsed_ms=%d), triggering failover",
		emptyReason,
		processor.MessageStartSent(),
		b.preContentKeepaliveSent,
		b.seenLines,
		b.lastFinishReason,
		b.lastBlockReason,
		b.lastPromptTokens,
		b.lastCandidatesTokens,
		b.lastThoughtsTokens,
		time.Since(startTime).Milliseconds(),
	)

	if b.preContentKeepaliveSent {
		if sendError != nil {
			sendError("empty_stream")
		}
		return &antigravityStreamResult{
			usage:        convertUsage(agUsage),
			firstTokenMs: *firstTokenMs,
		}, newAntigravityCompatCommittedNoContentError("empty Antigravity claude stream after keepalive")
	}

	return nil, newAntigravityEmptyStreamFailoverErrorWithContext(
		b.lastFinishReason,
		b.lastBlockReason,
		b.seenLines,
		b.lastPromptTokens,
		b.reqBodyBytes,
		time.Since(startTime),
	)
}
