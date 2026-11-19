package service

import (
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/Wei-Shaw/sub2api/internal/config"
	"github.com/Wei-Shaw/sub2api/internal/pkg/antigravity"
	"github.com/gin-gonic/gin"
	"github.com/stretchr/testify/require"
)

func TestHandleClaudeStreamingResponse_EmptyContentReturnsFailoverError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tests := []struct {
		name             string
		sse              string
		wantRetryable    bool
		wantNextAcctStop bool
	}{
		{
			name:             "usage_only_stop_zero_output",
			sse:              `data: {"response":{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":394822,"cachedContentTokenCount":392768,"candidatesTokenCount":0,"thoughtsTokenCount":0},"modelVersion":"gemini-4-hydrogen"}}` + "\n\n",
			wantRetryable:    true,
			wantNextAcctStop: false,
		},
		{
			name:             "empty_finish_reason_zero_output",
			sse:              `data: {"response":{"usageMetadata":{"promptTokenCount":186997,"candidatesTokenCount":0,"thoughtsTokenCount":0}}}` + "\n\n",
			wantRetryable:    true,
			wantNextAcctStop: false,
		},
		{
			name:             "malformed_function_call_signature_only",
			sse:              `data: {"response":{"candidates":[{"content":{"parts":[{"thoughtSignature":"sig-only"}]},"finishReason":"MALFORMED_FUNCTION_CALL"}],"usageMetadata":{"promptTokenCount":120,"candidatesTokenCount":0}}}` + "\n\n",
			wantRetryable:    true,
			wantNextAcctStop: false,
		},
		{
			name:             "max_tokens_zero_output_retains_same_account_retry",
			sse:              `data: {"response":{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":37000,"candidatesTokenCount":0,"thoughtsTokenCount":1024}}}` + "\n\n",
			wantRetryable:    true,
			wantNextAcctStop: false,
		},
		{
			name:             "prompt_blocked_no_candidates_stops_immediately",
			sse:              `data: {"response":{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":500,"candidatesTokenCount":0}}}` + "\n\n",
			wantRetryable:    false,
			wantNextAcctStop: true,
		},
		{
			name:             "candidate_prohibited_content_stops_immediately",
			sse:              `data: {"response":{"candidates":[{"finishReason":"PROHIBITED_CONTENT"}],"usageMetadata":{"promptTokenCount":500,"candidatesTokenCount":0}}}` + "\n\n",
			wantRetryable:    false,
			wantNextAcctStop: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := newAntigravityTestService(&config.Config{
				Gateway: config.GatewayConfig{MaxLineSize: defaultMaxLineSize},
			})

			rec := httptest.NewRecorder()
			c, _ := gin.CreateTestContext(rec)
			c.Request = httptest.NewRequest(http.MethodPost, "/antigravity/v1/messages", nil)
			sizeBefore := c.Writer.Size()

			pr, pw := io.Pipe()
			resp := &http.Response{StatusCode: http.StatusOK, Body: pr, Header: http.Header{}}
			go func() {
				defer func() { _ = pw.Close() }()
				_, _ = io.WriteString(pw, tt.sse)
			}()

			res, err := svc.handleClaudeStreamingResponse(c, resp, time.Now(), "gemini-4-hydrogen")
			_ = pr.Close()

			require.Nil(t, res)
			require.Error(t, err)
			var failoverErr *UpstreamFailoverError
			require.ErrorAs(t, err, &failoverErr)
			require.Equal(t, http.StatusBadGateway, failoverErr.StatusCode)
			require.Equal(t, tt.wantRetryable, failoverErr.RetryableOnSameAccount)
			require.Zero(t, failoverErr.SameAccountRetryMax)
			require.Equal(t, !tt.wantNextAcctStop, failoverErr.ShouldRetryNextAccount())
			require.True(t, failoverErr.RequestScopedTransient)
			require.Equal(t, sizeBefore, c.Writer.Size(), "nothing should be written before meaningful content")
			require.Empty(t, rec.Body.String())
			require.Empty(t, rec.Header().Get("Content-Type"))
		})
	}
}

func TestHandleClaudeStreamingResponse_BuffersUntilRealContentArrives(t *testing.T) {
	gin.SetMode(gin.TestMode)
	svc := newAntigravityTestService(&config.Config{
		Gateway: config.GatewayConfig{MaxLineSize: defaultMaxLineSize},
	})

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/antigravity/v1/messages", nil)

	pr, pw := io.Pipe()
	resp := &http.Response{StatusCode: http.StatusOK, Body: pr, Header: http.Header{}}
	go func() {
		defer func() { _ = pw.Close() }()
		// First frame only carries usageMetadata (must NOT emit early/commit without content)
		fmt.Fprintln(pw, `data: {"response":{"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":80}}}`)
		fmt.Fprintln(pw, "")
		// Second frame carries actual text
		fmt.Fprintln(pw, `data: {"response":{"candidates":[{"content":{"parts":[{"text":"recovered"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":80,"candidatesTokenCount":5}}}`)
		fmt.Fprintln(pw, "")
	}()

	res, err := svc.handleClaudeStreamingResponse(c, resp, time.Now(), "gemini-4-hydrogen")
	_ = pr.Close()

	require.NoError(t, err)
	require.NotNil(t, res)
	require.Equal(t, 20, res.usage.InputTokens)
	require.Equal(t, 80, res.usage.CacheReadInputTokens)
	require.Equal(t, 5, res.usage.OutputTokens)

	body := rec.Body.String()
	require.Contains(t, body, "event: message_start")
	require.Contains(t, body, "recovered")
	require.Contains(t, body, "event: message_stop")
	require.Equal(t, "text/event-stream", rec.Header().Get("Content-Type"))
}

func TestHandleClaudeStreamingResponse_GroundingOnlyIsPreserved(t *testing.T) {
	gin.SetMode(gin.TestMode)
	svc := newAntigravityTestService(&config.Config{
		Gateway: config.GatewayConfig{MaxLineSize: defaultMaxLineSize},
	})

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/antigravity/v1/messages", nil)

	pr, pw := io.Pipe()
	resp := &http.Response{StatusCode: http.StatusOK, Body: pr, Header: http.Header{}}
	go func() {
		defer func() { _ = pw.Close() }()
		fmt.Fprintln(pw, `data: {"response":{"candidates":[{"groundingMetadata":{"webSearchQueries":["golang"]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":50,"candidatesTokenCount":12}}}`)
		fmt.Fprintln(pw, "")
	}()

	res, err := svc.handleClaudeStreamingResponse(c, resp, time.Now(), "gemini-2.5-pro")
	_ = pr.Close()

	require.NoError(t, err)
	require.NotNil(t, res)
	require.Contains(t, rec.Body.String(), "event: message_start")
	require.Contains(t, rec.Body.String(), "golang")
	require.Contains(t, rec.Body.String(), "event: message_stop")
}

func TestHandleClaudeStreamingResponse_SignedTextSingleChunkIsPreserved(t *testing.T) {
	gin.SetMode(gin.TestMode)
	svc := newAntigravityTestService(&config.Config{
		Gateway: config.GatewayConfig{MaxLineSize: defaultMaxLineSize},
	})

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/antigravity/v1/messages", nil)

	pr, pw := io.Pipe()
	resp := &http.Response{StatusCode: http.StatusOK, Body: pr, Header: http.Header{}}
	go func() {
		defer func() { _ = pw.Close() }()
		fmt.Fprintln(pw, `data: {"response":{"candidates":[{"content":{"parts":[{"text":"signed hello","thoughtSignature":"sig-xyz"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":12,"candidatesTokenCount":3}}}`)
		fmt.Fprintln(pw, "")
	}()

	res, err := svc.handleClaudeStreamingResponse(c, resp, time.Now(), "gemini-4-hydrogen")
	_ = pr.Close()

	require.NoError(t, err)
	require.NotNil(t, res)
	require.Equal(t, 12, res.usage.InputTokens)
	require.Equal(t, 3, res.usage.OutputTokens)
	require.Contains(t, rec.Body.String(), "signed hello")
	require.Contains(t, rec.Body.String(), "event: message_stop")
}

func TestAntigravityClaudeStreamBuffer_PreContentKeepaliveGraceAndErrorFallback(t *testing.T) {
	gin.SetMode(gin.TestMode)
	buf := newAntigravityClaudeStreamBuffer()
	start := time.Now().Add(-15 * time.Second)
	lastData := start

	// At 15s (before 25s grace), pre-content keepalive must NOT fire even if interval is 10s
	require.False(t, buf.ShouldSendKeepalive(start.Add(15*time.Second), start, lastData, 10*time.Second))
	// At 25s, pre-content keepalive is allowed
	require.True(t, buf.ShouldSendKeepalive(start.Add(25*time.Second), start, lastData, 10*time.Second))

	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/antigravity/v1/messages", nil)
	flusher := c.Writer.(http.Flusher)
	cw := newAntigravityClientWriter(c.Writer, flusher, "test")
	cw.beforeFirstWrite = func() {
		c.Header("Content-Type", "text/event-stream")
		c.Status(http.StatusOK)
	}

	require.True(t, buf.SendKeepalive(cw))
	require.True(t, buf.preContentKeepaliveSent)
	require.Contains(t, rec.Body.String(), "event: ping")

	// Once pre-content ping committed HTTP 200, EOF without content must send event: error instead of end_turn
	var ft *int
	errSent := ""
	res, err := buf.Finish(cw, antigravity.NewStreamingProcessor("gemini-4-hydrogen"), start, &ft, func(u *antigravity.ClaudeUsage) *ClaudeUsage {
		return &ClaudeUsage{}
	}, func(reason string) {
		errSent = reason
	})
	require.Error(t, err)
	require.True(t, IsAntigravityCompatCommittedNoContent(err), "got %v", err)
	require.NotNil(t, res)
	require.Equal(t, "empty_stream", errSent)
	require.NotContains(t, rec.Body.String(), "end_turn")
}

func TestAntigravityCompatStream_EmptyStreamRetryAndFilterPolicy(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tests := []struct {
		name             string
		sse              string
		wantRetryable    bool
		wantNextAcctStop bool
	}{
		{
			name:             "empty_finish_reason",
			sse:              `data: {"response":{"usageMetadata":{"promptTokenCount":8}}}` + "\n\n",
			wantRetryable:    true,
			wantNextAcctStop: false,
		},
		{
			name:             "malformed_function_call",
			sse:              `data: {"response":{"candidates":[{"content":{"parts":[{}]},"finishReason":"MALFORMED_FUNCTION_CALL"}],"usageMetadata":{"promptTokenCount":8}}}` + "\n\n",
			wantRetryable:    true,
			wantNextAcctStop: false,
		},
		{
			name:             "max_tokens_zero_output_retains_same_account_retry",
			sse:              `data: {"response":{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":37000,"candidatesTokenCount":0,"thoughtsTokenCount":1024}}}` + "\n\n",
			wantRetryable:    true,
			wantNextAcctStop: false,
		},
		{
			name:             "prompt_blocked_safety_stops_immediately",
			sse:              `data: {"response":{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":500,"candidatesTokenCount":0}}}` + "\n\n",
			wantRetryable:    false,
			wantNextAcctStop: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			svc := newAntigravityCompatService(config.GatewayConfig{MaxLineSize: defaultMaxLineSize}, nil)
			c, _ := newAntigravityCompatContext(http.MethodPost, "/v1/chat/completions", nil)
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(io.LimitReader(nil, 0)),
			}
			pr, pw := io.Pipe()
			resp.Body = pr
			go func() {
				defer func() { _ = pw.Close() }()
				_, _ = io.WriteString(pw, tt.sse)
			}()

			res, err := svc.handleChatCompletionsStreamingFromAntigravity(c, resp, time.Now(), "gemini-4-argon-medium", true)
			_ = pr.Close()

			require.Nil(t, res)
			var failoverErr *UpstreamFailoverError
			require.ErrorAs(t, err, &failoverErr)
			require.Equal(t, tt.wantRetryable, failoverErr.RetryableOnSameAccount)
			require.Zero(t, failoverErr.SameAccountRetryMax)
			require.Equal(t, !tt.wantNextAcctStop, failoverErr.ShouldRetryNextAccount())
			require.True(t, failoverErr.RequestScopedTransient)
		})
	}
}

func TestAntigravityNonStream_EmptyAndSignatureOnlyResponsesTriggerFailover(t *testing.T) {
	gin.SetMode(gin.TestMode)
	tests := []struct {
		name             string
		sse              string
		wantRetryable    bool
		wantNextAcctStop bool
	}{
		{
			name:             "empty_stop_frame_without_parts",
			sse:              `data: {"response":{"candidates":[{"finishReason":"STOP"}]}}` + "\n\n",
			wantRetryable:    true,
			wantNextAcctStop: false,
		},
		{
			name:             "max_tokens_hidden_thoughts_without_parts",
			sse:              `data: {"response":{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":34578,"candidatesTokenCount":0,"thoughtsTokenCount":2112}}}` + "\n\n",
			wantRetryable:    true,
			wantNextAcctStop: false,
		},
		{
			name:             "signature_only_part",
			sse:              `data: {"response":{"candidates":[{"content":{"parts":[{"thoughtSignature":"sig-only"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":0}}}` + "\n\n",
			wantRetryable:    true,
			wantNextAcctStop: false,
		},
		{
			name:             "empty_thought_carrier_part",
			sse:              `data: {"response":{"candidates":[{"content":{"parts":[{"thought":true,"text":"","thoughtSignature":"sig-only"}]},"finishReason":"STOP"}]}}` + "\n\n",
			wantRetryable:    true,
			wantNextAcctStop: false,
		},
		{
			name:             "safety_filter_empty_response_stops_immediately",
			sse:              `data: {"response":{"candidates":[{"finishReason":"PROHIBITED_CONTENT"}],"usageMetadata":{"promptTokenCount":500,"candidatesTokenCount":0}}}` + "\n\n",
			wantRetryable:    false,
			wantNextAcctStop: true,
		},
		{
			name:             "prompt_feedback_safety_stops_immediately",
			sse:              `data: {"response":{"promptFeedback":{"blockReason":"SAFETY"}}}` + "\n\n",
			wantRetryable:    false,
			wantNextAcctStop: true,
		},
	}

	for _, tt := range tests {
		for _, endpoint := range []string{"claude", "chat"} {
			t.Run(endpoint+"/"+tt.name, func(t *testing.T) {
				svc := newAntigravityCompatService(config.GatewayConfig{MaxLineSize: defaultMaxLineSize}, nil)
				c, rec := newAntigravityCompatContext(http.MethodPost, "/antigravity/v1/messages", nil)
				resp := &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
					Body:       io.NopCloser(strings.NewReader(tt.sse)),
				}

				var (
					res *antigravityStreamResult
					err error
				)
				if endpoint == "claude" {
					res, err = svc.handleClaudeStreamToNonStreaming(c, resp, time.Now(), "gemini-4-argon-low", 4096)
				} else {
					res, err = svc.handleChatCompletionsNonStreamingFromAntigravity(c, resp, time.Now(), "gemini-4-argon-low", 4096)
				}

				require.Nil(t, res)
				require.Empty(t, rec.Body.String())
				var failoverErr *UpstreamFailoverError
				require.ErrorAs(t, err, &failoverErr)
				require.Equal(t, tt.wantRetryable, failoverErr.RetryableOnSameAccount)
				require.Equal(t, !tt.wantNextAcctStop, failoverErr.ShouldRetryNextAccount())
				require.True(t, failoverErr.RequestScopedTransient)
			})
		}
	}
}

func TestAntigravityNonStream_PreservesTrailingUsageFinishReasonAndGrounding(t *testing.T) {
	gin.SetMode(gin.TestMode)
	sse := strings.Join([]string{
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello "}]}}]}}`,
		"",
		`data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"world"}]}}]}}`,
		"",
		`data: {"response":{"candidates":[{"finishReason":"MAX_TOKENS","groundingMetadata":{"webSearchQueries":["sub2api"]}}],"usageMetadata":{"promptTokenCount":63500,"candidatesTokenCount":42,"thoughtsTokenCount":100}}}`,
		"",
	}, "\n") + "\n"

	t.Run("claude_non_stream", func(t *testing.T) {
		svc := newAntigravityCompatService(config.GatewayConfig{MaxLineSize: defaultMaxLineSize}, nil)
		c, rec := newAntigravityCompatContext(http.MethodPost, "/antigravity/v1/messages", nil)
		resp := &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(sse)),
		}

		res, err := svc.handleClaudeStreamToNonStreaming(c, resp, time.Now(), "gemini-4-argon-low", 4096)
		require.NoError(t, err)
		require.NotNil(t, res)
		require.NotNil(t, res.usage)
		require.Equal(t, 63500, res.usage.InputTokens)
		require.Equal(t, 142, res.usage.OutputTokens)
		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), "Hello world")
		require.Contains(t, rec.Body.String(), "sub2api")
		require.Contains(t, rec.Body.String(), `"stop_reason":"max_tokens"`)
	})

	t.Run("gemini_non_stream", func(t *testing.T) {
		svc := newAntigravityCompatService(config.GatewayConfig{MaxLineSize: defaultMaxLineSize}, nil)
		c, rec := newAntigravityCompatContext(http.MethodPost, "/v1beta/models/gemini-4-hydrogen:generateContent", nil)
		resp := &http.Response{
			StatusCode: http.StatusOK,
			Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
			Body:       io.NopCloser(strings.NewReader(sse)),
		}

		res, err := svc.handleGeminiStreamToNonStreaming(c, resp, time.Now())
		require.NoError(t, err)
		require.NotNil(t, res)
		require.Equal(t, http.StatusOK, rec.Code)
		require.Contains(t, rec.Body.String(), "Hello world")
		require.Contains(t, rec.Body.String(), `"finishReason":"MAX_TOKENS"`)
		require.Contains(t, rec.Body.String(), `"promptTokenCount":63500`)
	})

	t.Run("tiny_max_tokens_probe_passes_through_200", func(t *testing.T) {
		probeSSE := `data: {"response":{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":18,"candidatesTokenCount":0,"thoughtsTokenCount":1}}}` + "\n\n"
		for _, endpoint := range []string{"claude", "chat"} {
			svc := newAntigravityCompatService(config.GatewayConfig{MaxLineSize: defaultMaxLineSize}, nil)
			c, rec := newAntigravityCompatContext(http.MethodPost, "/antigravity/v1/messages", nil)
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader(probeSSE)),
			}

			var (
				res *antigravityStreamResult
				err error
			)
			if endpoint == "claude" {
				res, err = svc.handleClaudeStreamToNonStreaming(c, resp, time.Now(), "gemini-4-hydrogen", 256)
			} else {
				res, err = svc.handleChatCompletionsNonStreamingFromAntigravity(c, resp, time.Now(), "gemini-4-hydrogen", 256)
			}

			require.NoError(t, err)
			require.NotNil(t, res)
			require.Equal(t, 18, res.usage.InputTokens)
			require.Equal(t, 1, res.usage.OutputTokens)
			require.Equal(t, http.StatusOK, rec.Code)
		}
	})
}

func TestCollectClaudeStreamResponse_TinyMaxTokensProbePolicy(t *testing.T) {
	gin.SetMode(gin.TestMode)

	t.Run("probe_max_output_tokens_256_max_tokens_thoughts_256_passes_200", func(t *testing.T) {
		sse := `data: {"response":{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":24,"candidatesTokenCount":0,"thoughtsTokenCount":256}}}` + "\n\n"
		for _, endpoint := range []string{"claude", "chat", "responses"} {
			t.Run(endpoint, func(t *testing.T) {
				svc := newAntigravityCompatService(config.GatewayConfig{MaxLineSize: defaultMaxLineSize}, nil)
				c, rec := newAntigravityCompatContext(http.MethodPost, "/v1/messages", nil)
				resp := &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
					Body:       io.NopCloser(strings.NewReader(sse)),
				}

				var (
					res *antigravityStreamResult
					err error
				)
				switch endpoint {
				case "claude":
					res, err = svc.handleClaudeStreamToNonStreaming(c, resp, time.Now(), "gemini-4-hydrogen", 256)
				case "chat":
					res, err = svc.handleChatCompletionsNonStreamingFromAntigravity(c, resp, time.Now(), "gemini-4-hydrogen", 256)
				case "responses":
					res, err = svc.handleResponsesNonStreamingFromAntigravity(c, resp, time.Now(), "gemini-4-hydrogen", 256)
				}

				require.NoError(t, err)
				require.NotNil(t, res)
				require.Equal(t, 24, res.usage.InputTokens)
				require.Equal(t, 256, res.usage.OutputTokens)
				require.Equal(t, http.StatusOK, rec.Code)
			})
		}
	})

	failoverCases := []struct {
		name            string
		maxOutputTokens int
		sse             string
	}{
		{
			name:            "large_budget_4096_max_tokens_thoughts_200_must_failover",
			maxOutputTokens: 4096,
			sse:             `data: {"response":{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":1200,"candidatesTokenCount":0,"thoughtsTokenCount":200}}}` + "\n\n",
		},
		{
			name:            "unknown_budget_0_max_tokens_thoughts_200_must_failover",
			maxOutputTokens: 0,
			sse:             `data: {"response":{"candidates":[{"finishReason":"MAX_TOKENS"}],"usageMetadata":{"promptTokenCount":1200,"candidatesTokenCount":0,"thoughtsTokenCount":200}}}` + "\n\n",
		},
		{
			name:            "probe_budget_256_stop_thoughts_10_must_failover",
			maxOutputTokens: 256,
			sse:             `data: {"response":{"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":24,"candidatesTokenCount":0,"thoughtsTokenCount":10}}}` + "\n\n",
		},
	}

	for _, tc := range failoverCases {
		t.Run(tc.name, func(t *testing.T) {
			svc := newAntigravityCompatService(config.GatewayConfig{MaxLineSize: defaultMaxLineSize}, nil)
			c, rec := newAntigravityCompatContext(http.MethodPost, "/v1/messages", nil)
			resp := &http.Response{
				StatusCode: http.StatusOK,
				Header:     http.Header{"Content-Type": []string{"text/event-stream"}},
				Body:       io.NopCloser(strings.NewReader(tc.sse)),
			}

			claudeResp, res, err := svc.collectClaudeStreamResponse(c, resp, time.Now(), "gemini-4-hydrogen", tc.maxOutputTokens)
			require.Nil(t, claudeResp)
			require.Nil(t, res)
			require.Empty(t, rec.Body.String())
			var failoverErr *UpstreamFailoverError
			require.ErrorAs(t, err, &failoverErr)
			require.Equal(t, http.StatusBadGateway, failoverErr.StatusCode)
			require.True(t, failoverErr.RetryableOnSameAccount)
		})
	}

	t.Run("extractGeminiMaxOutputTokens", func(t *testing.T) {
		require.Equal(t, 0, extractGeminiMaxOutputTokens(nil))
		require.Equal(t, 0, extractGeminiMaxOutputTokens([]byte(`{}`)))
		require.Equal(t, 0, extractGeminiMaxOutputTokens([]byte(`{"request":{"generationConfig":{"maxOutputTokens":0}}}`)))
		require.Equal(t, 0, extractGeminiMaxOutputTokens([]byte(`{"request":{"generationConfig":{"maxOutputTokens":-1}}}`)))
		require.Equal(t, 0, extractGeminiMaxOutputTokens([]byte(`{"request":{"generationConfig":{"maxOutputTokens":256.5}}}`)))
		require.Equal(t, 0, extractGeminiMaxOutputTokens([]byte(`{"request":{"generationConfig":{"maxOutputTokens":"256"}}}`)))
		require.Equal(t, 256, extractGeminiMaxOutputTokens([]byte(`{"request":{"generationConfig":{"maxOutputTokens":256}}}`)))
		require.Equal(t, 4096, extractGeminiMaxOutputTokens([]byte(`{"request":{"generationConfig":{"maxOutputTokens":4096}}}`)))
	})
}

func TestAntigravityCompatStreamSession_ConsumeTracksDiagnosticsWithoutPollutingUsage(t *testing.T) {
	gin.SetMode(gin.TestMode)
	rec := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(rec)
	c.Request = httptest.NewRequest(http.MethodPost, "/v1/chat/completions", nil)
	writer := newAntigravityClientWriter(c.Writer, c.Writer.(http.Flusher), "test")

	zeroSession := newAntigravityCompatStreamSession(
		"gemini-4-hydrogen",
		time.Time{},
		newAntigravityChatStreamAdapter("gemini-4-hydrogen", true),
		writer,
	)
	require.Equal(t, int64(0), zeroSession.elapsedMs())

	session := newAntigravityCompatStreamSession(
		"gemini-4-hydrogen",
		time.Now().Add(-50*time.Millisecond),
		newAntigravityChatStreamAdapter("gemini-4-hydrogen", true),
		writer,
	)
	require.GreaterOrEqual(t, session.elapsedMs(), int64(0))

	// Non-data lines do not increment seenLines.
	session.consume(": ping")
	session.consume("")
	session.consume("data: [DONE]")
	require.Equal(t, 0, session.seenLines)

	// Frame 1: no usageMetadata -> message_start initializes usage with 0/0.
	session.consume(`data: {"response":{"candidates":[{"content":{"parts":[]}}]}}`)
	require.Equal(t, 1, session.seenLines)
	require.Equal(t, int64(0), session.lastPromptTokens)
	require.Equal(t, int64(0), session.lastCandidatesTokens)
	require.Equal(t, int64(0), session.lastThoughtsTokens)
	require.Equal(t, &ClaudeUsage{}, session.usage)

	// Frame 2: usageMetadata without finishReason -> diagnostics update, s.usage stays untouched.
	session.consume(`data: {"response":{"usageMetadata":{"promptTokenCount":34578,"candidatesTokenCount":0,"thoughtsTokenCount":2112}}}`)
	require.Equal(t, 2, session.seenLines)
	require.Equal(t, int64(34578), session.lastPromptTokens)
	require.Equal(t, int64(0), session.lastCandidatesTokens)
	require.Equal(t, int64(2112), session.lastThoughtsTokens)
	require.Equal(t, &ClaudeUsage{}, session.usage, "diagnostic tracking must not touch or pollute s.usage")

	// Frame 3: blockReason and updated usageMetadata without finishReason.
	session.consume(`data: {"response":{"promptFeedback":{"blockReason":"SAFETY"},"usageMetadata":{"promptTokenCount":34578,"candidatesTokenCount":7,"thoughtsTokenCount":2500}}}`)
	require.Equal(t, 3, session.seenLines)
	require.Equal(t, "SAFETY", session.lastBlockReason)
	require.Equal(t, int64(34578), session.lastPromptTokens)
	require.Equal(t, int64(7), session.lastCandidatesTokens)
	require.Equal(t, int64(2500), session.lastThoughtsTokens)
	require.Equal(t, &ClaudeUsage{}, session.usage, "diagnostic tracking must not touch or pollute s.usage")
}
