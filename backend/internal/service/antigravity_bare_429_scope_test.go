package service

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

type bare429ScopeRepoStub struct {
	AccountRepository
	rateLimitedCalls []int64
	modelLimitCalls  []string
}

func (r *bare429ScopeRepoStub) SetRateLimited(_ context.Context, id int64, _ time.Time) error {
	r.rateLimitedCalls = append(r.rateLimitedCalls, id)
	return nil
}

func (r *bare429ScopeRepoStub) SetModelRateLimit(_ context.Context, _ int64, scope string, _ time.Time, _ ...string) error {
	r.modelLimitCalls = append(r.modelLimitCalls, scope)
	return nil
}

func newBare429TestAccount(planType string) *Account {
	creds := map[string]any{
		"model_mapping": map[string]any{
			"gemini-4-hydrogen-medium": "gemini-4-hydrogen-medium",
			"gemini-4-argon-high":      "gemini-4-argon-high",
		},
	}
	if planType != "" {
		creds["plan_type"] = planType
	}
	return &Account{
		ID:          42,
		Name:        "test-ag-account",
		Platform:    PlatformAntigravity,
		Type:        AccountTypeOAuth,
		Status:      StatusActive,
		Schedulable: true,
		Credentials: creds,
	}
}

func TestHandleUpstreamError_Bare429ScopeAndSingleAccountIsolation(t *testing.T) {
	bare429Body := []byte(`{"error":{"code":429,"message":"Resource has been exhausted (e.g. check quota).","status":"RESOURCE_EXHAUSTED"}}`)
	quota429Body := []byte(`{"error":{"code":429,"message":"Quota exceeded","status":"RESOURCE_EXHAUSTED","details":[{"@type":"type.googleapis.com/google.rpc.ErrorInfo","reason":"RATE_LIMIT_EXCEEDED","metadata":{"quotaResetDelay":"120s"}}]}}`)

	t.Run("multi-account bare 429 isolates cooldown to requested model without gemini family key", func(t *testing.T) {
		repo := &bare429ScopeRepoStub{}
		svc := &AntigravityGatewayService{accountRepo: repo}
		account := newBare429TestAccount("Ultra")

		svc.handleUpstreamError(context.Background(), "[test]", account, http.StatusTooManyRequests, nil, bare429Body, "gemini-4-hydrogen-medium", 0, "", false)

		require.Empty(t, repo.rateLimitedCalls)
		require.Equal(t, []string{"gemini-4-hydrogen-medium"}, repo.modelLimitCalls)
	})

	t.Run("multi-account bare 429 on pro argon isolates cooldown without argon family key", func(t *testing.T) {
		repo := &bare429ScopeRepoStub{}
		svc := &AntigravityGatewayService{accountRepo: repo}
		account := newBare429TestAccount("Pro")

		svc.handleUpstreamError(context.Background(), "[test]", account, http.StatusTooManyRequests, nil, bare429Body, "gemini-4-argon-high", 0, "", false)

		require.Empty(t, repo.rateLimitedCalls)
		require.Equal(t, []string{"gemini-4-argon-high"}, repo.modelLimitCalls)
	})

	t.Run("multi-account 429 with quotaResetDelay still cools model and family key", func(t *testing.T) {
		repo := &bare429ScopeRepoStub{}
		svc := &AntigravityGatewayService{accountRepo: repo}
		account := newBare429TestAccount("Ultra")

		svc.handleUpstreamError(context.Background(), "[test]", account, http.StatusTooManyRequests, nil, quota429Body, "gemini-4-hydrogen-medium", 0, "", false)

		require.Empty(t, repo.rateLimitedCalls)
		require.Equal(t, []string{"gemini-4-hydrogen-medium", antigravityGeminiModelRateLimitKey}, repo.modelLimitCalls)
	})

	t.Run("single-account bare 429 skips cooldown completely", func(t *testing.T) {
		for name, ctx := range map[string]context.Context{
			"explicit": WithSingleAccountRetry(context.Background(), true, false),
			"probe":    WithSingleAccountRetryProbe(context.Background(), func(context.Context) bool { return true }),
		} {
			t.Run(name, func(t *testing.T) {
				repo := &bare429ScopeRepoStub{}
				svc := &AntigravityGatewayService{accountRepo: repo}
				account := newBare429TestAccount("Ultra")

				svc.handleUpstreamError(ctx, "[test]", account, http.StatusTooManyRequests, nil, bare429Body, "gemini-4-hydrogen-medium", 0, "", false)

				require.Empty(t, repo.rateLimitedCalls)
				require.Empty(t, repo.modelLimitCalls)
			})
		}
	})

	t.Run("single-account bare 429 with empty model key skips fallback account cooldown", func(t *testing.T) {
		repo := &bare429ScopeRepoStub{}
		svc := &AntigravityGatewayService{accountRepo: repo}
		account := newBare429TestAccount("Ultra")
		ctx := WithSingleAccountRetryProbe(context.Background(), func(context.Context) bool { return true })

		svc.handleUpstreamError(ctx, "[test]", account, http.StatusTooManyRequests, nil, bare429Body, "", 0, "", false)

		require.Empty(t, repo.rateLimitedCalls)
		require.Empty(t, repo.modelLimitCalls)
	})

	t.Run("single-account 429 with quotaResetDelay still applies cooldown", func(t *testing.T) {
		repo := &bare429ScopeRepoStub{}
		svc := &AntigravityGatewayService{accountRepo: repo}
		account := newBare429TestAccount("Ultra")
		ctx := WithSingleAccountRetryProbe(context.Background(), func(context.Context) bool { return true })

		svc.handleUpstreamError(ctx, "[test]", account, http.StatusTooManyRequests, nil, quota429Body, "gemini-4-hydrogen-medium", 0, "", false)

		require.Empty(t, repo.rateLimitedCalls)
		require.Equal(t, []string{"gemini-4-hydrogen-medium", antigravityGeminiModelRateLimitKey}, repo.modelLimitCalls)
	})
}
