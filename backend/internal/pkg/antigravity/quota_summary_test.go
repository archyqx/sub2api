//go:build unit

package antigravity

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// The shape Google returned for a consumer Ultra account on 2026-11-04.
const quotaSummaryBody = `{
  "groups": [
    {"buckets": [
       {"bucketId": "gemini-weekly", "displayName": "Weekly Limit Remaining", "window": "weekly",
        "resetTime": "2026-11-11T15:52:48Z", "remainingFraction": 0.75},
       {"bucketId": "gemini-5h", "displayName": "Five Hour Limit Remaining", "window": "5h",
        "resetTime": "2026-11-04T20:52:48Z", "remainingFraction": 0.4}],
     "displayName": "Gemini Models", "description": "Models within this group: Gemini Flash, Gemini Pro"},
    {"buckets": [
       {"bucketId": "3p-weekly", "window": "weekly", "resetTime": "2026-11-11T15:52:48Z", "remainingFraction": 1},
       {"bucketId": "3p-5h", "window": "5h", "remainingFraction": 1}],
     "displayName": "Claude and GPT models"}
  ],
  "description": "Within each group, models share a weekly limit and a 5-hour limit."
}`

func TestClient_RetrieveUserQuotaSummaryFromURL(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1internal:retrieveUserQuotaSummary" || r.Method != http.MethodPost {
			t.Errorf("unexpected request: %s %s", r.Method, r.URL.Path)
		}
		if r.Header.Get("Authorization") != "Bearer tok" {
			t.Errorf("unexpected authorization header")
		}
		var body FetchAvailableModelsRequest
		if err := json.NewDecoder(r.Body).Decode(&body); err != nil || body.Project != "proj" {
			t.Errorf("unexpected project: %+v, %v", body, err)
		}
		_, _ = w.Write([]byte(quotaSummaryBody))
	}))
	defer server.Close()

	client := mustNewClient(t, "")
	summary, err := client.RetrieveUserQuotaSummaryFromURL(context.Background(), server.URL, "tok", "proj")
	if err != nil {
		t.Fatalf("summary: %v", err)
	}
	if len(summary.Groups) != 2 || summary.Groups[0].DisplayName != "Gemini Models" {
		t.Fatalf("groups: %+v", summary.Groups)
	}
	weekly := summary.Groups[0].Buckets[0]
	if weekly.BucketID != "gemini-weekly" || weekly.Window != "weekly" || weekly.RemainingFraction != 0.75 ||
		weekly.ResetTime != "2026-11-11T15:52:48Z" {
		t.Fatalf("weekly bucket: %+v", weekly)
	}
	if summary.Groups[1].Buckets[1].ResetTime != "" {
		t.Fatalf("a bucket without resetTime keeps it empty")
	}
}

func TestClient_RetrieveUserQuotaSummaryFromURL_Errors(t *testing.T) {
	for _, tc := range []struct {
		status int
		body   string
	}{{http.StatusForbidden, `{"error":{"code":403}}`}, {http.StatusInternalServerError, `{}`}, {http.StatusOK, `not json`}} {
		server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(tc.status)
			_, _ = w.Write([]byte(tc.body))
		}))
		client := mustNewClient(t, "")
		if _, err := client.RetrieveUserQuotaSummaryFromURL(context.Background(), server.URL, "tok", "proj"); err == nil {
			t.Errorf("status %d body %q must be an error", tc.status, tc.body)
		}
		server.Close()
	}
}

func TestClient_WithURLsMethodsAndProjectOmitEmpty(t *testing.T) {
	raw, err := json.Marshal(FetchAvailableModelsRequest{})
	if err != nil {
		t.Fatalf("marshal empty request: %v", err)
	}
	if strings.Contains(string(raw), `"project"`) {
		t.Fatalf("expected empty project to be omitted, got %s", string(raw))
	}

	var hitPaths []string
	daily := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hitPaths = append(hitPaths, "daily:"+r.URL.Path)
		switch r.URL.Path {
		case "/v1internal:fetchAvailableModels":
			_, _ = w.Write([]byte(`{"models":{"gemini-4-hydrogen-high":{"quotaInfo":{"remainingFraction":0.8}}}}`))
		case "/v1internal:loadCodeAssist":
			_, _ = w.Write([]byte(`{"cloudaicompanionProject":"proj-1","paidTier":{"id":"g1-ultra-tier"}}`))
		case "/v1internal:retrieveUserQuotaSummary":
			_, _ = w.Write([]byte(quotaSummaryBody))
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer daily.Close()

	prod := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hitPaths = append(hitPaths, "prod:"+r.URL.Path)
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer prod.Close()

	client := mustNewClient(t, "")
	urls := []string{daily.URL}

	modelsResp, _, err := client.FetchAvailableModelsWithURLs(context.Background(), "tok", "", 1<<20, urls)
	if err != nil || modelsResp.Models["gemini-4-hydrogen-high"].QuotaInfo == nil {
		t.Fatalf("FetchAvailableModelsWithURLs failed: %v, %+v", err, modelsResp)
	}
	loadResp, _, err := client.LoadCodeAssistWithURLs(context.Background(), "tok", urls)
	if err != nil || loadResp.GetTier() != "g1-ultra-tier" {
		t.Fatalf("LoadCodeAssistWithURLs failed: %v, %+v", err, loadResp)
	}
	summary, err := client.RetrieveUserQuotaSummaryWithURLs(context.Background(), "tok", "", urls)
	if err != nil || len(summary.Groups) != 2 {
		t.Fatalf("RetrieveUserQuotaSummaryWithURLs failed: %v, %+v", err, summary)
	}
	for _, hit := range hitPaths {
		if strings.HasPrefix(hit, "prod:") {
			t.Fatalf("unexpected prod hit: %v", hitPaths)
		}
	}
	if len(hitPaths) != 3 {
		t.Fatalf("expected 3 daily hits, got %v", hitPaths)
	}
}
