package antigravity

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestStreamingProcessorResponseShapesPreserveContent(t *testing.T) {
	for _, tc := range []struct {
		name    string
		parts   string
		markers []string
	}{
		{
			name:    "text",
			parts:   `[{"text":"你好，请把研究方案发给我。"}]`,
			markers: []string{"text_delta", "你好，请把研究方案发给我。"},
		},
		{
			name:    "thinking",
			parts:   `[{"thought":true,"text":"先整理研究问题。"}]`,
			markers: []string{"thinking_delta", "先整理研究问题。"},
		},
		{
			name:    "tool_call",
			parts:   `[{"functionCall":{"name":"read_file","args":{"path":"notes.md"}}}]`,
			markers: []string{"tool_use", "read_file", "notes.md"},
		},
	} {
		for _, shape := range []string{"direct", "wrapped"} {
			t.Run(tc.name+"/"+shape, func(t *testing.T) {
				body := `{"candidates":[{"content":{"role":"model","parts":` + tc.parts + `},"finishReason":"STOP"}],` +
					`"usageMetadata":{"promptTokenCount":100,"cachedContentTokenCount":30,"candidatesTokenCount":20,"thoughtsTokenCount":5},"responseId":"response-shape"}`
				if shape == "wrapped" {
					body = `{"response":` + body + `}`
				}
				p := NewStreamingProcessor("gemini-model")
				events := p.ProcessLine("data: " + body)
				finalEvents, usage := p.Finish()
				events = append(events, finalEvents...)
				if !p.HasContent() {
					t.Fatalf("valid %s response lost its content: %s", shape, events)
				}
				for _, marker := range append(tc.markers, "response-shape", "message_stop") {
					if !strings.Contains(string(events), marker) {
						t.Errorf("missing %q in events: %s", marker, events)
					}
				}
				if usage.InputTokens != 70 || usage.CacheReadInputTokens != 30 || usage.OutputTokens != 25 {
					t.Errorf("usage lost: %+v", usage)
				}
			})
		}
	}
}

func TestStreamingProcessorResponseShapesPreserveUsageWithoutContent(t *testing.T) {
	for _, shape := range []string{"direct", "wrapped"} {
		t.Run(shape, func(t *testing.T) {
			body := `{"usageMetadata":{"promptTokenCount":123,"candidatesTokenCount":0}}`
			if shape == "wrapped" {
				body = `{"response":` + body + `}`
			}
			p := NewStreamingProcessor("gemini-model")
			p.ProcessLine("data: " + body)
			_, usage := p.Finish()
			if usage.InputTokens != 123 || usage.OutputTokens != 0 {
				t.Fatalf("usage lost: %+v", usage)
			}
			if p.HasContent() {
				t.Fatal("usage-only response must not become content")
			}
		})
	}
}

func TestStreamingProcessor_SignedTextAndGroundingMarkHasContent(t *testing.T) {
	t.Run("signed non-empty text marks HasContent", func(t *testing.T) {
		p := NewStreamingProcessor("gemini-4-hydrogen")
		line := `data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"signed reply","thoughtSignature":"sig-123"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4}}}`
		events := p.ProcessLine(line)
		finalEvents, _ := p.Finish()
		events = append(events, finalEvents...)
		require.True(t, p.HasContent())
		require.Contains(t, string(events), "signed reply")
		require.Contains(t, string(events), "sig-123")
	})

	t.Run("grounding-only with finishReason marks HasContent", func(t *testing.T) {
		p := NewStreamingProcessor("gemini-4-hydrogen")
		line := `data: {"response":{"candidates":[{"groundingMetadata":{"webSearchQueries":["sub2api"]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4}}}`
		events := p.ProcessLine(line)
		finalEvents, _ := p.Finish()
		events = append(events, finalEvents...)
		require.True(t, p.HasContent())
		require.Contains(t, string(events), "sub2api")
	})

	t.Run("grounding-only without finishReason marks HasContent on Finish", func(t *testing.T) {
		p := NewStreamingProcessor("gemini-4-hydrogen")
		line := `data: {"response":{"candidates":[{"groundingMetadata":{"webSearchQueries":["golang"]}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":4}}}`
		events := p.ProcessLine(line)
		require.False(t, p.HasContent())
		finalEvents, _ := p.Finish()
		events = append(events, finalEvents...)
		require.True(t, p.HasContent())
		require.Contains(t, string(events), "golang")
	})

	t.Run("empty text with signature does not mark HasContent", func(t *testing.T) {
		p := NewStreamingProcessor("gemini-4-hydrogen")
		line := `data: {"response":{"candidates":[{"content":{"role":"model","parts":[{"text":"","thoughtSignature":"sig-only"}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":0}}}`
		_ = p.ProcessLine(line)
		_, _ = p.Finish()
		require.False(t, p.HasContent())
	})
}

func TestNonStreamingProcessor_FlushesPlainTextBeforeSignedText(t *testing.T) {
	raw := []byte(`{
		"response": {
			"responseId": "resp_order_1",
			"candidates": [{
				"content": {
					"role": "model",
					"parts": [
						{"text": "First chunk, "},
						{"text": "second chunk.", "thoughtSignature": "sig-tail"}
					]
				},
				"finishReason": "STOP"
			}],
			"usageMetadata": {"promptTokenCount": 20, "candidatesTokenCount": 8}
		}
	}`)

	claudeBytes, usage, err := TransformGeminiToClaude(raw, "gemini-4-hydrogen")
	require.NoError(t, err)
	require.Equal(t, 20, usage.InputTokens)
	require.Equal(t, 8, usage.OutputTokens)

	var resp ClaudeResponse
	require.NoError(t, json.Unmarshal(claudeBytes, &resp))
	require.Len(t, resp.Content, 3)
	require.Equal(t, "text", resp.Content[0].Type)
	require.Equal(t, "First chunk, ", resp.Content[0].Text)
	require.Equal(t, "text", resp.Content[1].Type)
	require.Equal(t, "second chunk.", resp.Content[1].Text)
	require.Equal(t, "thinking", resp.Content[2].Type)
	require.Equal(t, "", resp.Content[2].Thinking)
	require.Equal(t, "sig-tail", resp.Content[2].Signature)
}
