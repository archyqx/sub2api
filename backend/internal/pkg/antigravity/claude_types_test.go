package antigravity

import (
	"encoding/json"
	"testing"
)

func TestDefaultModels_ContainsNewAndLegacyImageModels(t *testing.T) {
	t.Parallel()

	models := DefaultModels()
	byID := make(map[string]ClaudeModel, len(models))
	for _, m := range models {
		byID[m.ID] = m
	}

	requiredIDs := []string{
		"claude-fable-5-1",
		"claude-fable-5",
		"claude-opus-4-8",
		"claude-opus-4-6-thinking",
		"gemini-2.5-flash-image",
		"gemini-2.5-flash-image-preview",
		"gemini-3.1-flash-image",
		"gemini-3.1-flash-image-preview",
		"gemini-3-pro-image", // legacy compatibility
		"gemini-3.6-flash",
		"gemini-3.6-flash-high",
		"gemini-3.6-flash-low",
		"gemini-3.6-flash-medium",
		"gemini-3.6-flash-tiered",
		"gemini-3.7-flash",
		"gemini-3.7-flash-high",
		"gemini-3.7-flash-low",
		"gemini-3.7-flash-medium",
		"gemini-3.7-flash-tiered",
		"gemini-3.8-flash",
		"gemini-3.8-flash-high",
		"gemini-3.8-flash-low",
		"gemini-3.8-flash-medium",
		"gemini-3.8-flash-tiered",
		"gemini-4-argon-high",
		"gemini-4-argon-low",
		"gemini-4-argon-medium",
		"claude-opus-5-5-high",
		"claude-opus-5-5-low",
		"claude-opus-5-5-medium",
		"claude-sonnet-5-5-high",
		"claude-sonnet-5-5-low",
		"claude-sonnet-5-5-medium",
	}

	for _, id := range requiredIDs {
		if _, ok := byID[id]; !ok {
			t.Fatalf("expected model %q to be exposed in DefaultModels", id)
		}
	}
}

func TestClaudeContentItem_ThinkingAlwaysHasSignature(t *testing.T) {
	t.Parallel()

	data, err := json.Marshal(ClaudeContentItem{Type: "thinking", Thinking: "plan"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(data), `{"thinking":"plan","signature":"","type":"thinking"}`; got != want {
		t.Fatalf("thinking block = %s, want %s", got, want)
	}

	data, err = json.Marshal(ClaudeContentItem{Type: "text", Text: "hi"})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := string(data), `{"type":"text","text":"hi"}`; got != want {
		t.Fatalf("text block = %s, want %s", got, want)
	}
}

func TestIsGeminiReasoningModel_IncludesProAgent(t *testing.T) {
	t.Parallel()

	reasoningModels := []string{
		"gemini-pro-agent",
		"gemini-2.5-pro",
		"gemini-2.5-pro-preview-06-05",
		"gemini-2.5-flash-thinking",
		"gemini-3-flash",
		"gemini-3-pro-low",
		"gemini-3-pro-high",
		"gemini-3.1-pro",
		"gemini-3.1-pro-low",
		"gemini-3.1-pro-high",
		"gemini-3.8-flash",
		"gemini-3.8-flash-low",
		"gemini-3.8-flash-medium",
		"gemini-3.8-flash-high",
		"gemini-4-argon",
		"gemini-4-argon-high",
		"gemini-4-hydrogen",
		"gemini-4-hydrogen-low",
		"gemini-4-hydrogen-medium",
		"gemini-4-hydrogen-high",
		"gemini-4-carbon",
		"gemini-4-carbon-high",
		"models/gemini-4-hydrogen-medium",
		"GEMINI-4-HYDROGEN",
	}
	for _, m := range reasoningModels {
		if !IsGeminiReasoningModel(m) {
			t.Fatalf("expected %q to be recognized as a Gemini reasoning model", m)
		}
	}

	nonReasoningModels := []string{
		"gemini-2.5-flash",
		"gemini-2.5-flash-lite",
		"gemini-2.5-flash-image",
		"gemini-2.5-flash-image-preview",
		"gemini-3.1-flash-image",
		"gemini-3.1-flash-image-preview",
		"gemini-3-pro-image",
		"gemini-4-image-preview",
		"claude-opus-4-6-thinking",
		"claude-sonnet-4-6",
		"claude-3-5-sonnet-latest",
	}
	for _, m := range nonReasoningModels {
		if IsGeminiReasoningModel(m) {
			t.Fatalf("did not expect %q to be recognized as a Gemini reasoning model", m)
		}
	}
}
