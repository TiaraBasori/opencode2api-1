package protocol

import (
	"encoding/json"
	"fmt"
	"strings"

	"opencode2api/internal/identity"
)

// Markup tool-call recovery.
//
// A model behind a compatibility lane sometimes writes its tool call into the
// message text instead of returning a structured call: an Anthropic-style
// <invoke>/<parameter> block, the same wrapped in DSML delimiters, or a JSON
// object inside <tool_call>. The client then receives the markup as ordinary
// prose, shows it to the user, and never runs the tool.
//
// markupExtractor recognises those shapes in the text stream and turns them
// back into tool calls. It only acts when the client declared the tool it
// finds, so a model that happens to write XML in prose is left alone, and it
// only holds back text that could still turn out to be markup, so an ordinary
// reply keeps streaming token by token.

// markupOpeners are the beginnings of every supported container. A buffer is
// held only while its trimmed prefix is a prefix of one of these.
var markupOpeners = []string{
	"<tool_calls", "<function_calls", "<tool_call", "<invoke",
	"||DSML||", "|DSML|", "<|DSML|",
}

// markupMaxBuffer caps how much text is held while waiting for a container to
// close, so an unterminated opener cannot stall the stream indefinitely.
const markupMaxBuffer = 16 << 10

// MarkupToolCall is one tool call recovered from text.
type MarkupToolCall struct {
	ID        string
	Name      string
	Arguments string // JSON object text
}

// RecoverToolMarkup parses a complete response text as tool-call markup. It
// reports false when the text is ordinary prose, when it holds no call for a
// tool the client declared, or when no tools were declared at all. prose is
// whatever text preceded the markup and belongs to the reply.
func RecoverToolMarkup(text string, tools []string) ([]MarkupToolCall, string, bool) {
	if len(tools) == 0 || strings.TrimSpace(text) == "" {
		return nil, text, false
	}
	names := make(map[string]bool, len(tools))
	for _, name := range tools {
		if name != "" {
			names[name] = true
		}
	}
	if len(names) == 0 {
		return nil, text, false
	}
	calls, prose, ok := splitMarkupCalls(text, names)
	if !ok || len(calls) == 0 {
		return nil, text, false
	}
	for i := range calls {
		calls[i].ID = newMarkupToolID(i)
	}
	return calls, prose, true
}

// markupExtractor decides, incrementally, which part of the text stream is
// prose and which part is a call block.
type markupExtractor struct {
	names map[string]bool
	buf   strings.Builder
	calls []MarkupToolCall
	done  bool // the buffer overflowed; stop trying
}

// newMarkupExtractor returns nil when the client declared no tools, which
// disables recovery entirely.
func newMarkupExtractor(tools []string) *markupExtractor {
	if len(tools) == 0 {
		return nil
	}
	names := make(map[string]bool, len(tools))
	for _, name := range tools {
		if name != "" {
			names[name] = true
		}
	}
	if len(names) == 0 {
		return nil
	}
	return &markupExtractor{names: names}
}

// Feed consumes a text delta and returns the part of it that must be emitted
// now. Prose is emitted as soon as it cannot be part of a marker, so a normal
// reply keeps streaming; only a suffix that could still grow into an opener is
// held back, and an open container holds until it closes (bounded by
// markupMaxBuffer, so a model that writes an unterminated tag cannot stall the
// stream).
func (x *markupExtractor) Feed(text string) string {
	if x == nil || x.done {
		return text
	}
	x.buf.WriteString(text)
	for {
		buffered := x.buf.String()
		if buffered == "" {
			return ""
		}
		if at, _ := earliestOpener(buffered); at >= 0 {
			if calls, ok := parseMarkupCalls(buffered[at:], x.names); ok {
				x.calls = calls
				x.buf.Reset()
				x.done = true
				return buffered[:at]
			}
			// The opener is there; the block is not complete yet. Emit what
			// precedes it and keep waiting.
			prose := buffered[:at]
			x.buf.Reset()
			x.buf.WriteString(buffered[at:])
			if x.buf.Len() > markupMaxBuffer {
				return prose + x.passThrough()
			}
			return prose
		}
		// No opener. Hold back only the longest suffix that is still a prefix
		// of one, so a marker split across deltas is not emitted as prose.
		hold := 0
		for _, opener := range markupOpeners {
			for n := min(len(opener), len(buffered)); n > hold; n-- {
				if strings.HasPrefix(opener, buffered[len(buffered)-n:]) {
					hold = n
					break
				}
			}
		}
		if hold > 0 && len(buffered) > markupMaxBuffer {
			return x.passThrough()
		}
		prose := buffered[:len(buffered)-hold]
		x.buf.Reset()
		x.buf.WriteString(buffered[len(buffered)-hold:])
		return prose
	}
}

// passThrough gives up on recovery and hands back everything buffered.
func (x *markupExtractor) passThrough() string {
	x.done = true
	out := x.buf.String()
	x.buf.Reset()
	return out
}

// Flush ends the stream: it returns the prose that must still be emitted and
// any recovered tool calls.
func (x *markupExtractor) Flush() (string, []MarkupToolCall) {
	if x == nil {
		return "", nil
	}
	buffered := x.buf.String()
	x.buf.Reset()
	if len(x.calls) > 0 {
		return buffered, x.calls
	}
	if buffered == "" {
		return "", nil
	}
	calls, prose, ok := splitMarkupCalls(buffered, x.names)
	if !ok {
		return buffered, nil
	}
	x.calls = calls
	return prose, calls
}

// earliestOpener returns the position of the first markup opener in text and
// the opener found, or -1.
func earliestOpener(text string) (int, string) {
	best, found := -1, ""
	for _, opener := range markupOpeners {
		if at := strings.Index(text, opener); at >= 0 && (best < 0 || at < best) {
			best, found = at, opener
		}
	}
	return best, found
}

// parseMarkupCalls reports whether the whole buffer is a call container and
// returns the calls it holds.
func parseMarkupCalls(text string, names map[string]bool) ([]MarkupToolCall, bool) {
	calls, prose, ok := splitMarkupCalls(text, names)
	if !ok || strings.TrimSpace(prose) != "" {
		return nil, false
	}
	return calls, true
}

// splitMarkupCalls splits text into the prose before the first container and
// the calls that container holds. ok is false when no complete, recognised
// container is present.
func splitMarkupCalls(text string, names map[string]bool) ([]MarkupToolCall, string, bool) {
	for _, opener := range []string{"<tool_calls", "<function_calls", "||DSML||", "|DSML|", "<|DSML|"} {
		idx := strings.Index(text, opener)
		if idx < 0 {
			continue
		}
		prose := text[:idx]
		rest := text[idx:]
		body, closed := cutContainer(rest)
		if !closed {
			return nil, prose, false
		}
		calls := parseInvokeBlocks(body, names)
		if len(calls) == 0 {
			return nil, prose, false
		}
		return calls, prose, true
	}
	// A bare single <tool_call> block holding JSON.
	idx := strings.Index(text, "<tool_call")
	if idx < 0 {
		return nil, text, false
	}
	prose := text[:idx]
	body, closed := cutContainer(text[idx:])
	if !closed {
		return nil, prose, false
	}
	calls := parseToolCallJSON(body, names)
	if len(calls) == 0 {
		return nil, prose, false
	}
	return calls, prose, true
}

// cutContainer trims the DSML markers down to the inner container and returns
// its body. closed is false when the closing tag is missing.
func cutContainer(text string) (string, bool) {
	for _, marker := range []string{"||DSML||", "|DSML|", "<|DSML|"} {
		text = strings.ReplaceAll(text, marker, "")
	}
	text = strings.TrimSpace(text)
	for _, tag := range []string{"tool_calls", "function_calls", "tool_call"} {
		open := "<" + tag + ">"
		closeTag := "</" + tag + ">"
		start := strings.Index(text, open)
		if start < 0 {
			// tolerate attributes on the opener
			if attrs := strings.Index(text, "<"+tag+" "); attrs >= 0 && strings.Index(text[attrs:], ">") > 0 {
				start = attrs
				open = text[attrs : attrs+strings.Index(text[attrs:], ">")+1]
			} else {
				continue
			}
		}
		end := strings.LastIndex(text, closeTag)
		if end < start {
			return "", false
		}
		return text[start+len(open) : end], true
	}
	return "", false
}

// parseInvokeBlocks reads <invoke name="..."> blocks, each holding <parameter>
// children, and keeps only the calls whose tool the client declared.
func parseInvokeBlocks(body string, names map[string]bool) []MarkupToolCall {
	var calls []MarkupToolCall
	rest := body
	for {
		start := strings.Index(rest, "<invoke")
		if start < 0 {
			break
		}
		head := rest[start:]
		gt := strings.Index(head, ">")
		if gt < 0 {
			break
		}
		name := attributeValue(head[:gt], "name")
		inner := head[gt+1:]
		end := strings.Index(inner, "</invoke>")
		if end < 0 {
			break
		}
		if names[name] {
			args := parametersObject(inner[:end])
			calls = append(calls, MarkupToolCall{Name: name, Arguments: args})
		}
		rest = inner[end+len("</invoke>"):]
	}
	return calls
}

// parseToolCallJSON reads <tool_call>{ "name": ..., "arguments": ...}</tool_call>.
func parseToolCallJSON(body string, names map[string]bool) []MarkupToolCall {
	body = strings.TrimSpace(body)
	// Strip a fenced code block if the model wrapped the JSON.
	if strings.HasPrefix(body, "```") {
		body = strings.TrimPrefix(body, "```json")
		body = strings.TrimPrefix(body, "```")
		body = strings.TrimSuffix(strings.TrimSpace(body), "```")
	}
	var payload struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(body), &payload); err != nil {
		return nil
	}
	if !names[payload.Name] {
		return nil
	}
	args := "{}"
	if len(payload.Arguments) > 0 {
		args = string(payload.Arguments)
		if trimmed := strings.TrimSpace(args); strings.HasPrefix(trimmed, "\"") {
			// arguments carried as a JSON string holding JSON text
			var inner string
			if json.Unmarshal([]byte(trimmed), &inner) == nil {
				args = inner
			}
		}
	}
	return []MarkupToolCall{{Name: payload.Name, Arguments: args}}
}

// parametersObject turns <parameter name="x">value</parameter> children into a
// JSON object. A parameter marked string="false" is decoded as JSON when it
// parses; everything else stays a string, because guessing further turns tool
// arguments into the wrong type.
func parametersObject(body string) string {
	values := map[string]any{}
	rest := body
	for {
		start := strings.Index(rest, "<parameter")
		if start < 0 {
			break
		}
		head := rest[start:]
		gt := strings.Index(head, ">")
		if gt < 0 {
			break
		}
		attrs := head[:gt]
		name := attributeValue(attrs, "name")
		inner := head[gt+1:]
		end := strings.Index(inner, "</parameter>")
		if end < 0 {
			break
		}
		value := inner[:end]
		if name != "" {
			if strings.EqualFold(attributeValue(attrs, "string"), "false") {
				var decoded any
				if json.Unmarshal([]byte(strings.TrimSpace(value)), &decoded) == nil {
					values[name] = decoded
					rest = inner[end+len("</parameter>"):]
					continue
				}
			}
			values[name] = value
		}
		rest = inner[end+len("</parameter>"):]
	}
	if len(values) == 0 {
		return "{}"
	}
	encoded, err := json.Marshal(values)
	if err != nil {
		return "{}"
	}
	return string(encoded)
}

// attributeValue returns the value of a name="value" (or name='value')
// attribute in an opening tag, or "".
func attributeValue(tag, name string) string {
	for _, quote := range []string{"\"", "'"} {
		needle := name + "=" + quote
		idx := strings.Index(tag, needle)
		if idx < 0 {
			continue
		}
		rest := tag[idx+len(needle):]
		end := strings.Index(rest, quote)
		if end < 0 {
			continue
		}
		return rest[:end]
	}
	return ""
}

// markupToolKey namespaces the synthetic tool keys so they cannot collide with
// a key an upstream stream itself uses.
func markupToolKey(index int) string { return fmt.Sprintf("markup-%d", index) }

// newMarkupToolID mints the call id a recovered tool call carries downstream.
func newMarkupToolID(index int) string {
	return fmt.Sprintf("call_%s_%d", identity.RandomID("markup", 8), index)
}

// min returns the smaller of two ints.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// DeclaredToolNames reads the function-tool names a request body offers, in the
// wire shape of any protocol: Chat nests the definition under "function", the
// Responses API keeps it flat, and Anthropic uses "name" with "input_schema".
// Built-in tools that carry no name are skipped, and an unreadable body yields
// no names, which disables recovery rather than guessing.
func DeclaredToolNames(body []byte) []string {
	if len(body) == 0 {
		return nil
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil
	}
	raw, ok := payload["tools"].([]any)
	if !ok {
		return nil
	}
	names := make([]string, 0, len(raw))
	for _, item := range raw {
		tool, ok := item.(map[string]any)
		if !ok {
			continue
		}
		name := ""
		if function, ok := tool["function"].(map[string]any); ok {
			name, _ = function["name"].(string)
		}
		if name == "" {
			name, _ = tool["name"].(string)
		}
		if name != "" {
			names = append(names, name)
		}
	}
	return names
}

// RecoverResponseMarkup recovers a tool call that a model wrote into the text of
// a complete response body whose protocol matches the client's. The body is
// decoded into the bridge model, the text is checked for markup, and the body is
// re-encoded only when a call for a declared tool was actually found — every
// other response keeps its original bytes, because a same-protocol reply is
// otherwise passed through untouched.
func RecoverResponseMarkup(protocol Protocol, body []byte, tools []string) []byte {
	if len(tools) == 0 || len(body) == 0 {
		return body
	}
	var input map[string]any
	if err := json.Unmarshal(body, &input); err != nil {
		return body
	}
	response, err := decodeBridgeResponse(protocol, input)
	if err != nil || response.Error != "" {
		return body
	}
	calls, prose, ok := RecoverToolMarkup(response.Text, tools)
	if !ok {
		return body
	}
	response.Text = prose
	for _, call := range calls {
		response.Tools = append(response.Tools, bridgeBlock{Kind: "tool", ID: call.ID, Name: call.Name, ArgumentsJSON: call.Arguments})
	}
	if !isToolStop(response.Stop) {
		response.Stop = "tool_calls"
	}
	encoded, err := json.Marshal(encodeBridgeResponse(protocol, response))
	if err != nil {
		return body
	}
	return encoded
}
