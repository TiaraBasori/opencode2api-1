package identity

import (
	"regexp"
	"testing"
)

func TestCanonicalSessionIDKeepsGenuineDescendingIDs(t *testing.T) {
	// Time parts the official client mints via Identifier.descending() with
	// counters 1, 2, 16, 48: low 12 hex digits 0xFFF - counter.
	genuine := []string{
		"ses_ecc5beb7cffe2U0aluUnb5nnpZ",
		"ses_ecc441adcffep73ndhq1lBqQim",
		"ses_f02a5a612ffepq6RK15QX04Se0",
		"ses_4892557ffffeDiICzp8gKhMR0u",
		"ses_ecc1f76c9fef3zWGdD1GSCEwPf",
		"ses_ecc1f76c9fcf3zWGdD1GSCEwPf",
		"ses_ecc1f76c9fce3zWGdD1GSCEwPf",
	}
	for _, id := range genuine {
		if got := CanonicalSessionID(id); got != id {
			t.Fatalf("CanonicalSessionID(%q) = %q, want unchanged", id, got)
		}
	}
}

func TestCanonicalSessionIDReencodesForeignEncodings(t *testing.T) {
	// Right shape, wrong encoding: ascending ID (001 suffix), counter 100
	// (9b1 suffix), and the raw-hash layout the gateway used to emit.
	foreign := []string{
		"ses_133a414840012U0aluUnb5nnpZ",
		"ses_ecc1f76c9f9b3zWGdD1GSCEwPf",
		"ses_088a704c30424N5kWDK0snKFbd",
	}
	pattern := regexp.MustCompile(`^ses_[0-9a-f]{12}[0-9A-Za-z]{14}$`)
	for _, id := range foreign {
		got := CanonicalSessionID(id)
		if !pattern.MatchString(got) {
			t.Fatalf("CanonicalSessionID(%q) = %q, not canonical shape", id, got)
		}
		if !descendingTimePartPattern.MatchString(got[4:16]) {
			t.Fatalf("CanonicalSessionID(%q) = %q, time part not descending-encoded", id, got)
		}
		if got == id {
			t.Fatalf("CanonicalSessionID(%q) kept a foreign encoding", id)
		}
	}
}

func TestCanonicalSessionIDDeterministic(t *testing.T) {
	a := CanonicalSessionID("uuid-4f1d2a90-1234-abcd-5678-9abcdef01234")
	b := CanonicalSessionID("uuid-4f1d2a90-1234-abcd-5678-9abcdef01234")
	if a != b {
		t.Fatalf("not deterministic: %q vs %q", a, b)
	}
	if !descendingTimePartPattern.MatchString(a[4:16]) {
		t.Fatalf("re-encoded id %q lacks a descending time part", a)
	}
}

func TestDescendingTimePartLayout(t *testing.T) {
	// hash[0:5] = 00 00 00 00 01 -> 36-bit stamp 0x00000001 -> low 12 bits
	// 0xFFF ^ 0x001 ... counter 1 ORed before the NOT: current = 0x1001,
	// inverted low 12 bits = 0xFFE.
	got := descendingTimePart([]byte{0, 0, 0, 0, 1, 0xFF})
	if got[9:] != "ffe" {
		t.Fatalf("descendingTimePart = %q, want suffix ffe", got)
	}
}
