# Prosody analytics (plan only, not built)

Goal: measure expressiveness so the pipeline can enforce "sufficient prosody"
at two points, when a voice reference is banked and when takes render, the
way it already enforces transcript fidelity and speaker identity.

## Why

- A clone reproduces the reference's prosody. A flat reference makes a flat
  book; a wildly expressive one (in-character dialogue) makes an unstable one.
  Today `audit_reference` computes one number, pitch spread, and uses it only
  as a second-speaker alarm.
- Zero-shot TTS collapses to monotone on some takes (long paragraphs,
  lists, quoted speech). Transcript and similarity checks are deaf to this:
  same words, same speaker, no life. Nothing in stage 5 catches it.
- Casting (see `../audiblameo`) needs a way to describe a voice's register
  in numbers so books can be matched to narrators.

## Features per utterance

Computed with Praat-style pitch tracking (parselmouth or pyworld), on voiced
frames only, all in speaker-relative units so voices with different base
pitch compare fairly:

| Feature | What it captures | Unit |
|---|---|---|
| F0 range | melodic span | semitones, 5th to 95th percentile |
| F0 std | overall variability | semitones |
| F0 slope per phrase | declination and question rises | semitones per second |
| Pitch-accent rate | how often stressed syllables get a pitch move | accents per second |
| Energy dynamic range | loudness contrast | dB, 5th to 95th percentile |
| Speech rate and its variance | pace and how much it changes | syllables per second, CV |
| Pause profile | count, mean, distribution of pauses | per minute, seconds |
| Final lengthening | phrase-final slowing, a marker of natural phrasing | ratio |
| Voice quality proxies | breathiness, creak (HNR, jitter, shimmer) | dimensionless |

Bundle these into a **prosody vector** per take and per reference.

## Two gates

**At bank time (grab, distill).** Compute the vector for the reference and
report it next to the pitch-spread audit. Warn below a floor (monotone, e.g.
F0 std under about two semitones for English narration) and above a ceiling
(performance register). Store the vector with the voice as its
**prosody profile**. Distill candidates are ranked by closeness to the real
reference's profile, not just similarity of timbre.

**At render (stage 5).** A new check, `prosody_collapse`, deterministic:

1. Compute each take's vector.
2. Compare to the render's own median profile (the centroid trick from
   `speaker_similarity`), which beats comparing to the reference because a
   reference is one utterance and a book is thousands.
3. Flag takes whose F0 std or energy range falls below a fraction of the
   median (start at 0.5) and whose duration is over about three seconds
   (short takes are legitimately flat). Also flag the reverse: a take far
   above the median is likely a collapse into shouting or a hallucinated
   register.
4. Flags are `fixable`, so they re-roll like any other take failure.

The LLM stays out of the measurement (this is acoustics) but adjudicates the
edge: a flagged take whose text is a list, a citation, or a heading is
expected to be flat, and the judge can waive it from the text alone.

## Analytics, not just gates

- Per-book prosody report: distribution of F0 std and speech rate by
  chapter, so a book that goes flat in the middle is visible on one chart.
- Per-voice profile card for the roster: register, range, pace, pause
  style. Casting compares a manuscript's needs (intimate, brisk, formal) to
  these.
- Reference-to-render drift: the render's median profile versus the
  reference's, to see what the clone keeps and loses.

## Calibration before enforcing anything

Measure first on renders already accepted by ear (Tsavo, Velveteen Rabbit,
Carnegie when it lands), and on the banked references, to learn the actual
distribution before choosing thresholds. Whisper and the judge were tuned the
same way. Thresholds set from theory will flag good narration.

## Open questions

- Pitch tracking on cloned audio at 24 kHz: verify tracker agreement
  (parselmouth vs pyworld) on a few takes before trusting either.
- Where flatness is right: dry academic prose, tables read aloud, epigraphs.
  The waiver path above handles it, but the rate of waivers is itself worth
  tracking.
- Whether a prosody floor belongs in the voice preflight canaries too. Cheap
  to add once the vector exists.
