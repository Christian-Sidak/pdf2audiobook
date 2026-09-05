# Room tone: engineer-grade bed (plan only, not built)

Companion to `prosody_analytics.md`. Both are assembly- or QC-time features:
nothing here re-renders a take.

## What we do today

Takes render dry and are overlaid at fixed offsets onto one continuous tone
bed. The bed is harvested from the voice's tone sidecar or reference by VAD,
with a 500 ms guard from any speech boundary, looped from 60 ms frames chosen
at random, half of them reversed, joined with equal-power crossfades, at a
fixed level (`mastering.room_tone_db`, default -58 dBFS). When the reference
has no harvestable air (every distilled reference), the bed is synthetic
pink-ish noise. Head and tail tone follow ACX. Mastering is two-pass loudnorm
to -18 LUFS with a true-peak ceiling.

Structurally this is how a dialogue editor works. The differences are in
how the bed is derived and matched.

## Feature set

1. **The render is its own room.** Derive the bed from the takes themselves
   rather than from the reference: collect the takes' inter-word and
   inter-sentence gaps (VAD, same guard), which carry the model's actual
   noise floor, and build the bed from that pool. Fallback order: render
   noise print, then tone sidecar, then reference air, then synthetic.
   Fixes the level mismatch and the generic fallback at once.

2. **Level and spectrum matched, not fixed.** Measure the takes' floor
   (RMS and a coarse spectral envelope) and set the bed to match within a
   configured tolerance, instead of a hard-coded dBFS. Keep the config
   value as an override.

3. **No stacked floors.** Under speech, either duck the bed by the
   take's measured floor or subtract a matched noise print from the take's
   gaps so bed plus take equals one floor, not two.

4. **Longer grain.** Loop seconds-long chunks with 100 to 250 ms
   equal-power crossfades when the pool allows, falling back to short
   frames only for thin pools. Kills the granular texture on tone with
   low-frequency movement.

5. **Bed hygiene.** Gentle high-pass on the bed (below about 60 Hz) and a
   spectral sanity check so a harvested pool containing a hum, click, or
   distant voice is rejected rather than looped for nine hours.

6. **Stage-6 QC additions.**
   - `noise_floor`: measured floor of the mastered program against the ACX
     -60 dB RMS target, per chapter.
   - `edge_continuity`: at every take edge, the floor just before and just
     after must agree within a tolerance; a jump means the bed and the take
     are mismatched. This is the check that hears what the ear hears.
   - `bed_periodicity`: autocorrelation of the bed over a long window must
     show no loop peak.

7. **Minimum prosody at assembly and QC.** Per `prosody_analytics.md`: a
   `prosody_collapse` check in stage 5, and at bank time a prosody profile
   per voice. Room tone and prosody are the two halves of "sounds like a
   recording room with a person in it": the bed breathes, the voice moves.

## Order of work

1 and 2 first (they fix the audible problem), then 6's `edge_continuity` to
prove it, then 3 through 5 as polish. Calibrate every tolerance on accepted
renders before enforcing, as with the similarity thresholds.

## How to A/B without cost

`python3 main.py reassemble <book_id>` rebuilds the program from cached
takes in minutes. Assemble one chapter both ways, listen to three take
edges each at high gain, and choose. Never re-render to tune tone.
