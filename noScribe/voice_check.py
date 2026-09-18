"""Check the speaker of each passage against the voice itself.

A transcript segment gets the speaker whose diarization turns overlap it most.
That goes wrong in two ways: a segment that straddles a turn hands the shorter
half to the wrong speaker, and the diarization itself mislabels stretches. Both
are repaired here by a second opinion the diarization never had -- the speaker
embedding of exactly the audio a passage of *text* covers, compared with the
diarization's own speaker centroids.

    units      a segment is cut where a speaker can change: after a sentence
               end, or at a pause of UNIT_PAUSE_S between two words
    voice      one embedding per unit, from pyannote's own model (a second,
               short call to pyannote_mp_worker once the transcript exists)
    decision   a unit moves to another speaker when its voice is closer to that
               speaker's centroid than to the current one: by MARGIN_AGREE when
               the diarization's turns inside the unit name that speaker too, by
               MARGIN_OVERRULE when the voice stands alone

Measured against ground truth at word level (does each word carry the right
speaker?), thresholds chosen on 16 AMI meetings, 16 CallHome German calls and
two spliced conversations, then frozen and checked on material never looked at
before -- 7 more AMI meetings, 61 CallHome calls in German, English, Spanish,
Japanese and Mandarin, 24 VoxConverse recordings, about 150 000 words per engine:

                         wrong words      repaired / broken
    faster-whisper       8036 -> 4774        3408 / 146   (96 % right)
    Voxtral              6701 -> 3699        3198 / 196   (94 % right)

Every language and every corpus is a net gain (87 % to 99.6 % right). What the
rule breaks sits in a few recordings whose diarization is badly off to begin
with, and those still come out ahead.

What was measured and left out, because it added nothing worth its weight:
cutting at speaker changes *without* the voice (on Whisper segments it breaks
about as many words as it repairs: 182 / 58 on AMI, 42 / 31 on Spanish calls);
assigning every word on its own, with or without smoothing (cuts inside a
phrase); smoothing the decisions over neighbouring units (a conversation is not
sticky: it broke more than it repaired); a minimum duration, a longer audio
crop, thresholds that depend on duration or on the diarization's share,
centroids rebuilt from long clean turns, pyannote's exclusive diarization or its
soft activations as the basis, CTC word stamps for Whisper, a guard for similar
voices, and separate thresholds per engine (the optimum is flat and shared).
Scoring in the pipeline's PLDA space (as a log-likelihood ratio or as a cosine)
reaches the same repairs at the same precision and no more; demanding that a
wider crop agrees lowers both; centroids rebuilt once from the confidently
scored units repair about 2 % more words -- real, and not worth a second pass.

The units themselves are not the limit: with perfect labels per unit 0.6 % of
the words would still be wrong, against 5.3 % today and 3.1 % with this rule.
"""

# Word endings that close a sentence, ignoring trailing quotes and brackets.
# Deliberately no colon: a colon lands mid-utterance ("And then he said: ...").
_SENTENCE_END = ('.', '!', '?', '…', '。', '！', '？')
_SENTENCE_TRAIL = '"\'»)] '

# A pause between two words that ends a unit even without punctuation. 0.3 s and
# 0.8 s score within a point of this; 0.25 s cut inside phrases on real speech.
UNIT_PAUSE_S = 0.5

# Cosine margins. 0.1 / 0.3 sit on a flat optimum: 0.0 / 0.25 repair about a
# tenth more words at one to two points less precision, 0.2 / 0.4 the reverse.
MARGIN_AGREE = 0.1
MARGIN_OVERRULE = 0.3


def split_units(words):
    """Group a segment's words into units. Returns a list of word lists.

    Words without both time stamps give no basis for a cut: the segment stays
    whole.
    """
    if not words:
        return []
    if any(w.get('start') is None or w.get('end') is None for w in words):
        return [list(words)]
    units, current = [], []
    for i, word in enumerate(words):
        current.append(word)
        token = (word.get('word') or '').rstrip(_SENTENCE_TRAIL)
        following = words[i + 1] if i + 1 < len(words) else None
        if token.endswith(_SENTENCE_END) or (
                following is not None and following['start'] - word['end'] >= UNIT_PAUSE_S):
            units.append(current)
            current = []
    if current:
        units.append(current)
    return units


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na > 0 and nb > 0 else None


def decide(current, turns_ms, embedding, centroids, single_unit=False):
    """The speaker label a unit should carry.

    current      label the unit has now (its segment's speaker)
    turns_ms     {label: milliseconds of that label's turns inside the unit}
    embedding    the unit's voice, or None when it could not be computed
    centroids    {label: centroid}
    single_unit  the unit is its whole segment: the diarization already had its
                 say on exactly this span, so only the voice alone can move it
    """
    if not embedding or current not in centroids:
        return current
    scores = {}
    for label, centroid in centroids.items():
        score = _cosine(embedding, centroid)
        if score is not None:
            scores[label] = score
    if current not in scores:
        return current
    label = current
    if not single_unit and turns_ms:
        named = max(turns_ms, key=turns_ms.get)
        if named != label and named in scores and scores[named] - scores[label] > MARGIN_AGREE:
            label = named
    best = max(scores, key=scores.get)
    if best != label and scores[best] - scores[label] > MARGIN_OVERRULE:
        label = best
    return label


def turns_inside(diarization, start_ms, end_ms):
    """Milliseconds of each label's turns inside [start_ms, end_ms]."""
    totals = {}
    for turn in diarization:
        if turn['start'] > end_ms:
            break
        inside = min(turn['end'], end_ms) - max(turn['start'], start_ms)
        if inside > 0:
            totals[turn['label']] = totals.get(turn['label'], 0) + inside
    return totals


def enabled():
    """NOSCRIBE_VOICE_CHECK=0 restores the plain overlap assignment."""
    import os
    return os.environ.get('NOSCRIBE_VOICE_CHECK', '1').strip().lower() not in ('0', 'false', 'no', 'off')


def _join_words(words):
    """Rebuild a passage's text from its words.

    faster-whisper's words carry their own leading space; another engine's word
    list may be bare tokens, so the separator follows the source.
    """
    tokens = [w.get('word') or '' for w in words]
    separator = '' if any(t.startswith((' ', '\u00a0')) for t in tokens) else ' '
    return ' ' + separator.join(tokens).strip()


def relabel(segments, diarization, centroids, embed):
    """Decide every segment's passages. Returns (passages, changed).

    segments     [(segment dict, label it was given)] in transcript order
    diarization  [{'start': ms, 'end': ms, 'label': str}], sorted by start
    centroids    {label: centroid}, labels spelled as in `segments`
    embed        callable: [[start_s, end_s]] -> [embedding or None]

    passages is [(segment dict, label or None, label it was given)]: None leaves
    the passage to the ordinary assignment, so an untouched segment is written
    exactly as before (the third item is what it falls back on where the
    diarization is silent). changed counts the units that moved.
    """
    if len(centroids) < 2:
        return [(segment, None, current) for segment, current in segments], 0
    plan, spans = [], []
    for segment, current in segments:
        units = split_units(segment.get('words')) if current in centroids else []
        plan.append((segment, current, units))
        spans += [[unit[0]['start'], unit[-1]['end']] for unit in units]
    embeddings = iter(embed(spans) if spans else [])

    passages, changed = [], 0
    for segment, current, units in plan:
        labels = []
        for unit in units:
            start_ms, end_ms = round(unit[0]['start'] * 1000), round(unit[-1]['end'] * 1000)
            labels.append(decide(current, turns_inside(diarization, start_ms, end_ms),
                                 next(embeddings, None), centroids, single_unit=len(units) == 1))
        moved = sum(1 for label in labels if label != current)
        if not moved:
            passages.append((segment, None, current))
            continue
        changed += moved
        runs = []
        for unit, label in zip(units, labels):
            if runs and runs[-1][0] == label:
                runs[-1][1].extend(unit)
            else:
                runs.append((label, list(unit)))
        for label, words in runs:
            whole = len(runs) == 1
            passages.append((segment if whole else {
                'start': words[0]['start'], 'end': words[-1]['end'],
                'text': _join_words(words), 'words': words}, label, current))
    return passages, changed
