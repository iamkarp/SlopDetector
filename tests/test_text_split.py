from __future__ import annotations

from slopdetector.text_split import split_paragraphs, split_sentences, split_sentences_with_spans


def test_split_sentences_handles_normal_spaced_prose():
    text = "First sentence. Second sentence. Third sentence."
    assert split_sentences(text) == ["First sentence.", "Second sentence.", "Third sentence."]


def test_split_sentences_handles_missing_space_after_period():
    # Real bug found scanning pasted/scraped text: "sidewalk.It is important"
    # has zero whitespace between sentences, and the old \s+-based regex
    # refused to split without at least one space to consume.
    text = "He walked down the sidewalk.It is important to remember things.When going out, plan ahead."
    sentences = split_sentences(text)
    assert sentences == [
        "He walked down the sidewalk.",
        "It is important to remember things.",
        "When going out, plan ahead.",
    ]


def test_split_sentences_with_spans_handles_missing_space_too():
    text = "First one.Second one."
    result = split_sentences_with_spans(text)
    sentences = [s for s, _start, _end in result]
    assert sentences == ["First one.", "Second one."]


def test_split_sentences_with_spans_offsets_are_correct_with_no_space():
    text = "First one.Second one."
    result = split_sentences_with_spans(text)
    for sent, start, end in result:
        assert text[start:end].strip() == sent


def test_split_sentences_does_not_split_mid_word_or_on_decimals():
    # Not every period is a sentence boundary; this regex only ever
    # considers [.!?] followed by an uppercase/digit/quote as a candidate,
    # so "3.14" and "Mr. Smith" mostly survive (a known, documented
    # limitation, not something this fix should make worse).
    text = "The value was 3.14 before rounding."
    assert split_sentences(text) == [text]


def test_split_sentences_does_not_split_currency_with_cents():
    text = "The total came to $5.10 for the milk. He paid with a ten."
    assert split_sentences(text) == ["The total came to $5.10 for the milk.", "He paid with a ten."]


def test_split_paragraphs_unaffected_by_sentence_regex_change():
    text = "Paragraph one line.\n\nParagraph two line."
    units = split_paragraphs(text)
    assert [u.text for u in units] == ["Paragraph one line.", "Paragraph two line."]
