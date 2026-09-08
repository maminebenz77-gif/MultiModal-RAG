# Chunking Strategy Field Notes

Working notes from comparing chunking approaches before landing on the
parent-child strategy used in production.

## Fixed-size chunking

Splits text into equal-length windows, optionally with overlap. It's the
fastest strategy to compute -- no parsing or embedding required, just
character counting -- but it frequently splits sentences, table rows, or
list items mid-way, since it has no awareness of document structure at
all. Downstream generation quality suffers noticeably whenever a cut
lands in the middle of the one sentence that actually answers the
question.

## Semantic chunking

Groups text by detecting topic-shift boundaries between adjacent sentences
or paragraphs, using embedding similarity to decide where one "topic"
ends and the next begins. This respects meaning far better than a fixed
character count, but it's the most expensive strategy of the three to
compute, since it requires embedding many intermediate windows just to
find the boundaries, before any of the resulting chunks are embedded for
retrieval.

## Parent-child chunking

Uses small child chunks for the actual similarity search (precise
matching against a short, focused span of text), but substitutes each
matched child's full parent section back in as the context handed to the
generator. This solves the classic tension where small chunks retrieve
accurately but lack surrounding context, and large chunks have context
but retrieve imprecisely.

In an internal comparison against flat fixed-size chunking, parent-child
chunking improved citation precision by roughly 15 percentage points --
citations pointed at the chunk that actually contained the cited fact far
more often -- at the cost of roughly double the indexing time, since every
section now gets chunked at two granularities instead of one.
