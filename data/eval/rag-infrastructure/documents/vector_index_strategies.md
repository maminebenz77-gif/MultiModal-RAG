# Vector Index Strategy Notes: HNSW vs IVF

Internal notes comparing the two indexing strategies we evaluated for the
vector store before settling on our current default.

## HNSW (Hierarchical Navigable Small World)

HNSW builds a multi-layer graph structure incrementally, one vector at a
time, so the index is always queryable during ingestion -- no separate
training step is required. In our benchmark corpus, HNSW achieved 98%
recall at an average query latency of 2ms, at the cost of roughly 1.5x the
memory footprint of a flat (brute-force) index, since each vector also
stores its graph connections.

Qdrant, the vector store this project uses, defaults to HNSW for exactly
this reason: incremental build time matters more to us than the extra
memory overhead, since documents get ingested continuously rather than in
one large batch.

## IVF (Inverted File Index)

IVF clusters the vector space into a fixed number of coarse partitions
(centroids) and only searches the partitions nearest the query vector,
rather than the whole graph. In the same benchmark, IVF with 100 clusters
achieved 91% recall at 1.2ms average query latency -- faster per query, but
with a meaningfully lower recall ceiling than HNSW at this corpus size.

The real limitation is operational, not just accuracy: IVF requires a
training/clustering step over a representative sample of vectors before it
can serve any queries at all, which makes it a poor fit for a store that
needs to accept new documents incrementally without a rebuild.

## Conclusion

For a corpus that grows continuously through ingestion rather than being
indexed once in bulk, HNSW's incremental-build property outweighs IVF's
raw query-latency edge, even though IVF is marginally faster per query.
