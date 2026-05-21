//! Agglomerative clustering for speaker embeddings using cosine distance.

use ndarray::Array1;

/// Single-linkage agglomerative clustering on cosine distance.
/// - If `num_clusters` is Some(k), merge until exactly k clusters remain.
/// - Otherwise stop when the smallest pairwise distance exceeds `threshold`.
/// Returns one label per embedding.
pub fn agglomerative(
    embeddings: &[Array1<f32>],
    num_clusters: Option<usize>,
    threshold: f32,
) -> Vec<usize> {
    let n = embeddings.len();
    if n == 0 {
        return Vec::new();
    }
    if n == 1 {
        return vec![0];
    }

    // Pairwise cosine distance matrix.
    let mut dist = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = cosine_distance(&embeddings[i], &embeddings[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    // Each point starts as its own cluster.
    let mut cluster_of: Vec<usize> = (0..n).collect();
    let mut active: Vec<bool> = vec![true; n];
    let mut remaining = n;

    let target = num_clusters.unwrap_or(1).max(1);

    loop {
        if num_clusters.is_some() && remaining <= target {
            break;
        }
        if remaining <= 1 {
            break;
        }

        // Find closest pair of active clusters (single linkage: min member distance).
        let mut best = (f32::INFINITY, 0usize, 0usize);
        for i in 0..n {
            if !active[i] {
                continue;
            }
            for j in (i + 1)..n {
                if !active[j] {
                    continue;
                }
                // Single linkage: smallest distance between any pair (a in cluster_i, b in cluster_j).
                let mut d = f32::INFINITY;
                for a in 0..n {
                    if cluster_of[a] != i {
                        continue;
                    }
                    for b in 0..n {
                        if cluster_of[b] != j {
                            continue;
                        }
                        if dist[a][b] < d {
                            d = dist[a][b];
                        }
                    }
                }
                if d < best.0 {
                    best = (d, i, j);
                }
            }
        }

        if num_clusters.is_none() && best.0 > threshold {
            break;
        }

        // Merge cluster `best.2` into `best.1`.
        for c in cluster_of.iter_mut() {
            if *c == best.2 {
                *c = best.1;
            }
        }
        active[best.2] = false;
        remaining -= 1;
    }

    // Relabel to dense 0..K range.
    let mut remap = std::collections::HashMap::new();
    let mut next = 0usize;
    let mut out = Vec::with_capacity(n);
    for &c in &cluster_of {
        let label = *remap.entry(c).or_insert_with(|| {
            let v = next;
            next += 1;
            v
        });
        out.push(label);
    }
    out
}

fn cosine_distance(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    // Assumes inputs are L2-normalised; clamp to be safe.
    1.0 - dot.clamp(-1.0, 1.0)
}
