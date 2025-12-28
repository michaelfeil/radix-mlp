use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use radix_mlp::compute_fold_and_scatter;

fn configure_criterion() -> Criterion {
    Criterion::default()
        .sample_size(50) // Reduce from default 100
        .warm_up_time(std::time::Duration::from_millis(100)) // Reduce warm-up
        .measurement_time(std::time::Duration::from_millis(500)) // Reduce measurement time
}

fn generate_test_data(
    batch_size: usize,
    seq_len: usize,
    shared_prefix_ratio: f64,
) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let shared_prefix_len = (seq_len as f64 * shared_prefix_ratio) as usize;
    let total_tokens = batch_size * seq_len;

    let mut input_ids = Vec::with_capacity(total_tokens);
    let mut position_ids = Vec::with_capacity(total_tokens);
    let mut cu_seq_lengths = Vec::with_capacity(batch_size + 1);
    cu_seq_lengths.push(0);

    for seq_idx in 0..batch_size {
        // Shared prefix across all sequences
        for j in 0..shared_prefix_len {
            let token = (j as u32 % 1000) + 1;
            input_ids.push(token);
            position_ids.push(j as u32);
        }

        // Unique tail per sequence
        for k in shared_prefix_len..seq_len {
            let token = 1_000_000u32 + (seq_idx as u32) * 10_000 + (k as u32);
            input_ids.push(token);
            position_ids.push(k as u32);
        }

        cu_seq_lengths.push(input_ids.len() as u32);
    }

    (input_ids, position_ids, cu_seq_lengths)
}

fn bench_compute_fold_and_scatter(c: &mut Criterion) {
    let mut group = c.benchmark_group("compute_fold_and_scatter");

    // Benchmark different batch sizes with fixed sequence length
    for batch_size in [4, 8, 16, 32, 64, 256, 2048].iter() {
        let seq_len = 512;
        let (input_ids, position_ids, cu_seq_lengths) =
            generate_test_data(*batch_size, seq_len, 0.25);

        group.bench_with_input(
            BenchmarkId::new("batch_size", batch_size),
            batch_size,
            |b, _| {
                b.iter(|| {
                    compute_fold_and_scatter(
                        black_box(&input_ids),
                        black_box(&position_ids),
                        black_box(&cu_seq_lengths),
                        black_box(false),
                    )
                })
            },
        );
    }

    // Benchmark different sequence lengths with fixed batch size
    for seq_len in [128, 256, 512, 1024, 2048].iter() {
        let batch_size = 32;
        let (input_ids, position_ids, cu_seq_lengths) =
            generate_test_data(batch_size, *seq_len, 0.25);

        group.bench_with_input(
            BenchmarkId::new("seq_len", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    compute_fold_and_scatter(
                        black_box(&input_ids),
                        black_box(&position_ids),
                        black_box(&cu_seq_lengths),
                        black_box(false),
                    )
                })
            },
        );
    }

    // Benchmark different shared prefix ratios
    for prefix_ratio in [0.0, 0.25, 0.5, 0.75, 1.0].iter() {
        let batch_size = 32;
        let seq_len = 512;
        let (input_ids, position_ids, cu_seq_lengths) =
            generate_test_data(batch_size, seq_len, *prefix_ratio);

        group.bench_with_input(
            BenchmarkId::new("prefix_ratio", prefix_ratio),
            prefix_ratio,
            |b, _| {
                b.iter(|| {
                    compute_fold_and_scatter(
                        black_box(&input_ids),
                        black_box(&position_ids),
                        black_box(&cu_seq_lengths),
                        black_box(false),
                    )
                })
            },
        );
    }

    // Benchmark with padding enabled
    for seq_len in [128, 512, 1024, 4096, 16384, 131072].iter() {
        let batch_size = 32;
        let (input_ids, position_ids, cu_seq_lengths) =
            generate_test_data(batch_size, *seq_len, 0.25);

        group.bench_with_input(
            BenchmarkId::new("with_padding", seq_len),
            seq_len,
            |b, _| {
                b.iter(|| {
                    compute_fold_and_scatter(
                        black_box(&input_ids),
                        black_box(&position_ids),
                        black_box(&cu_seq_lengths),
                        black_box(true),
                    )
                })
            },
        );
    }

    // Benchmark edge cases
    // Empty input
    group.bench_function("empty_input", |b| {
        let input_ids: Vec<u32> = vec![];
        let position_ids: Vec<u32> = vec![];
        let cu_seq_lengths: Vec<u32> = vec![];

        b.iter(|| {
            compute_fold_and_scatter(
                black_box(&input_ids),
                black_box(&position_ids),
                black_box(&cu_seq_lengths),
                black_box(false),
            )
        })
    });

    // Single sequence
    group.bench_function("single_sequence", |b| {
        let input_ids: Vec<u32> = (0..512).map(|x| x as u32).collect();
        let position_ids: Vec<u32> = (0..512).map(|x| x as u32).collect();
        let cu_seq_lengths: Vec<u32> = vec![0, 512];

        b.iter(|| {
            compute_fold_and_scatter(
                black_box(&input_ids),
                black_box(&position_ids),
                black_box(&cu_seq_lengths),
                black_box(false),
            )
        })
    });

    // Identical sequences (maximum compression)
    group.bench_function("identical_sequences", |b| {
        let batch_size = 32;
        let seq_len = 512;
        let total_tokens = batch_size * seq_len;

        let mut input_ids = Vec::with_capacity(total_tokens);
        let mut position_ids = Vec::with_capacity(total_tokens);
        let mut cu_seq_lengths = Vec::with_capacity(batch_size + 1);
        cu_seq_lengths.push(0);

        for _ in 0..batch_size {
            for j in 0..seq_len {
                input_ids.push(j as u32);
                position_ids.push(j as u32);
            }
            cu_seq_lengths.push(input_ids.len() as u32);
        }

        b.iter(|| {
            compute_fold_and_scatter(
                black_box(&input_ids),
                black_box(&position_ids),
                black_box(&cu_seq_lengths),
                black_box(false),
            )
        })
    });

    // No overlap (no compression)
    group.bench_function("no_overlap", |b| {
        let batch_size = 32;
        let seq_len = 512;

        let mut input_ids = Vec::with_capacity(batch_size * seq_len);
        let mut position_ids = Vec::with_capacity(batch_size * seq_len);
        let mut cu_seq_lengths = Vec::with_capacity(batch_size + 1);
        cu_seq_lengths.push(0);

        for seq_idx in 0..batch_size {
            for j in 0..seq_len {
                let token = (seq_idx * seq_len + j) as u32;
                input_ids.push(token);
                position_ids.push(j as u32);
            }
            cu_seq_lengths.push(input_ids.len() as u32);
        }

        b.iter(|| {
            compute_fold_and_scatter(
                black_box(&input_ids),
                black_box(&position_ids),
                black_box(&cu_seq_lengths),
                black_box(false),
            )
        })
    });

    group.finish();
}

criterion_group! {
    name = benches;
    config = configure_criterion();
    targets = bench_compute_fold_and_scatter
}
criterion_main!(benches);