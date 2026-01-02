use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyModuleMethods;

/// Python wrapper for compute_fold_and_scatter
#[pyfunction]
fn compute_fold_and_scatter(
    py: pyo3::Python<'_>,
    input_ids: &Bound<PyArray1<u32>>,
    position_ids: &Bound<PyArray1<u32>>,
    cu_seq_lengths: &Bound<PyArray1<u32>>,
    pad_multiple_of: Option<usize>,
    bound_checks: Option<bool>,
) -> PyResult<(
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
)> {
    // Get basic lengths for data copying
    let input_len = input_ids.len()?;
    let position_len = position_ids.len()?;
    let cu_seq_len = cu_seq_lengths.len()?;

    // Copy numpy arrays to Rust Vecs for thread-safe access without the GIL
    let input_data_vec = unsafe { input_ids.as_slice()?.to_vec() };
    let position_data_vec = unsafe { position_ids.as_slice()?.to_vec() };
    let cu_seq_lengths_data_vec = unsafe { cu_seq_lengths.as_slice()?.to_vec() };

    // Default to enabling bound checks for safety
    let enable_bound_checks = bound_checks.unwrap_or(true);

    // Release GIL during computation (including optional bounds checking)
    let result = py.detach(|| {
        // Perform bounds checking if enabled
        if enable_bound_checks {
            // Check that input_ids and position_ids have the same length
            if input_len != position_len {
                return Err(format!(
                    "input_ids and position_ids must have the same length: {} != {}",
                    input_len, position_len
                ));
            }

            // Check that cu_seq_lengths is not empty and has at least 2 elements (start and end)
            if cu_seq_len < 2 {
                return Err(format!(
                    "cu_seq_lengths must have at least 2 elements (start and end), got {}",
                    cu_seq_len
                ));
            }

            // Check that cu_seq_lengths is properly formatted and indexable
            let cu_seq_slice = &cu_seq_lengths_data_vec;

            // Check that the first element is 0 (cumulative sequence lengths should start at 0)
            if cu_seq_slice[0] != 0 {
                return Err(format!(
                    "cu_seq_lengths must start with 0, got {}",
                    cu_seq_slice[0]
                ));
            }

            // Check that the sequence lengths are non-decreasing and don't exceed input length
            for i in 1..cu_seq_len {
                if cu_seq_slice[i] < cu_seq_slice[i - 1] {
                    return Err(format!(
                        "cu_seq_lengths must be non-decreasing at index {}: {} < {}",
                        i,
                        cu_seq_slice[i],
                        cu_seq_slice[i - 1]
                    ));
                }

                if cu_seq_slice[i] > input_len as u32 {
                    return Err(format!(
                        "cu_seq_lengths element {} ({}) exceeds input_ids length ({})",
                        i, cu_seq_slice[i], input_len
                    ));
                }
            }

            // Check that the last element equals the total input length
            if cu_seq_slice[cu_seq_len - 1] != input_len as u32 {
                return Err(format!(
                    "cu_seq_lengths last element ({}) must equal input_ids length ({})",
                    cu_seq_slice[cu_seq_len - 1],
                    input_len
                ));
            }
        }

        // Perform the actual computation
        Ok(radix_mlp::compute_fold_and_scatter(
            &input_data_vec,
            &position_data_vec,
            &cu_seq_lengths_data_vec,
            pad_multiple_of,
        ))
    });

    // Handle the result and convert errors if needed
    let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
        result.map_err(|msg| PyErr::new::<pyo3::exceptions::PyValueError, _>(msg))?;

    // Convert back to numpy arrays
    let compact_input_ids_arr = PyArray1::from_vec(py, compact_input_ids);
    let compact_position_ids_arr = PyArray1::from_vec(py, compact_position_ids);
    let scatter_indices_arr = PyArray1::from_vec(py, scatter_indices);
    let fold_gather_arr = PyArray1::from_vec(py, fold_gather);

    Ok((
        compact_input_ids_arr.into(),
        compact_position_ids_arr.into(),
        scatter_indices_arr.into(),
        fold_gather_arr.into(),
    ))
}

/// Python module definition
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compute_fold_and_scatter, m)?)?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
