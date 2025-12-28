use numpy::PyArray1;
use numpy::PyArrayMethods;
use pyo3::prelude::*;
use pyo3::types::PyModuleMethods;

/// Python wrapper for compute_fold_and_scatter
#[pyfunction]
fn compute_fold_and_scatter(
    input_ids: &Bound<PyArray1<u32>>,
    position_ids: &Bound<PyArray1<u32>>,
    cu_seq_lengths: &Bound<PyArray1<u32>>,
    pad_multiple_of: bool,
) -> PyResult<(
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
    Py<PyArray1<u32>>,
)> {
    // Convert numpy arrays to Rust slices
    let input_ids_slice = unsafe { input_ids.as_slice()? };
    let position_ids_slice = unsafe { position_ids.as_slice()? };
    let cu_seq_lengths_slice = unsafe { cu_seq_lengths.as_slice()? };

    // Call Rust function
    let (compact_input_ids, compact_position_ids, scatter_indices, fold_gather) =
        radix_mlp::compute_fold_and_scatter(
            input_ids_slice,
            position_ids_slice,
            cu_seq_lengths_slice,
            pad_multiple_of,
        );

    // Convert back to numpy arrays
    let py = input_ids.py();
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
