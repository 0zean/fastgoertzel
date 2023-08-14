extern crate ndarray;
use pyo3::prelude::*;
use ndarray::prelude::*;
use ndarray::Array;
use std::f64::consts::PI;

#[pyfunction]
/// Goertzel algorithm for the given input data and frequency.
///
/// This function calculates the amplitude and phase using the Goertzel algorithm.
///
/// Parameters
/// ----------
/// x : List[float]
///     The input data.
/// f : float
///     The frequency value.
///
/// Returns
/// -------
/// Tuple[float, float]
///     The amplitude and phase.
fn goertzel(x: Vec<f64>, f: f64) -> PyResult<(f64, f64)> {
    let x_array: Array1<f64> = Array::from(x);
    let (amp, phase) = _goertzel(&x_array, f);
    Ok((amp, phase))
}

/// A Python module implemented in Rust.
#[pymodule]
fn fastgoertzel(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(goertzel, m)?)?;
    Ok(())
}

fn _goertzel(x: &Array1<f64>, f: f64) -> (f64, f64) {
    let n = x.len();
    let k = (f * n as f64) as usize;
    
    let w = 2.0 * PI * k as f64 / n as f64;
    let cw = w.cos();
    let c = 2.0 * cw;
    let sw = w.sin();
    let mut z1 = 0.0;
    let mut z2 = 0.0;

    for i in 0..n {
        let z0 = x[i] + c * z1 - z2;
        z2 = z1;
        z1 = z0;
    }

    let ip = cw * z1 - z2;
    let qp = sw * z1;
    
    let amp = (ip.powi(2) + qp.powi(2)).sqrt() / (n as f64 / 2.0);
    let phase = qp.atan2(ip);
    (amp, phase)
}