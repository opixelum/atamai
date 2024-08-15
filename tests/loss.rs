use atamai::loss;
use atamai::loss::Loss::*;
use ndarray::{array, Array2};

#[test]
fn binary_cross_entropy() {
    let targets: Array2<f64> = array![[0.], [1.], [0.], [0.]];
    let predictions: Array2<f64> = array![[0.6], [0.3], [0.2], [0.8]];
    let loss: f64 = loss::loss(BinaryCrossEntropy, targets, predictions);
    let expected_loss: f64 = 0.9882112499871003;
    assert_eq!(loss, expected_loss);
}

#[test]
fn mean_absolute_error() {
    let targets: Array2<f64> = array![[10.], [15.], [12.], [18.], [20.]];
    let predictions: Array2<f64> = array![[12.], [15.], [10.], [20.], [18.]];
    let loss: f64 = loss::loss(MeanAbsoluteError, targets, predictions);
    let expected_loss: f64 = 1.6;
    assert_eq!(loss, expected_loss);
}

#[test]
fn mean_squared_error() {
    let targets: Array2<f64> = array![[10.], [20.], [30.], [40.], [50.]];
    let predictions: Array2<f64> = array![[12.], [18.], [32.], [38.], [48.]];
    let loss: f64 = loss::loss(MeanSquaredError, targets, predictions);
    let expected_loss: f64 = 4.;
    assert_eq!(loss, expected_loss);
}
