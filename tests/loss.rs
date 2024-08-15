use ndarray::{array, Array1};

use atamai::loss;
use atamai::loss::Loss::*;

#[test]
fn binary_cross_entropy() {
    let mut predictions: Array1<f64> = array![0.];
    let mut targets: Array1<f64> = array![0.];
    let mut loss: f64 = loss::loss(BinaryCrossEntropy, predictions, targets);
    let mut expected_loss: f64 = 0.;
    assert_eq!(loss, expected_loss);

    // TODO: check with TensorFlow
    predictions = array![0., 1., 0., 0.];
    targets = array![0.6, 0.3, 0.2, 0.8];
    loss = loss::loss(BinaryCrossEntropy, predictions, targets);
    expected_loss = 0.865; // to update
    assert_eq!(loss, expected_loss);
}

#[test]
fn mean_absolute_error() {
    let mut predictions: Array1<f64> = array![0.];
    let mut targets: Array1<f64> = array![0.];
    let mut loss: f64 = loss::loss(MeanAbsoluteError, predictions, targets);
    let mut expected_loss: f64 = 0.;
    assert_eq!(loss, expected_loss);

    predictions = array![12., 15., 10., 20., 18.];
    targets = array![10., 15., 12., 18., 20.];
    loss = loss::loss(MeanAbsoluteError, predictions, targets);
    expected_loss = 1.6;
    assert_eq!(loss, expected_loss);
}
