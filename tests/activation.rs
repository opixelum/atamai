use atamai::activation::*;
use ndarray::{array, Array1};

#[test]
fn identity_activation() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = activation(Activation::Identity, &inputs);
    let expected_outputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn identity_derivative() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = derivative(Activation::Identity, &inputs);
    let expected_outputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn linear_activation() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = activation(Activation::Linear, &inputs);
    let expected_outputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn linear_derivative() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = derivative(Activation::Linear, &inputs);
    let expected_outputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn logistic_activation() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = activation(Activation::Logistic, &inputs);
    let expected_outputs: Array1<f64> = array![
        0.11920292202211755,
        0.2689414213699951,
        0.5,
        0.7310585786300049,
        0.8807970779778823
    ];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn logistic_derivative() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = derivative(Activation::Logistic, &inputs);
    let expected_outputs: Array1<f64> = array![
        0.10499358540350652,
        0.19661193324148185,
        0.25,
        0.19661193324148185,
        0.10499358540350652
    ];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn relu_activation() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = activation(Activation::Relu, &inputs);
    let expected_outputs: Array1<f64> = array![0., 0., 0., 1., 2.];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn relu_derivative() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = derivative(Activation::Relu, &inputs);
    let expected_outputs: Array1<f64> = array![0., 0., 0., 1., 1.];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn sigmoid_activation() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = activation(Activation::Sigmoid, &inputs);
    let expected_outputs: Array1<f64> = array![
        0.11920292202211755,
        0.2689414213699951,
        0.5,
        0.7310585786300049,
        0.8807970779778823
    ];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn sigmoid_derivative() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = derivative(Activation::Sigmoid, &inputs);
    let expected_outputs: Array1<f64> = array![
        0.10499358540350652,
        0.19661193324148185,
        0.25,
        0.19661193324148185,
        0.10499358540350652
    ];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn sign_activation() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = activation(Activation::Sign, &inputs);
    let expected_outputs: Array1<f64> = array![-1., -1., 1., 1., 1.];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn sign_derivative() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = derivative(Activation::Sign, &inputs);
    let expected_outputs: Array1<f64> = array![0., 0., 0., 0., 0.];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn softmax_activation() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = activation(Activation::Softmax, &inputs);
    let expected_outputs: Array1<f64> = array![
        0.011656230956039605,
        0.03168492079612427,
        0.0861285444362687,
        0.23412165725273662,
        0.6364086465588308
    ];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn softmax_derivative() {}

#[test]
fn tanh_activation() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = activation(Activation::Tanh, &inputs);
    let expected_outputs: Array1<f64> = array![
        -0.9640275800758169,
        -0.7615941559557649,
        0.0,
        0.7615941559557649,
        0.9640275800758169
    ];
    assert_eq!(outputs, expected_outputs);
}

#[test]
fn tanh_derivative() {
    let inputs: Array1<f64> = array![-2., -1., 0., 1., 2.];
    let outputs: Array1<f64> = derivative(Activation::Tanh, &inputs);
    let expected_outputs: Array1<f64> = array![
        0.07065082485316443,
        0.41997434161402614,
        1.0,
        0.41997434161402614,
        0.07065082485316443
    ];
    assert_eq!(outputs, expected_outputs);
}
