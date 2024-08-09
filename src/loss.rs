use ndarray::Array1;

pub enum Loss {
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    HingeLoss,
    HuberLoss,
    MeanAbsoluteError,
    MeanSquaredError,
}

pub fn loss(loss: Loss, predictions: Array1<f64>, targets: Array1<f64>) -> f64 {
    match loss {
        Loss::BinaryCrossEntropy => binary_cross_entropy(predictions, targets),
        Loss::CategoricalCrossEntropy => 0.,
        Loss::HingeLoss => 0.,
        Loss::HuberLoss => 0.,
        Loss::MeanAbsoluteError => 0.,
        Loss::MeanSquaredError => 0.,
    }
}

fn binary_cross_entropy(predictions: Array1<f64>, targets: Array1<f64>) -> f64 {
    let epsilon: f64 = f64::MIN_POSITIVE;
    let mut sum: f64 = 0.;
    for (i, prediction) in predictions.iter().enumerate() {
        sum += targets[i] * (prediction + epsilon).log10()
            + (1. * targets[i]) * (1. - prediction + epsilon).log10()
    }
    -sum / predictions.len() as f64
}

/*
fn categorical_cross_entropy(inputs: Array1<f64>) -> f64 {}

fn hinge_loss(inputs: Array1<f64>) -> f64 {}

fn mean_absolute_error(inputs: Array1<f64>) -> f64 {}

fn mean_squared_error(inputs: Array1<f64>) -> f64 {}

fn huber_loss(inputs: Array1<f64>) -> f64 {}
*/
