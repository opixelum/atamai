use ndarray::Array1;

const EPSILON: f64 = f64::MIN_POSITIVE;

pub enum Loss {
    BinaryCrossEntropy,
    CategoricalCrossEntropy,
    MeanAbsoluteError,
    MeanSquaredError,
}

pub fn loss(loss: Loss, targets: Array1<f64>, predictions: Array1<f64>) -> f64 {
    match loss {
        Loss::BinaryCrossEntropy => binary_cross_entropy(targets, predictions),
        Loss::CategoricalCrossEntropy => categorical_cross_entropy(targets, predictions),
        Loss::MeanAbsoluteError => mean_absolute_error(targets, predictions),
        Loss::MeanSquaredError => mean_squared_error(targets, predictions),
    }
}

fn binary_cross_entropy(targets: Array1<f64>, predictions: Array1<f64>) -> f64 {
    let mut sum: f64 = 0.;
    for (i, prediction) in predictions.iter().enumerate() {
        sum += targets[i] * (prediction + EPSILON).ln()
            + (1. - targets[i]) * (1. - prediction + EPSILON).ln();
    }
    -sum / predictions.len() as f64
}

fn categorical_cross_entropy(targets: Array1<f64>, predictions: Array1<f64>) -> f64 {
    let mut sum: f64 = 0.;
    for (i, prediction) in predictions.iter().enumerate() {
        sum += targets[i] * (prediction + EPSILON).ln()
    }
    sum
}

fn mean_absolute_error(targets: Array1<f64>, predictions: Array1<f64>) -> f64 {
    let mut sum: f64 = 0.;
    for (i, prediction) in predictions.iter().enumerate() {
        sum += (prediction - targets[i]).abs();
    }
    sum / predictions.len() as f64
}

fn mean_squared_error(targets: Array1<f64>, predictions: Array1<f64>) -> f64 {
    let mut sum: f64 = 0.;
    for (i, prediction) in predictions.iter().enumerate() {
        sum += (prediction - targets[i]).powi(2);
    }
    sum / predictions.len() as f64
}
