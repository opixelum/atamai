use ndarray::Array1;

const EPSILON: f64 = f64::MIN_POSITIVE;

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
        Loss::CategoricalCrossEntropy => categorical_cross_entropy(predictions, targets),
        Loss::HingeLoss => 0.,
        Loss::HuberLoss => 0.,
        Loss::MeanAbsoluteError => mean_absolute_error(predictions, targets),
        Loss::MeanSquaredError => 0.,
    }
}

fn binary_cross_entropy(predictions: Array1<f64>, targets: Array1<f64>) -> f64 {
    let mut sum: f64 = 0.;
    for (i, prediction) in predictions.iter().enumerate() {
        sum += targets[i] * (prediction + EPSILON).log10()
            + (1. * targets[i]) * (1. - prediction + EPSILON).log10()
    }
    -sum / predictions.len() as f64
}

fn categorical_cross_entropy(predictions: Array1<f64>, targets: Array1<f64>) -> f64 {
    let mut sum: f64 = 0.;
    for (i, prediction) in predictions.iter().enumerate() {
        sum += targets[i] * (prediction + EPSILON).log10()
    }
    sum
}

// fn hinge_loss(predictions: Array1<f64>, targets: Array1<f64>) -> f64 {}

// fn huber_loss(predictions: Array1<f64>, targets: Array1<f64>) -> f64 {}

fn mean_absolute_error(predictions: Array1<f64>, targets: Array1<f64>) -> f64 {
    let mut sum: f64 = 0.;
    for (i, prediction) in predictions.iter().enumerate() {
        sum += (prediction - targets[i]).abs();
    }
    sum / predictions.len() as f64
}

// fn mean_squared_error(predictions: Array1<f64>, targets: Array1<f64>) -> f64 {}
