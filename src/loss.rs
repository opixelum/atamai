use ndarray::Array2;

const EPSILON: f64 = f64::MIN_POSITIVE;

pub enum Loss {
    BinaryCrossEntropy,
    L1,
    L2,
    LogLoss,
    MeanAbsoluteError,
    MeanSquaredError,
}

pub fn loss(loss: Loss, targets: &Array2<f64>, predictions: &Array2<f64>) -> f64 {
    match loss {
        Loss::BinaryCrossEntropy | Loss::LogLoss => binary_cross_entropy(targets, predictions),
        Loss::MeanAbsoluteError | Loss::L1 => mean_absolute_error(targets, predictions),
        Loss::MeanSquaredError | Loss::L2 => mean_squared_error(targets, predictions),
    }
}

fn binary_cross_entropy(targets: &Array2<f64>, predictions: &Array2<f64>) -> f64 {
    let mut sum: f64 = 0.;
    for (i, prediction) in predictions.iter().enumerate() {
        sum += targets[(i, 0)] * (prediction + EPSILON).ln()
            + (1. - targets[(i, 0)]) * (1. - prediction + EPSILON).ln();
    }
    -sum / predictions.len() as f64
}

fn mean_absolute_error(targets: &Array2<f64>, predictions: &Array2<f64>) -> f64 {
    let mut sum: f64 = 0.;
    for (i, prediction) in predictions.iter().enumerate() {
        sum += (prediction - targets[(i, 0)]).abs();
    }
    sum / predictions.len() as f64
}

fn mean_squared_error(targets: &Array2<f64>, predictions: &Array2<f64>) -> f64 {
    let mut sum: f64 = 0.;
    for (i, prediction) in predictions.iter().enumerate() {
        sum += (prediction - targets[(i, 0)]).powi(2);
    }
    sum / predictions.len() as f64
}
