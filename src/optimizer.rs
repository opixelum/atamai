use ndarray::Array2;

pub enum Optimizer {
    Adam(f64, f64, f64), // Learning rate, beta1 & beta2
    Momentum(f64, f64),  // Learning rate & momentum factor
    RMSProp(f64, f64),   // Learning rate & decay rate
    SGD(f64),            // Learning rate
}

pub fn update_weights(optimizer: Optimizer, weights: &mut Array2<f64>, gradients: &Array2<f64>) {
    match optimizer {
        Optimizer::Adam(learning_rate, beta1, beta2) => {
            adam(weights, gradients, learning_rate, beta1, beta2)
        }
        Optimizer::Momentum(learning_rate, momentum_factor) => {
            momentum(weights, gradients, learning_rate, momentum_factor)
        }
        Optimizer::RMSProp(learning_rate, decay_rate) => {
            rms_prop(weights, gradients, learning_rate, decay_rate)
        }
        Optimizer::SGD(learning_rate) => sgd(weights, gradients, learning_rate),
    }
}

fn adam(
    weights: &mut Array2<f64>,
    gradients: &Array2<f64>,
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
) {
}

fn momentum(
    weights: &mut Array2<f64>,
    gradients: &Array2<f64>,
    learning_rate: f64,
    momentum_factor: f64,
) {
}

fn rms_prop(weights: &mut Array2<f64>, gradients: &Array2<f64>, learning_rate: f64, decay: f64) {}

fn sgd(weights: &mut Array2<f64>, gradients: &Array2<f64>, learning_rate: f64) {
    for neuron_idx in 0..weights.shape()[0] {
        for weight_idx in 0..weights.shape()[1] {
            weights[(neuron_idx, weight_idx)] = weights[(neuron_idx, weight_idx)]
                - learning_rate * gradients[(neuron_idx, weight_idx)]
        }
    }
}
