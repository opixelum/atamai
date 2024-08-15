use ndarray::Array2;

const EPSILON: f64 = f64::MIN_POSITIVE;

pub enum Optimizer {
    Adam(f64, f64, f64), // Learning rate, beta1 & beta2
    Momentum(f64, f64),  // Learning rate & momentum factor
    RMSProp(f64, f64),   // Learning rate & decay rate
    SGD(f64),            // Learning rate
}

struct Optimizer2D {
    optimizer: Optimizer,
    velocity: f64,
}

impl Optimizer2D {
    pub fn new(optimizer: Optimizer) -> Self {
        Optimizer2D {
            optimizer,
            velocity: 0.,
        }
    }

    pub fn update_weights(&mut self, weights: &mut Array2<f64>, gradients: &Array2<f64>) {
        match self.optimizer {
            Optimizer::Adam(learning_rate, beta1, beta2) => {
                adam(weights, gradients, learning_rate, beta1, beta2)
            }
            Optimizer::Momentum(learning_rate, momentum_factor) => {
                momentum(weights, gradients, learning_rate, momentum_factor)
            }
            Optimizer::RMSProp(learning_rate, decay_rate) => rms_prop(
                weights,
                gradients,
                learning_rate,
                decay_rate,
                &mut self.velocity,
            ),
            Optimizer::SGD(learning_rate) => sgd(weights, gradients, learning_rate),
        }
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

fn rms_prop(
    weights: &mut Array2<f64>,
    gradients: &Array2<f64>,
    learning_rate: f64,
    decay_rate: f64,
    velocity: &mut f64,
) {
    for neuron_idx in 0..weights.shape()[0] {
        for weight_idx in 0..weights.shape()[1] {
            *velocity = *velocity * decay_rate
                + (1. - decay_rate) * gradients[(neuron_idx, weight_idx)].powi(2);

            weights[(neuron_idx, weight_idx)] = weights[(neuron_idx, weight_idx)]
                - learning_rate * gradients[(neuron_idx, weight_idx)] / (velocity.sqrt() + EPSILON)
        }
    }
}

fn sgd(weights: &mut Array2<f64>, gradients: &Array2<f64>, learning_rate: f64) {
    for neuron_idx in 0..weights.shape()[0] {
        for weight_idx in 0..weights.shape()[1] {
            weights[(neuron_idx, weight_idx)] = weights[(neuron_idx, weight_idx)]
                - learning_rate * gradients[(neuron_idx, weight_idx)]
        }
    }
}
