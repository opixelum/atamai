use ndarray::Array1;

pub enum Activation {
    Identity,
    Linear,
    Logistic,
    Relu,
    Sigmoid,
    Softmax,
    Tanh,
}

pub fn activation(function: Activation, inputs: &Array1<f64>) -> Array1<f64> {
    match function {
        Activation::Identity | Activation::Linear => Identity::activation(inputs),
        Activation::Logistic | Activation::Sigmoid => Logistic::activation(inputs),
        Activation::Relu => Relu::activation(inputs),
        Activation::Softmax => Softmax::activation(inputs),
        Activation::Tanh => Tanh::activation(inputs),
    }
}

pub fn derivative(function: Activation, inputs: &Array1<f64>) -> Array1<f64> {
    match function {
        Activation::Identity | Activation::Linear => Identity::derivative(inputs),
        Activation::Logistic | Activation::Sigmoid => Logistic::derivative(inputs),
        Activation::Relu => Relu::derivative(inputs),
        Activation::Softmax => Softmax::derivative(inputs),
        Activation::Tanh => Tanh::derivative(inputs),
    }
}

struct Identity;
impl Identity {
    fn activation(inputs: &Array1<f64>) -> Array1<f64> {
        inputs.clone()
    }

    fn derivative(inputs: &Array1<f64>) -> Array1<f64> {
        inputs.clone()
    }
}

struct Logistic;
impl Logistic {
    fn activation(inputs: &Array1<f64>) -> Array1<f64> {
        inputs.mapv(|x| 1.0 / (1.0 + (-x).exp()))
    }

    fn derivative(inputs: &Array1<f64>) -> Array1<f64> {
        inputs.mapv(|x| (-x).exp() / (2. * (-x).exp() + ((-x).exp()).powi(2) + 1.))
    }
}

struct Relu;
impl Relu {
    fn activation(inputs: &Array1<f64>) -> Array1<f64> {
        inputs.mapv(|x| if x > 0.0 { x } else { 0.0 })
    }

    fn derivative(inputs: &Array1<f64>) -> Array1<f64> {
        inputs.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }
}

struct Softmax;
impl Softmax {
    fn activation(inputs: &Array1<f64>) -> Array1<f64> {
        let max: f64 = inputs.fold(inputs[0], |acc, &x| if x > acc { x } else { acc });
        let exps: Array1<f64> = inputs.mapv(|x| (x - max).exp());
        let sum: f64 = exps.sum();
        exps / sum
    }

    fn derivative(inputs: &Array1<f64>) -> Array1<f64> {
        Array1::ones(inputs.len())
    }
}

struct Tanh;
impl Tanh {
    fn activation(inputs: &Array1<f64>) -> Array1<f64> {
        inputs.mapv(|x| x.tanh())
    }

    fn derivative(inputs: &Array1<f64>) -> Array1<f64> {
        inputs.mapv(|x| -(x.tanh().powi(2)) + 1.)
    }
}
