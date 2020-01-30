use ann_lib::NeuronLayer;
use std::intrinsics::expf64;

// currently output size is 10
pub fn train_epoch(input: Vec<f64>, output: Vec<f64>, nn: &mut ANN, img_num: usize, img_size: usize) {

    for i in 0..img_num {
        let net_input = input[i * img_size..(i+1) * img_size].iter().map(|val| *val).collect();
        let net_output = input[i * 4..(i+1) * 4].iter().map(|val| *val).collect();
        nn.training(net_input, net_output);
        println!("Training {}th image, error : {}",i, nn.error_sum);
        nn.error_sum = 0.0;
    }

}

pub struct ANN {
    pub num_input : usize,
    pub num_output : usize,
    pub num_hidden_layers : usize,
    pub lr : f64, // learning rate
    pub error_sum : f64,
    pub neuron_layers : Vec<NeuronLayer>
}

impl ANN {

    pub fn add_neuron_layer(&mut self, neuron_num: usize) {

        let num_input_per_neuron = if self.neuron_layers.is_empty() {
            self.num_input
        } else {
            self.neuron_layers.iter().last().unwrap().neuron_num
        };
        self.neuron_layers.push(
           NeuronLayer::new(neuron_num, num_input_per_neuron)
        );

        self.num_hidden_layers = if !self.neuron_layers.is_empty() {
            self.neuron_layers.len() - 1
        }else {
            0
        }
    }

    fn sigmoid_activate(activation : f64) -> f64 {
        (1.0 / (1.0 + unsafe { expf64(-activation) }))
    }

    fn update_neuron_layer( nl : &mut NeuronLayer, input : &Vec<f64>) {
        let num_neuron = nl.neuron_num;
        let neurons = &mut nl.neurons;
        for i in 0..num_neuron {
            let mut tmp = 0.0;
            for j in 0..neurons[i].weights.len() - 1 {
                tmp += neurons[i].weights[j] * input[j];
            }
            tmp += *neurons[i].weights.last().unwrap();
            neurons[i].out_activation = ANN::sigmoid_activate(tmp);
        }
    }

    // propagate from left to right
    pub fn process(&mut self, mut input: Vec<f64>, output: &mut Vec<f64>) {
        self.neuron_layers.iter_mut().for_each( |layer| {
            ANN::update_neuron_layer(layer, &input);
            input = layer.neurons.iter().map(|n| n.out_activation).collect();
        });

        assert_eq!(self.neuron_layers.len(), self.num_hidden_layers + 1);
        // the last layer refers to the output layer
        for (idx, n) in self.neuron_layers.last().unwrap().neurons.iter().enumerate() {
            output[idx] = n.out_activation;
        }
    }
    // f' for sigmoid function
    fn backactive(x : f64) -> f64 {
        x * (1.0 - x)
    }

    fn train_update(&mut self ,mut input : Vec<f64>, target: Vec<f64>) {
        self.neuron_layers.iter_mut().for_each( |layer| {
            ANN::update_neuron_layer(layer, &input);
            input = layer.neurons.iter().map(|n| n.out_activation).collect();
        });

        // get the last layer and calculate error between target and activation ( activation range (0, 1))
        for (idx, n) in self.neuron_layers.last_mut().unwrap().neurons.iter_mut().enumerate() {
            n.out_error = target[idx] - n.out_activation;   // target - activation
            self.error_sum += n.out_error * n.out_error;
        };
    }

    fn train_neuron_layer(nl : &mut NeuronLayer, prev_activation: Vec<f64>, prev_error: &mut Vec<f64>, lr : f64) {

        for (idx, n) in nl.neurons.iter_mut().enumerate() {
            let err = n.out_error * ANN::sigmoid_activate(n.out_activation);

            for i in 0..nl.input_num_per_neuron {
                if !prev_error.is_empty() {
                    prev_error[i] += n.weights[i] * err;
                }
                n.weights[i] += err * lr * prev_activation[i];
            }
            n.weights[nl.input_num_per_neuron] += err * lr;    // last is the bias item
        }
    }

    fn training(&mut self, input : Vec<f64>, target: Vec<f64>) {
        println!("Training!");
        ANN::train_update(self, input.clone(), target);

        for idx in (0..self.neuron_layers.len()).rev() {
            let mut prev_activation = vec![];
            let mut prev_error = vec![];
            if idx > 0 {
                prev_activation = self.neuron_layers[idx - 1].neurons.iter().map( | n| n.out_activation).collect();
                prev_error = vec![0.0;self.neuron_layers[idx - 1].neurons.len()]
            }else{
                prev_activation = input.clone();
            }
            ANN::train_neuron_layer(&mut self.neuron_layers[idx], prev_activation, &mut prev_error, self.lr);

            //update error
            if idx > 0 {
                assert_ne!(prev_error.len(), 0);
                for i in 0..prev_error.len(){
                    self.neuron_layers[idx - 1].neurons[i].out_error = prev_error[i];
                }
            }
        }
    }
}