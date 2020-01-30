mod neuron;
use neuron::Neuron;

pub struct NeuronLayer {
    pub input_num_per_neuron : usize,
    pub neuron_num : usize, // current layer num
    pub neurons : Vec<Neuron>
}

impl NeuronLayer {
    fn new( num_neurons : usize, num_per_neuron : usize) -> NeuronLayer{
        let mut tmp_vec = vec![];
        for _ in 0..num_neurons {
            tmp_vec.push(
                Neuron {
                    out_activation: 0.0, // TODO : RANDOM START
                    out_error: 0.0,
                    weights: vec![0.0;num_per_neuron]
                }
            )
        }
        NeuronLayer {
            input_num_per_neuron: num_per_neuron,
            neuron_num: num_neurons,
            neurons: tmp_vec
        }
    }
}