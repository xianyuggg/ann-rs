mod neuron;
use neuron::Neuron;
use rand_distr::{Normal, Distribution};
use rand;

pub struct NeuronLayer {
    pub input_num_per_neuron : usize,
    pub neuron_num : usize, // current layer num
    pub neurons : Vec<Neuron>
}

impl NeuronLayer {
    pub fn new( num_neurons : usize, num_per_neuron : usize) -> NeuronLayer{
        let mut tmp_vec = vec![];

        for _ in 0..num_neurons {

            let normal = Normal::new(2.0, 1.0).unwrap();
            let vec = (0..num_per_neuron+1).map(|_| normal.sample(&mut rand::thread_rng())).collect();

            tmp_vec.push(
                Neuron {
                    out_activation: 0.0, // TODO : RANDOM START
                    out_error: 0.0,
                    weights: vec
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