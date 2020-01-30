#![feature(core_intrinsics)]
mod nn;
use nn::ANN;
use nn::train_epoch;

const HIDDEN_NUM : usize = 5;
const OUTPUT_NUM : usize = 4;

const EPOCH : usize = 1000;

fn main() {
    let input: Vec<f64> = vec![
                        1.0, 0.0, 0.0, 0.0,
                        0.0, 1.0, 0.0, 0.0,
                        0.0, 0.0, 1.0, 0.0,
                        0.0, 0.0, 0.0, 1.0
    ];
    let output: Vec<f64> = vec![
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0
    ];

    let mut ann = ANN {
        num_input: 4,
        num_output: 4,
        num_hidden_layers: 0,
        lr: 0.3,
        error_sum: 0.0,
        neuron_layers: vec![]
    };

    ann.add_neuron_layer(HIDDEN_NUM);
    ann.add_neuron_layer( HIDDEN_NUM * 2);
    ann.add_neuron_layer(OUTPUT_NUM);

    // a 4, 4, 4 network

    for i in 0..EPOCH {
        println!("Training {}th EPOCH!", i);
        train_epoch(input.clone(), output.clone(), &mut ann, 4, 4);
    }
    let mut output = vec![0.0, 0.0, 0.0, 0.0];
    ann.process(vec![-0.1, 0.9, 0.1, 0.2], &mut output);
    println!("Test output : {:?}", output);

}
