#![no_std]
#![no_main]

use esp_hal::clock::CpuClock;
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use esp_hal::rng::Rng;

// Abbiamo aggiunto OutputConfig qui!
use esp_hal::gpio::{Level, Output, OutputConfig}; 

use esp_println::println;
use esp_backtrace as _;




#[path = "../model_weights.rs"]
mod model_weights;
#[path = "../test_data.rs"]
mod test_data;
const LABELS: [&str; 6] = [
    "LAYING", 
    "SITTING", 
    "STANDING", 
    "WALKING", 
    "WALKING_DOWNSTAIRS", 
    "WALKING_UPSTAIRS"
];

esp_bootloader_esp_idf::esp_app_desc!();
fn predict_activity(input_features: &[f32;5])->usize{
    let mut logits=[0.0; 6];
    
    // W.T @ x + b
    for k in 0..6{
        logits[k]=model_weights::B_VECTOR[k];
        for i in 0..5{
            logits[k]+=model_weights::W_MATRIX[i*6+k]*input_features[i];
        }
    }

    let mut max_index=0;
    let mut max_score=logits[0];

    for k in 1..6{
        if logits[k] > max_score{
            max_score=logits[k];
            max_index=k
        }
    }
    
    max_index
        
}
#[main]
fn main() -> ! {
    let config = esp_hal::Config::default().with_cpu_clock(CpuClock::max());
    let peripherals = esp_hal::init(config);
    let mut trng = Rng::new(); //hardware random generator
    let mut led = Output::new(peripherals.GPIO2, Level::Low, OutputConfig::default());

    println!("Starting Edge Biomechanics Classifier...");
    println!("Weights preparing completed: 5 features, 6 classes");
    
    let mut current_sample=0;
    let mut correct_predictions=0;
    
    loop {
        let delay_start = Instant::now();
        let test_sample=test_data::TEST_FEATURES[current_sample];
        let start_inference=Instant::now();
        let predicted_idx=predict_activity(&test_sample);
        let inference_time=start_inference.elapsed().as_micros();

        let correct_idx: usize=test_data::TEST_LABELS[current_sample];

        println!("--------------------------------------------------");
        println!("Input (Latent Space) : {:?}", test_sample);
        println!("Predicted Class      : {} ({})", predicted_idx, LABELS[predicted_idx]);
        println!("Correct Label   : {} ({})", correct_idx, LABELS[correct_idx]);
        println!("Inference Time   : {} microseconds", inference_time);
        
        if correct_idx == predicted_idx {
            println!("CORRECT!!!");
            correct_predictions+=1;
            led.set_high();
            while delay_start.elapsed() < Duration::from_millis(2000) {}
        

        }else if correct_idx != predicted_idx{
            println!("Not correct..");
        }
        led.set_low();
        while delay_start.elapsed() < Duration::from_millis(2000) {}
        current_sample +=1;
        
        if current_sample==50{
            current_sample=0;
        }
    }
}