extern crate rustc_serialize;
extern crate json_request;

use json_request::{request, Method};

#[derive(Debug, RustcEncodable)]
struct EuropeanCallOption {
        T: f64,
        r: f64,
        S_0: f64,
        sigma: f64,
        strike: f64,
}

#[derive(Debug, RustcDecodable)]
struct ResponseData {
    name: String,
    value: f64,
}



fn main() {
    let data = RequestData {
        T: 1.0,
        r: 0.05,
        S_0: 100.0,
        sigma: 0.2,
        strike: 100.0,
    };

    let response = request(Method::Post, "http://localhost:8000/contract/", Some(data));
}
