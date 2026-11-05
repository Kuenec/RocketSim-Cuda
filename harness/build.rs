use std::path::PathBuf;

fn main() {
    let dir = match std::env::var("RSC_LIB_DIR") {
        Ok(dir) => PathBuf::from(dir),
        Err(_) => {
            PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("../build/cmake")
        }
    };
    let dir = dir
        .canonicalize()
        .unwrap_or_else(|err| panic!("library directory {} is missing: {err}", dir.display()));
    println!("cargo:rerun-if-env-changed=RSC_LIB_DIR");
    println!(
        "cargo:rerun-if-changed={}",
        dir.join("librocketsim_cuda_c.so").display()
    );
    println!("cargo:rustc-link-search=native={}", dir.display());
    println!("cargo:rustc-link-lib=dylib=rocketsim_cuda_c");
    println!("cargo:rustc-link-arg=-Wl,-rpath,{}", dir.display());
}
