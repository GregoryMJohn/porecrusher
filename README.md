# porecrusher
A rewrite of [PoreBlazer](https://github.com/SarkisovGitHub/PoreBlazer) in Rust which adds SIMD support and parallelization, allowing execution on multiple cores/threads which dramatically improves performance and allows for porosimetry simulations of much larger systems. 

# Installation
Requires nightly version of rust, which can be installed with rustup.

```
sudo apt install rustup
rustup update
rustup toolchain install nightly
rustup default nightly
```

Then add the following lines to ~/.cargo/conf.toml:
```
[build]
rustflags = ["-Znext-solver=coherence"]
```
Next, clone the repository and navigate to the porecrusher directory, then compile with cargo.
```
git clone https://github.com/GregoryMJohn/porecrusher.git
cd porecrusher/
cargo build --release
```
