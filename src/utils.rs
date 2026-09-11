use serde::Deserialize;

use crate::{
	atoms::{Atoms, Epsilon, Sigma},
	unit_cell::UnitCell,
};

#[derive(Clone, Deserialize)]
pub struct Settings {
	force_field:        String,
	helium:             Helium,
	nitrogen:           Nitrogen,
	temperature:        f64,
	cutoff:             f64,
	samples_per_atom:   u64,
	cubelet_size:       f64,
	max_pore_diameter:  f64,
	psd_bin_size:       f64,
	visualization_mode: String,
}

impl Settings {
	pub fn from_toml() -> Self {
		let mut path_to_settings = std::env::current_dir().unwrap();
		path_to_settings.push("settings.toml");

		let settings =
			if let Ok(settings) = std::fs::read_to_string(path_to_settings) {
				settings
			} else {
				let mut path = std::path::PathBuf::new();
				path.push(env!("CARGO_MANIFEST_DIR"));
				path.push("settings.toml");
				std::fs::read_to_string(path).unwrap()
			};

		toml::from_str(&settings).unwrap()
	}

	pub fn force_field(&self) -> &str {
		&self.force_field
	}

	pub fn helium(&self) -> &Helium {
		&self.helium
	}

	pub fn nitrogen(&self) -> &Nitrogen {
		&self.nitrogen
	}

	pub fn temperature(&self) -> f64 {
		self.temperature
	}

	pub fn cutoff(&self) -> f64 {
		self.cutoff
	}

	pub fn samples_per_atom(&self) -> u64 {
		self.samples_per_atom
	}

	pub fn cubelet_size(&self) -> f64 {
		self.cubelet_size
	}

	pub fn cubelet_size_mut(&mut self) -> &mut f64 {
		&mut self.cubelet_size
	}

	pub fn max_pore_diameter(&self) -> f64 {
		self.max_pore_diameter
	}

	pub fn psd_bin_size(&self) -> f64 {
		self.psd_bin_size
	}

	pub fn visualization_mode(&self) -> &str {
		&self.visualization_mode
	}
}

#[derive(Deserialize, Clone)]
pub struct Helium {
	sigma:   f64,
	epsilon: f64,
}

impl Helium {
	pub fn sigma(&self) -> Sigma {
		self.sigma.into()
	}

	pub fn espilon(self) -> Epsilon {
		self.epsilon.into()
	}
}

#[derive(Deserialize, Clone)]
pub struct Nitrogen {
	sigma: f64,
}

impl Nitrogen {
	pub fn sigma(&self) -> Sigma {
		self.sigma.into()
	}

	pub fn sigma2(&self) -> f64 {
		(self.sigma * self.sigma).into()
	}
}

pub fn parse_cl_args() -> (Atoms, UnitCell, Settings) {
	let args: Vec<String> = std::env::args().collect();
	let fh = args
		.get(1)
		.expect("No input file specified. Please specify an input file")
		.to_owned();

	let lines: Vec<String> = std::fs::read_to_string(&fh)
		.expect(&format!("Could not find '{:?}'", fh))
		.lines()
		.map(String::from)
		.collect();

	let xyz_file = std::path::Path::new(
		lines
			.get(0)
			.expect("Missing path to XYZ file in input file"),
	);

	let mut dimensions: Vec<f64> = lines
		.get(1)
		.expect("Missing cell dimensions in input file")
		.to_owned()
		.split_whitespace()
		.map(String::from)
		.map(|s| s.parse::<f64>().unwrap())
		.collect();

	assert_eq!(
		6,
		dimensions.len(),
		"Expected cell side lengths and angles (a b c α β γ)"
	);

	let angles = *dimensions
		.drain(3..)
		.collect::<Vec<f64>>()
		.as_array()
		.unwrap();

	let dims = *dimensions.as_array().unwrap();

	let unit_cell = UnitCell::new(dims, angles);
	let settings = Settings::from_toml();
	let atoms = Atoms::from_xyz_file(&xyz_file);

	(atoms, unit_cell, settings)
}

pub mod io {
	use std::sync::mpsc::Receiver;
	pub use std::sync::mpsc::Sender;
	pub struct Reporter {
		msgs: Receiver<Msg>,
	}

	impl Reporter {
		#[inline]
		pub fn new(receiver: Receiver<Msg>) -> Self {
			Self {
				msgs: receiver
			}
		}

		#[inline]
		pub fn next(&self) -> Result<Msg, std::sync::mpsc::RecvError> {
			self.msgs.recv()
		}
	}

	pub struct Msg(String);

	impl Msg {
		#[inline]
		pub fn new(msg: String) -> Self {
			Self(msg)
		}

		#[inline]
		pub fn display(self) {
			println!("{}", self.0)
		}

		pub fn make_heading(self) -> Self {
			let mut line1 = String::from("!");
			line1.push_str(&"-".repeat(80));
			line1.push_str("!\n");
			let mut line2 = String::from("! ");
			line2.push_str(&self.0);
			line2.push_str(&" ".repeat(79 - self.0.len()));
			line2.push_str("!\n");

			let line3 = line1.clone();
			line1.push_str(&line2);
			line1.push_str(&line3);

			Self(line1)
		}
	}

	impl Into<Msg> for &'static str {
		#[inline]
		fn into(self) -> Msg {
			Msg(self.to_owned())
		}
	}

	impl Into<Msg> for String {
		#[inline]
		fn into(self) -> Msg {
			Msg(self)
		}
	}
}

#[cfg(test)]
mod serde_tests {
	use super::*;

	#[test]
	fn read_settings() {
		let mut pathbuf = std::path::PathBuf::new();
		pathbuf.push(env!("CARGO_MANIFEST_DIR"));
		pathbuf.push("settings.toml");
		let settings: Settings =
			toml::from_str(std::fs::read_to_string(pathbuf).unwrap().as_str())
				.unwrap();

		assert_eq!(settings.force_field, "uff.atoms");
		assert_eq!(settings.helium.sigma, 2.58);
		assert_eq!(settings.helium.epsilon, 10.22);
		assert_eq!(settings.nitrogen.sigma, 3.314);
		assert_eq!(settings.temperature, 298.15);
		assert_eq!(settings.cutoff, 12.8);
		assert_eq!(settings.samples_per_atom, 500);
		assert_eq!(settings.cubelet_size, 0.2);
		assert_eq!(settings.max_pore_diameter, 20.0);
		assert_eq!(settings.psd_bin_size, 0.25);
		assert_eq!(settings.visualization_mode, "none");
	}
}
