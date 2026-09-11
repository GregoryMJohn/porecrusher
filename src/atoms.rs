use std::{
	collections::HashMap,
	fmt::{self, Display},
	iter::Sum,
	ops::{Add, AddAssign, Div, Mul, Sub},
	path::Path,
};

use macros::Scalar;

use crate::linalg::Vec3;

const AVOGADRO: f64 = 6.02214076E23;

pub trait Scalar:
	Sized
	+ Copy
	+ Clone
	+ Add
	+ Sub
	+ Mul
	+ Mul<f64>
	+ Div
	+ Div<f64>
	+ Into<f64>
	+ From<f64>
	+ Display
{
	fn as_f64(self) -> f64;
}

pub struct Atoms {
	atom_type_info: HashMap<AtomType, AtomTypeInfo>,
	atom_types:     Vec<AtomType>,
	xs:             Vec<f64>,
	ys:             Vec<f64>,
	zs:             Vec<f64>,
	mcoords:        Vec<Vec3<f64>>,
}

impl Atoms {
	pub fn from_xyz_file(xyz_file: &Path) -> Self {
		let mut atom_types = Vec::new();
		let mut n_unique_atomtypes = 0u8;
		let mut xs = Vec::new();
		let mut ys = Vec::new();
		let mut zs = Vec::new();

		let mut atomtype_to_element = HashMap::new();
		//let mut n_atom_types = 0;
		let mut element_to_atomtype = HashMap::new();

		let contents = std::fs::read_to_string(xyz_file)
			.expect(format!("File '{:?}' not found", xyz_file).as_str());

		let lines = contents.lines().collect::<Vec<&str>>();

		let (header, lines) = lines.split_at(2);
		let n_atoms = header[0].parse::<usize>().expect(
			"Invalid entry for number of atoms on line 1 of '{xyz_file}'",
		);

		let mut element_list = Vec::new();

		for (i, line) in lines.iter().enumerate() {
			let line_num = i + 2;
			let [element, x, y, z] =
				line.split_whitespace().collect::<Vec<&str>>()[..]
			else {
				panic!(
					"Invalid entry in {:?} on line {line_num}: \n'{line}'\n\n",
					xyz_file
				)
			};

			let element = Element(element.to_string());

			match element_to_atomtype.get(&element) {
				| Some(atom_type) => atom_types.push(*atom_type),
				| None => {
					let atom_type = AtomType(n_unique_atomtypes);
					n_unique_atomtypes += 1;
					atom_types.push(atom_type);
					atomtype_to_element.insert(atom_type, element.clone());
					element_to_atomtype.insert(element.clone(), atom_type);
					element_list.push(element.clone())
				}
			}

			xs.push(
				x.parse::<f64>().expect(
					format!(
						"Invalid entry in x-coordinate for atom {i} on line {line_num} of {:?}",
						xyz_file
					)
					.as_str(),
				),
			);

			ys.push(
				y.parse::<f64>().expect(
					format!(
						"Invalid entry in y-coordinate for atom {i} on line {line_num} of {:?}",
						xyz_file
					)
					.as_str(),
				),
			);

			zs.push(
				z.parse::<f64>().expect(
					format!(
						"Invalid entry in z-coordinate for atom {i} on line {line_num} of {:?}",
						xyz_file
					)
					.as_str(),
				),
			);
		}

		if atom_types.len() != n_atoms {
			panic!(
				"Number of entries in XYZ file ({n_atoms}) does not match number of atoms listed on line 1 ({}).",
				atom_types.len()
			)
		}

		let atom_type_info = Self::initialize_atom_types(&atomtype_to_element);
		let mut mcoords = Vec::new();
		for i in 0..xs.len() {
			mcoords.push(Vec3::from([xs[i], ys[i], zs[i]]));
		}

		Self {
			atom_type_info,
			atom_types,
			xs,
			ys,
			zs,
			mcoords,
		}
	}

	fn initialize_atom_types(
		elements: &HashMap<AtomType, Element>,
	) -> HashMap<AtomType, AtomTypeInfo> {
		let mut uff = std::env::current_dir().unwrap();
		uff.push("uff.atoms");

		let contents = std::fs::read_to_string(&uff)
			.expect(format!("File '{:?}' not found", uff).as_str());

		let mut atom_type_infos = HashMap::new();

		let lines = contents.lines().collect::<Vec<&str>>();
		for (atomtype, elem) in elements.iter() {
			for line in &lines {
				let fields = line.split_whitespace().collect::<Vec<_>>();
				let [element, mass, sigma, epsilon] = fields[..] else {
					panic!(
						"Expected 3 fields (mass, sigma, epsilon), but only {} were found:\n\n{line}'",
						fields.len(),
					)
				};

				if element == elem.name() {
					let sigma = sigma.parse::<f64>().unwrap();
					let radius = Radius(sigma / 2.0);
					let sigma = Sigma(sigma);
					let mass = Mass(mass.parse::<f64>().unwrap());
					let epsilon = Epsilon(epsilon.parse::<f64>().unwrap());

					atom_type_infos.insert(
						atomtype.clone(),
						AtomTypeInfo {
							element: Element(String::from(element)),
							radius,
							mass,
							sigma,
							epsilon,
						},
					);
				}
			}
		}

		atom_type_infos
	}

	#[inline]
	pub fn mcoords(&self) -> &Vec<Vec3<f64>> {
		&self.mcoords
	}

	#[inline]
	pub fn mcoords_mut(&mut self) -> &mut Vec<Vec3<f64>> {
		&mut self.mcoords
	}

	pub fn atoms(&self) -> impl Iterator<Item = (&AtomTypeInfo, Vec3<f64>)> {
		self.atom_types.iter().enumerate().map(|(i, atype)| {
			let info = self.atom_type_info.get(atype).unwrap();
			let x = self.xs[i];
			let y = self.ys[i];
			let z = self.zs[i];

			(info, Vec3::from([x, y, z]))
		})
	}

	pub fn matoms(&self) -> impl Iterator<Item = (&AtomTypeInfo, Vec3<f64>)> {
		self.atom_types.iter().enumerate().map(|(i, atype)| {
			let info = self.atom_type_info.get(atype).unwrap();

			(info, self.mcoords[i])
		})
	}

	#[inline]
	pub fn atom_type_info(&self) -> &HashMap<AtomType, AtomTypeInfo> {
		&self.atom_type_info
	}

	#[inline]
	pub fn xs_mut(&mut self) -> &mut Vec<f64> {
		&mut self.xs
	}

	#[inline]
	pub fn ys_mut(&mut self) -> &mut Vec<f64> {
		&mut self.ys
	}

	#[inline]
	pub fn zs_mut(&mut self) -> &mut Vec<f64> {
		&mut self.zs
	}

	#[inline]
	pub fn xs(&self) -> &Vec<f64> {
		&self.xs
	}

	#[inline]
	pub fn ys(&self) -> &Vec<f64> {
		&self.ys
	}

	#[inline]
	pub fn zs(&self) -> &Vec<f64> {
		&self.zs
	}

	#[inline]
	pub fn total_molar_mass(&self) -> Mass {
		self.masses().sum()
	}

	#[inline]
	pub fn total_mass(&self) -> Mass {
		self.total_molar_mass() / AVOGADRO
	}

	#[inline]
	fn masses(&self) -> impl Iterator<Item = Mass> {
		self.atom_types
			.iter()
			.map(|atype| self.get_atom_type_info(*atype).mass)
	}

	#[inline]
	fn get_atom_type_info(&self, atom_type: AtomType) -> &AtomTypeInfo {
		&self.atom_type_info.get(&atom_type).unwrap()
	}
}

#[derive(Eq, PartialEq, Ord, PartialOrd, Hash, Clone, Debug)]
pub struct Element(String);
impl Element {
	#[inline]
	fn name(&self) -> &str {
		&self.0
	}
}

impl From<String> for Element {
	fn from(value: String) -> Self {
		Self(value)
	}
}

impl From<&str> for Element {
	fn from(value: &str) -> Self {
		Self(value.to_owned())
	}
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct AtomType(u8);

#[derive(Debug)]
pub struct AtomTypeInfo {
	element: Element,
	radius:  Radius,
	mass:    Mass,
	sigma:   Sigma,
	epsilon: Epsilon,
}

impl AtomTypeInfo {
	pub fn new(
		element: String,
		radius: f64,
		mass: f64,
		sigma: f64,
		epsilon: f64,
	) -> Self {
		let radius = Radius(radius);
		let mass = Mass(mass);
		let sigma = Sigma(sigma);
		let epsilon = Epsilon(epsilon);
		let element = Element(element);

		Self {
			element,
			radius,
			mass,
			sigma,
			epsilon,
		}
	}

	#[inline]
	pub fn element(&self) -> &str {
		self.element.name()
	}

	#[inline]
	pub const fn radius(&self) -> Radius {
		self.radius
	}

	#[inline]
	pub const fn mass(&self) -> Mass {
		self.mass
	}

	#[inline]
	pub const fn sigma(&self) -> Sigma {
		self.sigma
	}

	#[inline]
	pub fn sigma2(&self) -> f64 {
		(self.sigma() * self.sigma()).into()
	}

	#[inline]
	pub fn radius2(&self) -> f64 {
		(self.radius() * self.radius()).into()
	}

	#[inline]
	pub const fn epsilon(&self) -> Epsilon {
		self.epsilon
	}
}

#[derive(Copy, Clone, Scalar, Debug)]
pub struct Radius(f64);
impl Radius {
	pub fn new(value: f64) -> Self {
		Self(value)
	}
}

#[derive(Copy, Clone, Scalar, Debug)]
pub struct Mass(f64);

impl Sum for Mass {
	fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
		iter.fold(Mass(-0.0), |a, b| a + b)
	}
}

#[derive(Copy, Clone, Scalar)]
pub struct Volume(f64);

#[derive(Copy, Clone, Scalar)]
pub struct Density(f64);

impl Div<Volume> for Mass {
	type Output = Density;

	fn div(self, rhs: Volume) -> Self::Output {
		Density(self.0 / rhs.0)
	}
}

#[derive(Copy, Clone, Scalar, Debug)]
pub struct Sigma(f64);

impl Sigma {
	pub fn to_radius(self) -> Radius {
		Radius(self.as_f64() * 0.5)
	}
}

#[derive(Copy, Clone, Scalar, Debug)]
pub struct Epsilon(f64);

#[derive(Copy, Clone, Scalar)]
pub struct Radians(f64);

impl Radians {
	#[inline]
	pub fn sin(self) -> Self {
		Self(self.0.sin())
	}

	#[inline]
	pub fn cos(self) -> Self {
		Self(self.0.cos())
	}

	#[inline]
	pub fn acos(self) -> Self {
		Self(self.0.acos())
	}
}

impl From<[Radians; 3]> for Vec3<f64> {
	fn from(value: [Radians; 3]) -> Self {
		let value: [f64; 3] = *value
			.iter()
			.map(|x| x.0)
			.collect::<Vec<f64>>()
			.as_array()
			.unwrap();

		Self::from(value)
	}
}

#[derive(Copy, Clone, Scalar)]
pub struct Area(f64);

impl AddAssign<f64> for Area {
	#[inline]
	fn add_assign(&mut self, rhs: f64) {
		*self = *self + Self(rhs)
	}
}

#[derive(Copy, Clone, Scalar)]
pub struct AreaPerVolume(f64);

impl Div<Volume> for Area {
	type Output = AreaPerVolume;

	#[inline]
	fn div(self, rhs: Volume) -> Self::Output {
		AreaPerVolume(self.0 / rhs.0)
	}
}

#[derive(Copy, Clone, Scalar)]
pub struct AreaPerMass(f64);

impl Div<Density> for AreaPerVolume {
	type Output = AreaPerMass;

	#[inline]
	fn div(self, rhs: Density) -> Self::Output {
		AreaPerMass(self.0 / rhs.0)
	}
}
