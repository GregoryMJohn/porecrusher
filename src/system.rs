use std::collections::HashMap;

use rayon::prelude::*;

use crate::{
	atoms::{
		Area, AreaPerMass, AreaPerVolume, AtomTypeInfo, Atoms, Density,
		Element, Mass, Radius, Scalar, Volume,
	},
	lattice::{Cubelet, Lattice},
	linalg::{Dot, Vec3},
	unit_cell::UnitCell,
	utils::{Helium, Nitrogen, Settings},
};

const SURFACE_COEFF: f64 = 1.122462048309373;
const SURFACE_COEFF2: f64 = 1.2599210498948732;
const FOUR_PI: f64 = 4.0 * std::f64::consts::PI;

pub struct System {
	cell:               UnitCell,
	atoms:              Atoms,
	atom_nitro_radii:   HashMap<Element, f64>,
	atom_nitro_radii2:  HashMap<Element, f64>,
	atom_helium_radii2: HashMap<Element, f64>,
	helium:             Helium,
	nitrogen:           Nitrogen,
	cutoff:             Radius,
}

impl System {
	#[inline]
	pub fn new(cell: UnitCell, atoms: Atoms, settings: &Settings) -> Self {
		let mut atom_nitro_radii = HashMap::new();
		let mut atom_nitro_radii2 = HashMap::new();
		let mut atom_helium_radii2 = HashMap::new();
		let nitrogen = settings.nitrogen().clone();
		let helium = settings.helium().clone();

		for atom_type_info in atoms.atom_type_info().values() {
			let r_n =
				0.5 * (atom_type_info.sigma() + nitrogen.sigma()).as_f64();
			let r_n2 = r_n * r_n;
			let r_he = 0.5 * (atom_type_info.sigma() + helium.sigma()).as_f64();
			let r_he2 = r_he * r_he;

			let element = Element::from(atom_type_info.element().to_owned());

			atom_nitro_radii.insert(element.clone(), r_n);
			atom_nitro_radii2.insert(element.clone(), r_n2);
			atom_helium_radii2.insert(element.clone(), r_he2);
		}

		let mut system = Self {
			cell,
			atoms,
			atom_nitro_radii,
			atom_nitro_radii2,
			atom_helium_radii2,
			helium,
			nitrogen,
			cutoff: settings.cutoff().into(),
		};

		for mcoord in system.atoms.mcoords_mut() {
			system.cell.transform_slant(mcoord);
		}

		let min_x = *system
			.atoms
			.xs_mut()
			.iter()
			.min_by(|a, b| a.partial_cmp(b).unwrap())
			.unwrap();

		let min_y = *system
			.atoms
			.ys_mut()
			.iter()
			.min_by(|a, b| a.partial_cmp(b).unwrap())
			.unwrap();

		let min_z = *system
			.atoms
			.zs_mut()
			.iter()
			.min_by(|a, b| a.partial_cmp(b).unwrap())
			.unwrap();

		let mut min_xyz = Vec3::from([min_x, min_y, min_z]);

		for mcoord in system.mcoords_mut() {
			*mcoord -= min_xyz;
		}

		system.cell.transform_unslant(&mut min_xyz);
		let [min_x, min_y, min_z] = min_xyz.as_array();
		for x in system.atoms.xs_mut().iter_mut() {
			*x -= min_x;
		}

		for y in system.atoms.ys_mut().iter_mut() {
			*y -= min_y;
		}

		for z in system.atoms.zs_mut().iter_mut() {
			*z -= min_z;
		}

		system
	}

	#[inline]
	pub fn atoms(&self) -> &Atoms {
		&self.atoms
	}

	#[inline]
	fn mcoords_mut(&mut self) -> &mut Vec<Vec3<f64>> {
		self.atoms.mcoords_mut()
	}

	#[inline]
	fn matoms(&self) -> impl Iterator<Item = (&AtomTypeInfo, Vec3<f64>)> {
		self.atoms.matoms()
	}

	#[inline]
	pub fn atom_helium_radius2(&self, element: &Element) -> Option<&f64> {
		self.atom_helium_radii2.get(element)
	}

	#[inline]
	pub fn mass(&self) -> Mass {
		self.atoms.total_mass()
	}

	#[inline]
	pub fn molar_mass(&self) -> Mass {
		self.atoms.total_molar_mass()
	}

	#[inline]
	pub fn volume(&self) -> Volume {
		self.cell.volume()
	}

	#[inline]
	pub fn density(&self) -> Density {
		self.mass() / (self.volume() * 1.0E-24)
	}

	#[inline]
	pub fn x_len(&self) -> f64 {
		self.cell.a()
	}

	#[inline]
	pub fn y_len(&self) -> f64 {
		self.cell.b()
	}

	#[inline]
	pub fn z_len(&self) -> f64 {
		self.cell.c()
	}

	#[inline]
	pub fn transform_unslant(&self, vector: &mut Vec3<f64>) {
		self.cell.transform_unslant(vector);
	}

	#[inline]
	pub fn transform_slant(&self, vector: &mut Vec3<f64>) {
		self.cell.transform_slant(vector);
	}

	pub fn minimum_image_dist2(&self, a: Vec3<f64>, b: Vec3<f64>) -> f64 {
		let cell_dims = Vec3::from([self.x_len(), self.y_len(), self.z_len()]);
		let mut vector = a - b;
		vector -= cell_dims * (vector / cell_dims).round();

		if !self.cell.is_ortho() {
			self.cell.transform_slant(&mut vector);
		}

		vector.dot(vector)
	}

	pub fn surface_area(&self, lattice: &Lattice, settings: &Settings) -> Area {
		let samples = settings.samples_per_atom();
		let atoms: Vec<(usize, (&AtomTypeInfo, Vec3<f64>))> =
			self.atoms.atoms().enumerate().collect::<Vec<_>>();

		let surface_area: f64 = atoms
			.par_iter()
			.map(|(i, (atom, coord))| {
				let mut free: u64 = 0;

				for _ in 0..samples {
					let r = Radius::from(
						*self
							.atom_nitro_radii
							.get(&Element::from(atom.element().to_owned()))
							.unwrap(),
					) * SURFACE_COEFF;

					let mut probe = Vec3::random_point_on_sphere(r);

					probe += *coord;

					if self.cell.is_ortho() {
						self.cell.map_to_cell(&mut probe);
					} else {
						self.cell.transform_slant(&mut probe);
					}

					if let Cubelet::InsideAtom = *lattice.get_from_coord(probe)
					{
						continue;
					}

					let mut collided = false;
					for (j, (atom2, coord)) in self.matoms().enumerate() {
						let mut slanted_coord = coord.clone();
						self.cell.transform_slant(&mut slanted_coord);
						if *i == j {
							continue;
						} else if self.minimum_image_dist2(probe, slanted_coord)
							< SURFACE_COEFF2
								* *self
									.atom_nitro_radii2
									.get(&Element::from(
										atom2.element().to_owned(),
									))
									.unwrap()
						{
							collided = true;
							break;
						}
					}
					if collided {
						continue;
					}
					free += 1;
				}

				(free as f64 / samples as f64)
					* FOUR_PI * SURFACE_COEFF2
					* self
						.atom_nitro_radii2
						.get(&Element::from(atom.element().to_owned()))
						.unwrap()
			})
			.sum();

		surface_area.into()
	}

	pub fn surface_area_per_volume(&self, surface_area: Area) -> AreaPerVolume {
		surface_area / (self.volume() * 1.0E-4)
	}

	pub fn surface_area_per_mass(
		&self,
		sa_per_vol: AreaPerVolume,
	) -> AreaPerMass {
		sa_per_vol / (self.mass() / (self.volume() * 1.0E-24))
	}

	pub fn is_geom_accessible(&self, coord: Vec3<f64>) -> Cubelet {
		let mut rdist_surface_min = None;
		let mut rdist2_min = None;
		let mut lj_energy_tot = 0.0;
		let cutoff2 = (self.cutoff * self.cutoff).as_f64();

		for (atom, atom_coord) in self.matoms() {
			let rdist2 = self.minimum_image_dist2(coord, atom_coord);

			if rdist2 < 0.25 * atom.sigma2() {
				return Cubelet::InsideAtom;
			}
			let rdist_surface = rdist2.sqrt() - atom.radius().as_f64();
			rdist_surface_min = match rdist_surface_min {
				| None => Some(rdist_surface),
				| Some(value) => Some(rdist_surface.min(value)),
			};
			rdist2_min = match rdist2_min {
				| None => Some(rdist2),
				| Some(value) => Some(rdist2.min(value)),
			};

			if rdist2 < cutoff2 {
				let sig2_rdist2 = self
					.atom_helium_radii2
					.get(&atom.element().into())
					.unwrap() / rdist2;
				let rdist6 = sig2_rdist2.powf(3.0);
				let rdist12 = rdist6 * rdist6;
				let lj_energy = (self.helium.clone().espilon()
					* atom.epsilon())
				.as_f64()
				.sqrt() * (rdist12 - rdist6);
				lj_energy_tot += lj_energy;
			}
		}
		let rdist2_min = rdist2_min.unwrap();
		let min2 = rdist_surface_min.unwrap().powf(2.);

		if rdist2_min > self.nitrogen.sigma().as_f64().powf(2.) {
			Cubelet::NitrogenAccessible {
				min_dist2: min2,
				lj_energy: lj_energy_tot,
			}
		} else if rdist2_min > self.helium.sigma().to_radius().as_f64().powf(2.)
		{
			Cubelet::HeliumAccessible {
				min_dist2: min2,
				lj_energy: lj_energy_tot,
			}
		} else {
			Cubelet::GeomAccessible {
				min_dist2: min2
			}
		}
	}
}
