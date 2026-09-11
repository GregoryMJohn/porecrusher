use std::{
	collections::HashMap,
	fs::File,
	io::{Error, Write},
	ops::{Add, AddAssign, Index, IndexMut},
	sync::Arc,
	time::Duration,
	vec::IntoIter,
};

use console::style;
use indicatif::{ProgressBar, ProgressStyle};
use num::ToPrimitive;
use rayon::prelude::*;

use crate::{
	atoms::{Scalar, Volume},
	linalg::Vec3,
	system::System,
	utils::{Settings, io},
};

pub struct Lattice {
	cubelets: Vec<Cubelet>,
	cubesize: f64,
	x:        usize,
	y:        usize,
	z:        usize,
}

impl Lattice {
	pub fn new(
		system: &System,
		mut settings: Arc<Settings>,
		sender: io::Sender<io::Msg>,
	) -> Self {
		let x = (system.x_len() / settings.cubelet_size()).round() as usize;
		*Arc::make_mut(&mut settings).cubelet_size_mut() =
			system.x_len() / x as f64;
		sender
			.send(
				format!(
					"Corrected cubelet size to {}Å based on system geometry",
					settings.cubelet_size()
				)
				.into(),
			)
			.unwrap();

		let x = (system.x_len() / settings.cubelet_size()).round() as usize;
		let y = (system.y_len() / settings.cubelet_size()).round() as usize;
		let z = (system.z_len() / settings.cubelet_size()).round() as usize;

		let cubesize = settings.cubelet_size();

		let bar = ProgressBar::new_spinner();
		bar.set_style(
			ProgressStyle::with_template(
				"{prefix:.bold.dim} {spinner} {wide_msg}",
			)
			.unwrap(),
		);
		//bar.set_prefix(format!("{}", style("[1/1]").bold().dim()));
		bar.set_message("🔍 Checking accessibility of cubelets...");
		bar.enable_steady_tick(Duration::from_millis(100));

		let cubelets = (0..z)
			.into_par_iter()
			.flat_map(|k| {
				(0..y)
					.into_iter()
					.flat_map(|j| (0..x).into_iter().map(move |i| (i, j, k)))
					.map(|(i, j, k)| {
						let mut coord =
							Vec3::from([i as f64, j as f64, k as f64]);
						coord *= cubesize;
						coord += Vec3::splat(cubesize * 0.5);

						system.is_geom_accessible(coord)
					})
					.collect::<Vec<_>>()
			})
			.collect::<Vec<_>>();

		bar.finish_and_clear();

		Self {
			cubelets,
			cubesize,
			x,
			y,
			z,
		}
	}

	pub fn geom_accessible_cubelets(&self) -> Vec<usize> {
		self.cubelets
			.iter()
			.enumerate()
			.filter(|(_, cubelet)| !matches!(cubelet, Cubelet::InsideAtom))
			.map(|(i, _)| i)
			.collect()
	}

	pub fn get_from_coord(&self, coord: Vec3<f64>) -> &Cubelet {
		let nxyz = coord
			.iter()
			.zip(self.dimensions())
			.map(|(n, d)| {
				let mut u = (n / self.cubesize) as usize + 1;
				if u > d {
					u -= (u / d) * d;
				} else if u < 1 {
					u += (1 - (u / d)) * d;
				}
				u - 1
			})
			.collect::<Vec<usize>>();
		let [nx, ny, nz] = nxyz.as_array::<3>().unwrap();
		&self[(*nx, *ny, *nz)]
	}

	#[inline]
	pub fn cubelets(&self) -> &[Cubelet] {
		self.cubelets.as_slice()
	}

	#[inline]
	pub fn cubesize(&self) -> f64 {
		self.cubesize
	}

	#[inline]
	pub fn dimensions(&self) -> [usize; 3] {
		[self.x, self.y, self.z]
	}

	#[inline]
	pub const fn ntot_cubelets(&self) -> usize {
		self.x * self.y * self.z
	}

	#[inline]
	pub fn is_geom_accessible(&self, i: usize, j: usize, k: usize) -> bool {
		!matches!(self[(i, j, k)], Cubelet::InsideAtom)
	}

	#[inline]
	pub fn is_nitrogen_accessible(&self, i: usize, j: usize, k: usize) -> bool {
		matches!(self[(i, j, k)], Cubelet::NitrogenAccessible { .. })
	}

	#[inline]
	pub fn is_helium_accessible(&self, i: usize, j: usize, k: usize) -> bool {
		matches!(
			self[(i, j, k)],
			Cubelet::HeliumAccessible { .. }
				| Cubelet::NitrogenAccessible { .. }
		)
	}

	pub fn center_of_cubelet(&self, i: usize, j: usize, k: usize) -> Vec3<f64> {
		let mut coord = Vec3::from([i, j, k]) * self.cubesize;
		coord += Vec3::splat(self.cubesize * 0.5);
		coord
	}

	pub fn geom_accessible_grid(&self) -> Grid {
		let mut grid = Grid::new(self.x, self.y, self.z);
		grid.populate(self, None);
		grid
	}

	pub fn helium_accessible_grid(&self) -> Grid {
		let mut grid = Grid::new(self.x, self.y, self.z);
		grid.populate(self, Some(ProbeType::He));
		grid
	}

	pub fn nitrogen_accessible_grid(&self) -> Grid {
		let mut grid = Grid::new(self.x, self.y, self.z);
		grid.populate(self, Some(ProbeType::N));
		grid
	}

	pub fn max_cubelet_min_dist2(&self) -> f64 {
		let cubes = self
			.into_iter()
			.map(|cubelet| cubelet.minimum_distance2())
			.collect::<Vec<_>>();

		*cubes
			.iter()
			.filter_map(|x| Option::as_ref(&x))
			.max_by(|a, b| a.partial_cmp(&b).unwrap())
			.unwrap()
	}

	pub fn min_cubelet_min_dist2(&self) -> f64 {
		let cubes = self
			.into_iter()
			.map(|cubelet| cubelet.minimum_distance2())
			.collect::<Vec<_>>();

		*cubes
			.iter()
			.filter_map(|x| Option::as_ref(&x))
			.min_by(|a, b| a.partial_cmp(&b).unwrap())
			.unwrap()
	}

	pub fn pore_volume_he(&self, system: &System, temperature: f64) -> Volume {
		let mut lj_energy_tot = 0.0;
		for cubelet in self.cubelets.iter() {
			if let Cubelet::HeliumAccessible {
				lj_energy, ..
			}
			| Cubelet::NitrogenAccessible {
				lj_energy, ..
			} = cubelet
			{
				let mut lj_energy = 4.0 * lj_energy;
				lj_energy = f64::exp(-lj_energy / temperature);
				lj_energy_tot += lj_energy;
			}
		}

		(system.volume().as_f64() * lj_energy_tot
			/ self.ntot_cubelets().to_f64().unwrap())
		.into()
	}

	pub fn pore_volume_geom(&self, system: &System) -> Volume {
		let mut count: usize = 0;

		for cubelet in self.cubelets.iter() {
			if let Cubelet::InsideAtom = cubelet {
				continue;
			}

			count += 1;
		}

		Volume::from(
			system.volume() * count as f64 / self.ntot_cubelets() as f64,
		)
	}

	pub fn ijk_from_index(&self, index: usize) -> (usize, usize, usize) {
		let mut i = index;
		let xy = self.x * self.y;
		let k = i / xy;
		i -= xy * k;
		let j = i / self.x;
		i -= j * self.x;
		(i, j, k)
	}

	pub fn index_from_ijk(&self, ijk: (usize, usize, usize)) -> usize {
		let (i, j, k) = ijk;
		k * self.x * self.y + j * self.x + i
	}

	pub fn rand_avail_cubelet(&self) -> (&Cubelet, (usize, usize, usize)) {
		let n = self.ntot_cubelets();
		let mut i = rand::random_range(0..n);
		while let Cubelet::InsideAtom = self.cubelets[i] {
			i = rand::random_range(0..n);
		}

		let ijk = self.ijk_from_index(i);
		(&self.cubelets[i], ijk)
	}
}

impl Index<(usize, usize, usize)> for Lattice {
	type Output = Cubelet;

	#[inline]
	fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
		let (i, j, k) = index;
		&self.cubelets[k * self.x * self.y + j * self.x + i]
	}
}

impl IndexMut<(usize, usize, usize)> for Lattice {
	#[inline]
	fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
		let (x, y, z) = index;
		let i = z * self.x * self.y + y * self.x + x;
		&mut self.cubelets[i]
	}
}

impl IntoIterator for Lattice {
	type IntoIter = IntoIter<Cubelet>;
	type Item = Cubelet;

	fn into_iter(self) -> Self::IntoIter {
		self.cubelets.into_iter()
	}
}

impl<'a> IntoIterator for &'a Lattice {
	type IntoIter = std::slice::Iter<'a, Cubelet>;
	type Item = &'a Cubelet;

	fn into_iter(self) -> Self::IntoIter {
		self.cubelets.iter()
	}
}

impl<'a> IntoIterator for &'a mut Lattice {
	type IntoIter = std::slice::IterMut<'a, Cubelet>;
	type Item = &'a mut Cubelet;

	fn into_iter(self) -> Self::IntoIter {
		self.cubelets.iter_mut()
	}
}

#[derive(Copy, Clone)]
pub enum Cubelet {
	HeliumAccessible { min_dist2: f64, lj_energy: f64 },
	NitrogenAccessible { min_dist2: f64, lj_energy: f64 },
	GeomAccessible { min_dist2: f64 },
	InsideAtom,
}

impl Cubelet {
	fn minimum_distance2(self) -> Option<f64> {
		match self {
			| Self::HeliumAccessible {
				min_dist2, ..
			}
			| Self::NitrogenAccessible {
				min_dist2, ..
			}
			| Self::GeomAccessible {
				min_dist2,
			} => Some(min_dist2),
			| Self::InsideAtom => None,
		}
	}
}

#[derive(Copy, Clone)]
pub enum ProbeType {
	He,
	N,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Ord, PartialOrd, Debug)]
pub struct Cluster(u32);

impl From<u32> for Cluster {
	fn from(value: u32) -> Self {
		Self(value)
	}
}

impl Add<u32> for Cluster {
	type Output = Self;

	fn add(self, rhs: u32) -> Self::Output {
		Self(self.0 + rhs)
	}
}

impl AddAssign<u32> for Cluster {
	fn add_assign(&mut self, rhs: u32) {
		*self = *self + rhs;
	}
}

pub struct Grid {
	labels:   Vec<Cluster>,
	clusters: HashMap<Cluster, Vec<(usize, usize, usize)>>,
	x_len:    usize,
	y_len:    usize,
	z_len:    usize,
}

impl Grid {
	pub fn new(x_len: usize, y_len: usize, z_len: usize) -> Self {
		let labels = vec![Cluster(0); x_len * y_len * z_len];
		let clusters = HashMap::new();

		Self {
			labels,
			clusters,
			x_len,
			y_len,
			z_len,
		}
	}

	pub fn clear(&mut self) {
		for label in self.labels.iter_mut() {
			*label = Cluster(0);
		}

		self.clusters.drain();
	}

	pub fn populate(&mut self, lattice: &Lattice, probe: Option<ProbeType>) {
		let is_accessible = match probe {
			| Some(probetype) => match probetype {
				| ProbeType::He => Lattice::is_helium_accessible,
				| ProbeType::N => Lattice::is_nitrogen_accessible,
			},
			| None => Lattice::is_geom_accessible,
		};

		for k in 0..lattice.z {
			for j in 0..lattice.y {
				for i in 0..lattice.x {
					if is_accessible(lattice, i, j, k) {
						self[(i, j, k)] = 1.into();
					}
				}
			}
		}
	}

	fn sorted_cubelets(
		&self,
		lattice: &Lattice,
	) -> Vec<(f64, (usize, usize, usize))> {
		let mut cubes = Vec::new();

		for k in 0..lattice.z {
			for j in 0..lattice.y {
				for i in 0..lattice.x {
					if let Some(r) = lattice[(i, j, k)].minimum_distance2() {
						cubes.push((r, (i, j, k)));
					}
				}
			}
		}

		cubes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
		cubes
	}

	fn sorted_nitrogen_cubelets(
		&self,
		lattice: &Lattice,
	) -> Vec<(f64, (usize, usize, usize))> {
		let mut cubes = Vec::new();

		for k in 0..lattice.z {
			for j in 0..lattice.y {
				for i in 0..lattice.x {
					if let Cubelet::NitrogenAccessible {
						min_dist2, ..
					} = lattice[(i, j, k)]
					{
						cubes.push((min_dist2, (i, j, k)));
					}
				}
			}
		}

		cubes.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
		cubes
	}

	pub fn limiting_diameter(&self, lattice: &Lattice) -> (f64, f64) {
		let cubes = self.sorted_cubelets(lattice);

		let rmax = lattice.max_cubelet_min_dist2();
		let mut rhigh = rmax;
		let mut rlow = lattice.min_cubelet_min_dist2();
		let mut rdiff = 0.0;

		let mut grid = Self::new(self.x_len, self.y_len, self.z_len);
		for _ in 0..100000 {
			let rdiff_old = rdiff;
			rdiff = rhigh - rlow;
			if rdiff.abs() < 0.25 || rdiff == rdiff_old {
				break;
			}

			let rmiddle = 0.5 * (rhigh + rlow);

			grid.clear();
			let mut j = 0;

			for (i, (min_dist2, ijk)) in cubes.iter().enumerate().rev() {
				j = i;
				if *min_dist2 < rmiddle {
					break;
				}

				grid[*ijk] = Cluster(1);
			}

			grid.percolate();

			match grid.span() {
				| Some(_) => {
					rlow = cubes[j - 1].0;
				}
				| None => {
					rhigh = cubes[j + 1].0;
				}
			}
		}

		(2. * rhigh.sqrt(), 2. * rmax.sqrt())
	}

	pub fn pore_size_distribution(
		&self,
		lattice: &Lattice,
		system: &System,
		settings: &Settings,
	) -> Result<(Volume, f64), Error> {
		let max_pore_diameter = settings.max_pore_diameter();
		let nbins = (max_pore_diameter / settings.psd_bin_size()) as usize;
		let binsize = max_pore_diameter / nbins as f64;
		let mut psd = vec![0.; nbins + 100];
		let mut psd_cumul = vec![0.; nbins + 100];
		let cubes = self.sorted_nitrogen_cubelets(lattice);
		let mut r2_ref;
		let mut ffv = 0.0;

		let mut ivis: usize = 0;
		let mut connolly_volume = vec![0; lattice.ntot_cubelets()];

		let ngeom_avail_cubes = lattice.geom_accessible_cubelets().len();

		const N: usize = 10_000;
		let bar = ProgressBar::new_spinner();
		bar.set_style(
			ProgressStyle::with_template(
				"{prefix:.bold.dim} {spinner} {wide_msg}",
			)
			.unwrap(),
		);
		bar.set_prefix(format!("{}", style("[1/4]").bold().dim()));
		bar.set_message("Calculating free volume...");
		bar.enable_steady_tick(Duration::from_millis(100));

		for _ in 0..N {
			r2_ref = 0.0;

			let (_, (i, j, k)) = lattice.rand_avail_cubelet();

			let index = lattice.index_from_ijk((i, j, k));

			let coord = lattice.center_of_cubelet(i, j, k);

			for (mindist2, (i, j, k)) in cubes.iter().rev() {
				let coord2 = lattice.center_of_cubelet(*i, *j, *k);

				let r2 = system.minimum_image_dist2(coord, coord2);

				if r2 > *mindist2 {
					continue;
				} else {
					r2_ref = *mindist2;

					ffv += 1.0;
					connolly_volume[ivis] = index;
					ivis += 1;
					break;
				}
			}

			if r2_ref == 0.0 {
				continue;
			}

			let mut bin = (2.0 * r2_ref.sqrt() / binsize) as usize;
			if bin >= psd_cumul.len() {
				bin = psd_cumul.len() - 1;
			}

			for m in 0..=bin {
				psd_cumul[m] += 1.;
			}
		}

		ffv /= N as f64;
		ffv *= ngeom_avail_cubes as f64 / lattice.ntot_cubelets() as f64;
		let free_volume = system.volume() * ffv;

		bar.set_prefix(format!("{}", style("[2/4]").bold().dim()));
		bar.set_message(
			"Calculating total cumulative pore size distribution...",
		);

		let mut psd_cumul_file = File::create("Total_psd_cumulative.txt")?;
		writeln!(
			psd_cumul_file,
			"# Cumulative acessible volume distribution as a function of probe diameter"
		)?;
		writeln!(psd_cumul_file, "# ")?;
		writeln!(psd_cumul_file, "# d(probe) 				Volume Fraction")?;

		let first = psd_cumul[0];
		psd_cumul.iter_mut().for_each(|x| *x /= first);

		for i in 0..nbins {
			writeln!(
				psd_cumul_file,
				"{:.3}			{:.9}",
				binsize * (i as f64) - binsize / 2.0,
				psd_cumul[i]
			)?;
		}

		bar.set_prefix(format!("{}", style("[3/4]").bold().dim()));
		bar.set_message("Calculating total pore size distribution...");

		psd[0] = 0.0;
		psd[1] = 0.0;
		psd[nbins - 1] = 0.0;

		for i in 1..(nbins - 1) {
			psd[i] =
				-1.0 * (psd_cumul[i + 1] - psd_cumul[i - 1]) / (binsize * 2.0);
		}

		let mut psd_file = File::create("Total_psd.txt")?;
		writeln!(
			psd_file,
			"# Derivative distribution function -dV(r)/dr (or -dV(d)/dd) vs d"
		)?;

		for i in 1..(nbins - 1) {
			writeln!(
				psd_file,
				"{:.3}			{:.9}",
				binsize * (i as f64) - binsize / 2.0,
				psd[i]
			)?;
		}

		let mut probe_occ_vol_file =
			File::create("probe_occupiable_volume.xyz")?;

		writeln!(probe_occ_vol_file, "{ivis}")?;
		writeln!(probe_occ_vol_file, "")?;

		bar.set_prefix(format!("{}", style("[4/4]").bold().dim()));
		bar.set_message("Writing coordinates of helium probes within Connolly volume to 'probe_occupiable_volume.xyz'...");

		for index in 0..ivis {
			let isite = connolly_volume[index];
			let (i, j, k) = lattice.ijk_from_index(isite);

			let mut coord = lattice.center_of_cubelet(i, j, k);

			coord += Vec3::from([
				*system
					.atoms()
					.xs()
					.iter()
					.min_by(|a, b| a.partial_cmp(b).unwrap())
					.unwrap(),
				*system
					.atoms()
					.ys()
					.iter()
					.min_by(|a, b| a.partial_cmp(b).unwrap())
					.unwrap(),
				*system
					.atoms()
					.zs()
					.iter()
					.min_by(|a, b| a.partial_cmp(b).unwrap())
					.unwrap(),
			]);

			system.transform_unslant(&mut coord);

			writeln!(
				probe_occ_vol_file,
				"He 		{:.15}		{:.15}		{:.15}",
				coord.x(),
				coord.y(),
				coord.z()
			)?;
		}

		bar.finish_and_clear();

		println!(
			" Total cumulative PSD and differential PSD have been stored in the following files: "
		);
		println!("	Total_psd_cumulative.txt");
		println!(" 	Total_psd.txt");

		Ok((free_volume, ffv))
	}

	pub fn percolate(&mut self) {
		let mut n_clusters: u32 = 0;

		for k in 0..self.z_len {
			for j in 0..self.y_len {
				for i in 0..self.x_len {
					if self[(i, j, k)] == Cluster(0) {
						continue
					}

					let mut neighbors = self.nearest_neighbors(i, j, k);

					neighbors.retain(|coord| self[*coord] != Cluster(0));

					let cell = if neighbors.is_empty() {
						n_clusters += 1;
						let cluster = Cluster(n_clusters);
						self.clusters.insert(cluster, vec![(i, j, k)]);
						cluster
					} else {
						let cluster = self.union(neighbors);
						self.clusters
							.get_mut(&cluster)
							.unwrap()
							.push((i, j, k));
						cluster
					};

					self[(i, j, k)] = cell;
				}
			}
		}
	}

	fn nearest_neighbors(
		&self,
		i: usize,
		j: usize,
		k: usize,
	) -> Vec<(usize, usize, usize)> {
		let mut neighbors = vec![];
		if k > 0 {
			neighbors.push((i, j, k - 1))
		}
		if j > 0 {
			neighbors.push((i, j - 1, k))
		}
		if i > 0 {
			neighbors.push((i - 1, j, k))
		}
		if k == self.z_len - 1 {
			neighbors.push((i, j, 0))
		}
		if j == self.y_len - 1 {
			neighbors.push((i, 0, k))
		}
		if i == self.x_len - 1 {
			neighbors.push((0, j, k))
		}
		neighbors.iter().map(|coord| *coord).collect::<Vec<_>>()
	}

	fn union(&mut self, cells: Vec<(usize, usize, usize)>) -> Cluster {
		let min = cells
			.iter()
			.map(|coord| self[*coord])
			.min()
			.unwrap()
			.clone();

		for cell in cells.iter() {
			let cluster = self[*cell];
			if cluster == min {
				continue
			}

			let old = self.clusters.remove(&cluster).unwrap();
			for ijk in old.iter() {
				self[ijk.clone()] = min;
			}
			let min_cluster = self.clusters.get_mut(&min).unwrap();
			for ijk in old {
				min_cluster.push(ijk);
			}

			self[*cell] = min;
		}

		min
	}

	pub fn span(&self) -> Option<SpanDimensions> {
		let mut spans = [false; 3];

		let mut x_array = Vec::new();
		let mut y_array = Vec::new();
		let mut z_array = Vec::new();

		for (cluster, cells) in self.clusters.iter() {
			let ncells = cells.len();
			if ncells >= self.z_len
				|| ncells >= self.y_len
				|| ncells >= self.x_len
			{
				for k in 0..self.z_len {
					'next_slice: for j in 0..self.y_len {
						for i in 0..self.x_len {
							if self[(i, j, k)] == *cluster {
								z_array.push(1);
								break 'next_slice
							}
						}
					}
				}

				if z_array.len() == self.z_len {
					spans[2] = true;
				}

				for j in 0..self.y_len {
					'next_slice: for k in 0..self.z_len {
						for i in 0..self.x_len {
							if self[(i, j, k)] == *cluster {
								y_array.push(1);
								break 'next_slice
							}
						}
					}
				}

				if y_array.len() == self.y_len {
					spans[1] = true;
				}

				for i in 0..self.x_len {
					'next_slice: for k in 0..self.z_len {
						for j in 0..self.y_len {
							if self[(i, j, k)] == *cluster {
								x_array.push(1);
								break 'next_slice
							}
						}
					}
				}

				if x_array.len() == self.x_len {
					spans[0] = true;
				}
			}
		}

		match spans.iter().map(|d| *d as u8).sum() {
			| 0 => None,
			| d => Some(d.into()),
		}
	}
}

pub enum SpanDimensions {
	One,
	Two,
	Three,
}

impl SpanDimensions {
	pub fn to_string(self) -> String {
		match self {
			| Self::One => "1",
			| Self::Two => "2",
			| Self::Three => "3",
		}
		.into()
	}
}

impl From<u8> for SpanDimensions {
	fn from(value: u8) -> Self {
		match value {
			| 1 => Self::One,
			| 2 => Self::Two,
			| 3 => Self::Three,
			| _ => panic!(
				"The system can't be percolated in more than 3 dimensions!"
			),
		}
	}
}

impl Index<(usize, usize, usize)> for Grid {
	type Output = Cluster;

	#[inline]
	fn index(&self, index: (usize, usize, usize)) -> &Self::Output {
		let (i, j, k) = index;
		&self.labels[k * self.x_len * self.y_len + j * self.x_len + i]
	}
}

impl IndexMut<(usize, usize, usize)> for Grid {
	#[inline]
	fn index_mut(&mut self, index: (usize, usize, usize)) -> &mut Self::Output {
		let (i, j, k) = index;
		&mut self.labels[k * self.x_len * self.y_len + j * self.x_len + i]
	}
}

impl IntoIterator for Grid {
	type IntoIter = IntoIter<Cluster>;
	type Item = Cluster;

	fn into_iter(self) -> Self::IntoIter {
		self.labels.into_iter()
	}
}

impl<'a> IntoIterator for &'a Grid {
	type IntoIter = std::slice::Iter<'a, Cluster>;
	type Item = &'a Cluster;

	fn into_iter(self) -> Self::IntoIter {
		self.labels.iter()
	}
}

impl<'a> IntoIterator for &'a mut Grid {
	type IntoIter = std::slice::IterMut<'a, Cluster>;
	type Item = &'a mut Cluster;

	fn into_iter(self) -> Self::IntoIter {
		self.labels.iter_mut()
	}
}

#[cfg(test)]
mod test {
	use super::*;

	#[test]
	fn compare_clusters() {
		let a = Cluster(0);
		let b = Cluster(1);

		assert!(a < b);
		assert!(b != 0.into());
		assert!(b > 0.into());
	}
}
