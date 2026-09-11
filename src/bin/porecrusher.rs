use std::{
	sync::{Arc, mpsc::channel},
	thread,
};

use porecrusher::{
	atoms::Scalar,
	lattice::Lattice,
	system::System,
	utils::{
		self,
		io::{self, Msg, Reporter},
	},
};

fn main() -> Result<(), std::io::Error>{
	let (atoms, unit_cell, settings) = utils::parse_cl_args();

	let settings = Arc::new(settings);

	thread::scope(|s| {
		let (sender, receiver) = channel::<io::Msg>();
		let reporter = Reporter::new(receiver);
		s.spawn(move || {
			while let Ok(msg) = reporter.next() {
				msg.display();
			}
		});

		if !unit_cell.is_ortho() {
			panic!("Only orthorhombic systems are currently supported.")
		}

		let system = System::new(unit_cell, atoms, &settings);

		let system = Arc::new(system);

		let msgr = sender.clone();
		let arc_settings = settings.clone();
		let sys = system.clone();

		let lattice = s
			.spawn(move || {
				msgr.send(
					Msg::new(String::from("Initializing system"))
						.make_heading(),
				)
				.unwrap();
				msgr.send(
					format!(
						"Constructing lattice with cubelet size of {}Å",
						arc_settings.cubelet_size()
					)
					.into(),
				)
				.unwrap();

				let lattice = Lattice::new(&sys, arc_settings, msgr.clone());

				msgr.send("Finished constructing lattice.".into()).unwrap();

				lattice
			})
			.join()
			.unwrap();

		let msgr = sender.clone();
		let sys = system.clone();
		//let arc_settings = settings.clone();
		//let arc_lattice = Arc::new(lattice);

		let _: Result<(), std::io::Error> = s.spawn(move || {
			msgr.send(
				Msg::new(String::from("Initialization complete"))
					.make_heading(),
			)
			.unwrap();

			let volume = sys.volume();
			msgr.send(
				format!("Volume of system: {:.3} Å³", volume.as_f64()).into(),
			)
			.unwrap();

			msgr.send(
				format!(
					"Molecular weight of unit cell: {:.3} g/mol",
					sys.molar_mass().as_f64()
				)
				.into(),
			)
			.unwrap();

			msgr.send(
				format!("Density: {:.3} g/cm³", sys.density().as_f64()).into(),
			)
			.unwrap();

			
			let mut grid = lattice.geom_accessible_grid();
			msgr.send(
				Msg::new(
					"Calculating limiting pore diameter and maximum pore size".to_owned(),
				)
				.make_heading(),
			)
			.unwrap();

			grid.percolate();

			let msg: io::Msg = match grid.span() {
				| None => String::from("The system is not percolated in any direction.").into(),
				| Some(n_dimensions) => format!("The system is percolated in {} dimensions.", n_dimensions.to_string()).into()
			};

			let (limiting_diameter, max_pore_diameter) =
				grid.limiting_diameter(&lattice);

			msgr.send(format!("Pore limiting diameter: {limiting_diameter:.2} Å").into()).unwrap();
			msgr.send(format!("Maximum pore diameter: {max_pore_diameter:.2} Å").into()).unwrap();

			msgr.send(msg).unwrap();

			msgr.send(Msg::new("Limiting pore diameter and maximum pore size calculations complete".to_owned()).make_heading()).unwrap();

			msgr.send(
				Msg::new("Starting surface area calculations".to_owned())
					.make_heading(),
			)
			.unwrap();

			let surface_area = sys.surface_area(&lattice, &settings);
			msgr.send(format!("Total surface area: {:.2} Å²", surface_area.as_f64()).into()).unwrap();

			let sa_per_vol = sys.surface_area_per_volume(surface_area);
			msgr.send(format!("Total surface area per volume: {:.2} m²/cm³", sa_per_vol.as_f64()).into()).unwrap();

			let sa_per_mass = sys.surface_area_per_mass(sa_per_vol);

			msgr.send(format!("Total surface area per mass: {:.2} m²/g", sa_per_mass.as_f64()).into()).unwrap();

			msgr.send(Msg::new("Surface area calculations complete".to_owned()).make_heading()).unwrap();


			msgr.send(Msg::new("Starting pore size distribution calculations".to_owned()).make_heading()).unwrap();
			let (free_volume, ffv) = grid.pore_size_distribution(&lattice, &system, &settings)?;
			msgr.send(Msg::new("Pore size distribution calculations complete".to_owned()).make_heading()).unwrap();

			msgr.send(Msg::new("Starting pore volume calculations".to_owned()).make_heading()).unwrap();
			let he_volume = lattice.pore_volume_he(&system, settings.temperature());
			let geom_volume = lattice.pore_volume_geom(&system);
			let sys_mass = system.mass().as_f64();

			let he_vol_per_mass = he_volume*1.0E-24 / sys_mass;
			let geom_vol_per_mass = geom_volume*1.0E-24 / sys_mass;
			let free_volume_per_mass = free_volume*1.0E-24 / sys_mass;

			msgr.send(Msg::new(format!("Total helium volume: {he_volume:.3} Å³"))).unwrap();
			msgr.send(Msg::new(format!("Total helium volume per mass: {he_vol_per_mass:.3} cm³/g"))).unwrap();
			msgr.send(Msg::new(format!("Total geometric volume: {geom_volume:.3} Å³"))).unwrap();
			msgr.send(Msg::new(format!("Total geometric volume per mass: {geom_vol_per_mass:.3}"))).unwrap();
			msgr.send(Msg::new(format!("Total probe-occupiable volume: {free_volume:.3} Å³"))).unwrap();
			msgr.send(Msg::new(format!("Total probe-occupiable volume per mass: {free_volume_per_mass:.3} cm³/g"))).unwrap();
			msgr.send(Msg::new(format!("Total fraction of free volume: {ffv:.3}"))).unwrap();

			Ok(())
		}).join().unwrap();
	});

	Ok(())
}
