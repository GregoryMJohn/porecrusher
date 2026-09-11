use crate::{
	atoms::Volume,
	linalg::{Matrix, Vec3},
};

pub struct UnitCell {
	a:              f64,
	b:              f64,
	c:              f64,
	alpha:          f64,
	beta:           f64,
	gamma:          f64,
	origin:         Vec3<f64>,
	slant_matrix:   Option<Matrix<f64, 3, 3>>,
	unslant_matrix: Option<Matrix<f64, 3, 3>>,
	is_ortho:       bool,
}

impl UnitCell {
	pub fn new(dims: [f64; 3], angles: [f64; 3]) -> Self {
		let is_ortho = ((angles[0] - 90.0).abs()
			+ (angles[1] - 90.0).abs()
			+ (angles[2] - 90.0).abs())
			<= 0.001;

		let [a, b, c] = dims;
		let [mut alpha, mut beta, mut gamma] = angles;
		alpha = alpha.to_radians();
		beta = beta.to_radians();
		gamma = gamma.to_radians();

		let (slant_matrix, unslant_matrix) = if is_ortho {
			(None, None)
		} else {
			let elem_1_1 = (1. - gamma.cos().powf(2.)).sqrt();
			let tmp = alpha.cos() - gamma.cos() * beta.cos();
			let elem_1_2 = tmp / elem_1_1;
			let slant_matrix = matrix![
				1. gamma.cos() beta.cos();
				0. elem_1_1 elem_1_2;
				0. 0. beta.cos().powf(2.) - elem_1_2.powf(2.)
			];

			let unslant_matrix = slant_matrix.invert();

			(Some(slant_matrix), Some(unslant_matrix))
		};
		let origin = Vec3::zeros();

		Self {
			a,
			b,
			c,
			alpha,
			beta,
			gamma,
			origin,
			slant_matrix,
			unslant_matrix,
			is_ortho,
		}
	}

	#[inline]
	pub fn a(&self) -> f64 {
		self.a
	}

	#[inline]
	pub fn b(&self) -> f64 {
		self.b
	}

	#[inline]
	pub fn c(&self) -> f64 {
		self.c
	}

	#[inline]
	fn origin(&self) -> Vec3<f64> {
		self.origin
	}

	#[inline]
	pub fn is_ortho(&self) -> bool {
		self.is_ortho
	}

	pub fn volume(&self) -> Volume {
		let vol = Volume::from(self.a * self.b * self.c);
		if self.is_ortho {
			vol
		} else {
			vol * f64::sqrt(
				1.0 + (2.0
					* f64::cos(self.alpha)
					* f64::cos(self.beta)
					* f64::cos(self.gamma))
					- f64::cos(self.alpha).powf(2.0)
					- f64::cos(self.beta).powf(2.0)
					- f64::cos(self.gamma).powf(2.0),
			)
		}
	}

	#[inline]
	pub fn transform_slant(&self, vector: &mut Vec3<f64>) {
		if let Some(matrix) = self.slant_matrix {
			*vector = matrix * *vector;
		}
	}

	#[inline]
	pub fn transform_unslant(&self, vector: &mut Vec3<f64>) {
		if let Some(matrix) = self.unslant_matrix {
			*vector = matrix * *vector;
		}
	}

	pub fn map_to_cell(&self, vector: &mut Vec3<f64>) {
		let mut dummy_vec = *vector - self.origin();
		self.transform_slant(&mut dummy_vec);
		let edge_lengths = self.edge_lengths();
		dummy_vec *= Vec3::splat(1.0) / edge_lengths;

		for elem in dummy_vec.iter_mut() {
			*elem %= 1.0;
			if *elem < 0.0 {
				*elem += 1.0;
			}
		}

		dummy_vec *= edge_lengths;
		self.transform_unslant(&mut dummy_vec);

		*vector = dummy_vec + self.origin();
	}

	fn edge_lengths(&self) -> Vec3<f64> {
		Vec3::from([self.a, self.b, self.c])
	}
}

#[cfg(test)]
mod tests {
	use super::UnitCell;

	#[test]
	fn test_volume() {
		let unit_cell = UnitCell::new([30.0, 30.0, 30.0], [90.0, 90.0, 90.0]);

		assert_eq!(30.0 * 30.0 * 30.0, unit_cell.volume().into())
	}

	#[test]
	fn test_ortho() {
		let unit_cell = UnitCell::new([30.0, 30.0, 30.0], [90.0, 90.0, 90.0]);

		assert!(unit_cell.is_ortho())
	}
}
