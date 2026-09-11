use std::{
	f64::consts::PI,
	ops::{
		Add, AddAssign, Div, Index, IndexMut, Mul, MulAssign, Sub, SubAssign,
	},
	simd::{Simd, SimdElement, StdFloat, num::SimdFloat},
	slice::{Chunks, ChunksMut, Iter, IterMut},
};

use num::Float;

use crate::atoms::{Radius, Scalar};

#[derive(Copy, Clone)]
pub struct Vec3<T> {
	data: [T; 3],
}

impl From<[usize; 3]> for Vec3<f64> {
	#[inline]
	fn from(value: [usize; 3]) -> Self {
		let value = [value[0] as f64, value[1] as f64, value[2] as f64];
		unsafe { std::mem::transmute::<[f64; 3], Vec3<f64>>(value) }
	}
}

impl Vec3<&mut f64> {
	#[inline]
	pub fn from_ref_mut(value: [&mut f64; 3]) -> Self {
		unsafe { std::mem::transmute::<[&mut f64; 3], Vec3<&mut f64>>(value) }
	}
}

impl<F> From<[F; 3]> for Vec3<F>
where
	F: Float + Sized,
{
	#[inline]
	fn from(value: [F; 3]) -> Self {
		Self {
			data: value
		}
	}
}

impl Into<[f64; 3]> for Vec3<f64> {
	#[inline]
	fn into(self) -> [f64; 3] {
		self.data
	}
}

impl<T> Vec3<T>
where
	T: SimdElement + num::Float,
	Simd<T, 3>: Mul<Output = Simd<T, 3>> + SimdFloat<Scalar = T>,
{
	#[inline]
	pub fn splat(value: T) -> Self {
		Self {
			data: [value, value, value],
		}
	}

	#[inline]
	pub fn iter(&self) -> Iter<'_, T> {
		self.data.iter()
	}

	#[inline]
	pub fn iter_mut(&mut self) -> IterMut<'_, T> {
		self.data.iter_mut()
	}

	#[inline]
	pub fn norm(&self) -> T {
		let v = self.as_simd();
		(v * v).reduce_sum().sqrt()
	}

	#[inline]
	fn as_simd(&self) -> Simd<T, 3> {
		Simd::from(self.as_arr())
	}

	#[inline]
	fn as_arr(&self) -> [T; 3] {
		self.data
	}

	#[inline]
	pub fn x(&self) -> T {
		self.data[0]
	}

	#[inline]
	pub fn y(&self) -> T {
		self.data[1]
	}

	#[inline]
	pub fn z(&self) -> T {
		self.data[2]
	}
}

impl Vec3<f64> {
	#[inline]
	pub fn zeros() -> Self {
		Vec3::splat(0.0)
	}

	#[inline]
	pub fn round(&mut self) -> Self {
		self.as_simd().round().into()
	}

	#[inline]
	pub fn as_array(self) -> [f64; 3] {
		unsafe { std::mem::transmute::<Vec3<f64>, [f64; 3]>(self) }
	}

	#[inline]
	pub fn distance(&self, other: Self) -> f64 {
		(other - *self).norm()
	}

	#[inline]
	pub fn cross(&self, other: Self) -> Self {
		Vec3::from([
			self.y() * other.z() - self.z() * other.y(),
			self.x() * other.z() - self.z() * other.x(),
			self.x() * other.y() - self.y() * other.x(),
		])
	}

	pub fn random_point_on_sphere(r: Radius) -> Self {
		let u = rand::random_range(0.0..=1.0);
		let v = rand::random_range(0.0..=1.0);
		//let phi = 2. * PI * u;
		//let theta = f64::acos(2. * v - 1.0);
		//let x = r * phi.sin() * theta.cos();
		//let y = r * phi.sin() * theta.sin();
		//let z = r * phi.cos();

		let phi = PI - 2. * PI * u;
		let theta = f64::acos(1. - 2. * v);
		let x = r * theta.sin() * phi.cos();
		let y = r * theta.sin() * phi.sin();
		let z = r * theta.cos();

		Self::from([x.as_f64(), y.as_f64(), z.as_f64()])
	}
}

impl From<Simd<f64, 3>> for Vec3<f64> {
	fn from(value: Simd<f64, 3>) -> Self {
		Self::from(value.to_array())
	}
}

impl Mul<f64> for Vec3<f64> {
	type Output = Self;

	fn mul(self, rhs: f64) -> Self::Output {
		Self::from(self.as_simd() * Simd::<f64, 3>::splat(rhs))
	}
}

impl Mul<Vec3<f64>> for f64 {
	type Output = Vec3<f64>;

	fn mul(self, rhs: Vec3<f64>) -> Self::Output {
		Vec3::from(Simd::<f64, 3>::splat(self) * rhs.as_simd())
	}
}

impl Mul<Vec3<f64>> for Vec3<f64> {
	type Output = Self;

	fn mul(self, rhs: Vec3<f64>) -> Self::Output {
		Self::from(self.as_simd() * rhs.as_simd())
	}
}

impl MulAssign<Vec3<f64>> for Vec3<f64> {
	fn mul_assign(&mut self, rhs: Vec3<f64>) {
		*self = *self * rhs;
	}
}

impl MulAssign<f64> for Vec3<f64> {
	fn mul_assign(&mut self, rhs: f64) {
		*self = *self * rhs;
	}
}

impl Add<Vec3<f64>> for Vec3<f64> {
	type Output = Self;

	fn add(self, rhs: Vec3<f64>) -> Self::Output {
		Self::from(self.as_simd() + rhs.as_simd())
	}
}

impl AddAssign<Vec3<f64>> for Vec3<f64> {
	fn add_assign(&mut self, rhs: Vec3<f64>) {
		*self = *self + rhs;
	}
}

impl Sub<Vec3<f64>> for Vec3<f64> {
	type Output = Self;

	fn sub(self, rhs: Vec3<f64>) -> Self::Output {
		Self::from(self.as_simd() - rhs.as_simd())
	}
}

impl SubAssign<Vec3<f64>> for Vec3<f64> {
	fn sub_assign(&mut self, rhs: Vec3<f64>) {
		*self = *self - rhs;
	}
}

impl Div<Vec3<f64>> for Vec3<f64> {
	type Output = Self;

	fn div(self, rhs: Vec3<f64>) -> Self::Output {
		Self::from(self.as_simd() / rhs.as_simd())
	}
}

impl Div<f64> for Vec3<f64> {
	type Output = Self;

	fn div(self, rhs: f64) -> Self::Output {
		Self::from(self.as_simd() / Self::splat(rhs).as_simd())
	}
}

pub trait Dot<Rhs = Self> {
	type Output;

	fn dot(&self, rhs: Rhs) -> Self::Output;
}

impl<T> Dot for Vec3<T>
where
	T: SimdElement + num::Float,
	Simd<T, 3>: Mul<Output = Simd<T, 3>> + SimdFloat<Scalar = T>,
{
	type Output = T;

	#[inline]
	fn dot(&self, rhs: Self) -> Self::Output {
		(self.as_simd() * rhs.as_simd()).reduce_sum()
	}
}

#[derive(PartialEq, Eq, Debug, Copy, Clone)]
pub struct Matrix<T, const R: usize, const C: usize>
where
	[T; R * C]:,
{
	data: [T; R * C],
}

impl<T, const R: usize, const C: usize> Matrix<T, R, C>
where
	T: Copy + Clone,
	[T; R * C]:,
{
	#[inline]
	pub fn from_arr(arr: [T; R * C]) -> Self {
		Self {
			data: arr
		}
	}

	#[inline]
	pub fn iter(&self) -> Iter<'_, T> {
		self.data.iter()
	}

	#[inline]
	pub fn iter_mut(&mut self) -> IterMut<'_, T> {
		self.data.iter_mut()
	}

	#[inline]
	pub fn cols(&self) -> Chunks<'_, T> {
		self.data.chunks(R)
	}

	#[inline]
	pub fn cols_mut(&mut self) -> ChunksMut<'_, T> {
		self.data.chunks_mut(R)
	}

	pub fn transpose(&self) -> Matrix<T, C, R>
	where
		[(); C * R]:,
	{
		let mut data = Vec::new();
		for i in 0..R {
			for j in 0..R {
				data.push(self.data[i + j * R])
			}
		}

		Matrix::<T, C, R>::from_arr(*data.as_array().unwrap())
	}

	#[inline]
	pub fn shape(&self) -> (usize, usize) {
		(R, C)
	}
}

#[macro_export]
macro_rules! _count {
    () => (0usize);
    ( $x:tt $($xs:tt)* ) => (1usize + _count!($($xs)*));
}

#[macro_export]
macro_rules! _count_semicolon {
    ($($($n:expr),+);+) => { _count_semicolon_bracketed!($([$($n),+]);+) }
}

#[macro_export]
macro_rules! _count_semicolon_bracketed {
     ($($n:tt);+) => {_count!( $($n)+ )};
}

#[macro_export]
macro_rules! matrix {
    ($($($n:expr) +);+) => {{
        const LEN: usize = _count!($($($n)+)+);
        const ROWS: usize = _count_semicolon!($($($n),+);+);
        const COLS: usize = LEN / ROWS;

        Matrix::<_,ROWS,COLS>::from_arr([$($($n),+),+])
    }};
}

impl<const R: usize, const C: usize> Mul<Matrix<f64, R, C>> for f64
where
	[(); R * C]:,
{
	type Output
		= Matrix<f64, R, C>
	where
		[(); R * C]:;

	fn mul(self, rhs: Matrix<f64, R, C>) -> Self::Output {
		let data = *rhs
			.cols()
			.flat_map(|col| {
				(Simd::<f64, R>::from_slice(col) * Simd::<f64, R>::splat(self))
					.to_array()
			})
			.collect::<Vec<f64>>()
			.as_array()
			.unwrap();

		Matrix::from_arr(data)
	}
}

impl<T> Mul<Vec3<T>> for Matrix<T, 3, 3>
where
	T: SimdElement + num::Float,
	Simd<T, 3>: Mul<Output = Simd<T, 3>> + SimdFloat<Scalar = T>,
{
	type Output = Vec3<T>;

	fn mul(self, rhs: Vec3<T>) -> Self::Output {
		let vec3 = rhs.as_simd();

		let data: [T; 3] = *self
			.transpose()
			.cols()
			.map(|row| (Simd::<T, 3>::from_slice(row) * vec3).reduce_sum())
			.collect::<Vec<T>>()
			.as_array()
			.unwrap();

		Vec3::from(data)
	}
}

impl<T, const R1: usize, const C1: usize, const C2: usize>
	Mul<Matrix<T, C1, C2>> for Matrix<T, R1, C1>
where
	T: SimdElement + num::Float,
	[T; R1 * C1]:,
	[T; C1 * C2]:,
	[T; R1 * C2]:,
	[T; C1 * R1]:,
	Simd<T, 3>: Mul<Output = Simd<T, 3>> + SimdFloat<Scalar = T>,
{
	type Output
		= Matrix<T, R1, C2>
	where
		[T; R1 * C2]:;

	fn mul(self, rhs: Matrix<T, C1, C2>) -> Self::Output {
		//let data = self.transpose().cols().map(|row| rhs.cols().map(|col| ));
		let data: [T; R1 * C2] = *rhs
			.cols()
			.flat_map(|col| {
				*self
					.transpose()
					.cols()
					.map(|row| {
						(Simd::<T, 3>::from_slice(row)
							* Simd::<T, 3>::from_slice(col))
						.reduce_sum()
					})
					.collect::<Vec<T>>()
					.as_array::<3>()
					.unwrap()
			})
			.collect::<Vec<T>>()
			.as_array()
			.unwrap();

		Matrix::from_arr(data)
	}
}

impl<T, const R: usize, const C: usize> Index<(usize, usize)>
	for Matrix<T, R, C>
where
	[(); R * C]:,
{
	type Output = T;

	#[inline]
	fn index(&self, index: (usize, usize)) -> &Self::Output {
		let (i, j) = index;
		&self.data[i + j * R]
	}
}

impl<T, const R: usize, const C: usize> IndexMut<(usize, usize)>
	for Matrix<T, R, C>
where
	[(); R * C]:,
{
	#[inline]
	fn index_mut(&mut self, index: (usize, usize)) -> &mut Self::Output {
		let (i, j) = index;
		&mut self.data[i + j * R]
	}
}

impl Matrix<f64, 3, 3> {
	pub fn invert(&self) -> Self {
		let [a, b, c, d, e, f, g, h, i] = self.data;

		let res = matrix![
			e*i-f*h f*g-d*i d*h-e*g;
			c*h-b*i a*i-c*g b*g-a*h;
			b*f-c*e c*d-a*f a*e-b*d
		]
		.transpose();

		(1. / self.det()) * res
	}

	pub fn det(&self) -> f64 {
		let [a, b, c, d, e, f, g, h, i] = self.data;
		(Simd::<f64, 3>::from_slice(&[a, b, c])
			* Simd::from_slice(&[e, f, d])
			* Simd::from_slice(&[i, g, h]))
		.reduce_sum()
			- (Simd::<f64, 3>::from_slice(&[c, b, a])
				* Simd::from_slice(&[e, d, f])
				* Simd::from_slice(&[g, i, h]))
			.reduce_sum()
	}
}

#[cfg(test)]
mod tests {
	use super::*;

	#[test]
	fn test_matrix_macro() {
		let id = matrix![
			1. 0. 0.;
			0. 1. 0.;
			0. 0. 1.
		];

		let m = matrix![
			1. 2. 3.;
			4. 5. 6.;
			7. 8. 9.
		];

		let r = matrix![
			2. 4. 6.;
			8. 10. 12.;
			14. 16. 18.
		];

		assert_eq!(m, m * id);
		assert_eq!(5., m[(1, 1)]);
		assert_eq!(r, 2. * m);
		assert_eq!(1.0, id.det());
		assert_eq!(id, id.invert());
	}
}
