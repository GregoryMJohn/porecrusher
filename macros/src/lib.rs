extern crate proc_macro;
use proc_macro::TokenStream;
use quote::quote;
use syn::DeriveInput;

#[proc_macro_derive(Scalar)]
pub fn scalar_derive(input: TokenStream) -> TokenStream {
	let ast: DeriveInput = syn::parse(input).unwrap();
	let name = &ast.ident;

	TokenStream::from(quote! {
		impl Scalar for #name {
			#[inline]
			fn as_f64(self) -> f64 {
				self.0
			}
		}

		impl Into<f64> for #name {
			#[inline]
			fn into(self) -> f64 {
				self.0
			}
		}

		impl From<f64> for #name {
			#[inline]
			fn from(value: f64) -> Self {
				Self(value)
			}
		}

		impl Add for #name {
			type Output = Self;

			#[inline]
			fn add(self, rhs: Self) -> Self {
				Self(self.0 + rhs.0)
			}
		}

		impl Sub for #name {
			type Output = Self;

			#[inline]
			fn sub(self, rhs: Self) -> Self {
				Self(self.0 - rhs.0)
			}
		}

		impl Mul for #name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: Self) -> Self {
				Self(self.0 * rhs.0)
			}
		}

		impl Mul<f64> for #name {
			type Output = Self;

			#[inline]
			fn mul(self, rhs: f64) -> Self {
				Self(self.0 * rhs)
			}
		}

		impl Div for #name {
			type Output = Self;

			#[inline]
			fn div(self, rhs: Self) -> Self {
				Self(self.0 / rhs.0)
			}
		}

		impl Div<f64> for #name {
			type Output = Self;

			#[inline]
			fn div(self, rhs: f64) -> Self {
				Self(self.0 / rhs)
			}
		}

		impl Display for #name {
			fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> fmt::Result {
				self.0.fmt(f)
			}
		}
	})
}
