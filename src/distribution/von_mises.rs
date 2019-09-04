use std::f64;

use distribution::Univariate;
use rgsl::{bessel, Value};
use statistics::{Max, Min};
use {Result, StatsError};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct VonMises {
    location: f64,
    concentration: f64,
}

impl VonMises {
    /// Constructs a new von Mises distribution with location (mu) `location`,
    /// and concentration (kappa) `concentration`.
    ///
    /// # Errors
    ///
    /// Returns an error if `location` or `concentration` are `NaN`, or if
    /// `concentration <= 0.0`.
    ///
    /// # Examples
    ///
    /// ```
    /// use statrs::distribution::VonMises;
    ///
    /// let mut result = VonMises::new(0.0, 1.0);
    /// assert!(result.is_ok());
    ///
    /// result = VonMises::new(0.0, 0.0);
    /// assert!(result.is_err());
    /// ```
    pub fn new(location: f64, concentration: f64) -> Result<VonMises> {
        if location.is_nan() || concentration.is_nan() || concentration <= 0.0 {
            Err(StatsError::BadParams)
        } else {
            Ok(VonMises {
                location,
                concentration,
            })
        }
    }
}

impl Min<f64> for VonMises {
    /// Returns the minimum value in the domain of the
    /// von Mises distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// -π
    /// ```
    fn min(&self) -> f64 {
        -f64::consts::PI
    }
}

impl Max<f64> for VonMises {
    /// Returns the maximum value in the domain of the
    /// von Mises distribution representable by a double precision float
    ///
    /// # Formula
    ///
    /// ```ignore
    /// π
    /// ```
    fn max(&self) -> f64 {
        f64::consts::PI
    }
}

impl Univariate<f64, f64> for VonMises {
    fn cdf(&self, x: f64) -> f64 {
        let d = x - self.location;
        let mut results: [f64; 100] = [0.0; 100];
        match bessel::In_array(1, 100, self.concentration, &mut results) {
            Value::Success => {}
            other => panic!(other),
        };
        let sum: f64 = results
            .into_iter()
            .enumerate()
            .map(|(j, i_j)| i_j * ((j + 1) as f64 * d).sin() / (j + 1) as f64)
            .sum();
        0.5 + (d + (2.0 * sum / bessel::I0(self.concentration))) / (2.0 * f64::consts::PI)
    }
}

// impl Continuous<f64, f64> for VonMises {
// 	fn pdf(&self, x: f64) -> f64 {
// 		let d = (x - self.location) / self.scale;
// 		(self.concentration * d.cos()).exp() / (2.0 * f64::consts::PI * I0(self.concentration))
// 	}
// }

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cdf() {
        let vm = VonMises::new(0.0, 1.0).unwrap();
        assert_almost_eq!(vm.cdf(-3.0), 0.006569944565732455, 0.000001);
        assert_almost_eq!(vm.cdf(-2.0), 0.06575904411000724, 0.000001);
        assert_almost_eq!(vm.cdf(-1.0), 0.20564469256531126, 0.000001);
        assert_almost_eq!(vm.cdf(0.0), 0.5, 0.000001);
        assert_almost_eq!(vm.cdf(1.0), 0.7943553074346887, 0.000001);
        assert_almost_eq!(vm.cdf(2.0), 0.9342409558899928, 0.000001);
        assert_almost_eq!(vm.cdf(3.0), 0.9934300554342675, 0.000001);

        let vm = VonMises::new(0.0, 4.0).unwrap();
        assert_almost_eq!(vm.cdf(-3.0), 3.701352583693543e-05, 0.000001);
        assert_almost_eq!(vm.cdf(-2.0), 0.0008703772263199128, 0.000001);
        assert_almost_eq!(vm.cdf(-1.0), 0.033225820901453484, 0.000001);
        assert_almost_eq!(vm.cdf(0.0), 0.5, 0.000001);
        assert_almost_eq!(vm.cdf(1.0), 0.9667741790985465, 0.000001);
        assert_almost_eq!(vm.cdf(2.0), 0.9991296227736801, 0.000001);
        assert_almost_eq!(vm.cdf(3.0), 0.999962986474163, 0.000001);

        let vm = VonMises::new(1.0, 1.0).unwrap();
        assert_almost_eq!(vm.cdf(-2.0), 0.006569944565732455, 0.000001);
        assert_almost_eq!(vm.cdf(-1.0), 0.06575904411000724, 0.000001);
        assert_almost_eq!(vm.cdf(0.0), 0.20564469256531126, 0.000001);
        assert_almost_eq!(vm.cdf(1.0), 0.5, 0.000001);
        assert_almost_eq!(vm.cdf(2.0), 0.7943553074346887, 0.000001);
        assert_almost_eq!(vm.cdf(3.0), 0.9342409558899928, 0.000001);
    }
}
