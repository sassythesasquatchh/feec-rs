#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ThetaMethod(f64);

impl ThetaMethod {
  pub const BACKWARD_EULER: Self = Self(1.0);
  pub const CRANK_NICOLSON: Self = Self(0.5);

  pub fn new(theta: f64) -> Self {
    assert!(
      (0.5..=1.0).contains(&theta),
      "theta must satisfy 0.5 <= theta <= 1.0, got {theta}."
    );
    Self(theta)
  }

  pub fn theta(self) -> f64 {
    self.0
  }
}

impl Default for ThetaMethod {
  fn default() -> Self {
    Self::BACKWARD_EULER
  }
}

pub fn validate_time_grid(times: &[f64]) {
  assert!(
    times.len() >= 2,
    "time grid must contain at least two entries, got {}.",
    times.len()
  );

  for window in times.windows(2) {
    let [t0, t1] = window else { unreachable!() };
    assert!(
      t1 > t0,
      "time grid must be strictly increasing, but found {} followed by {}.",
      t0,
      t1
    );
  }
}

#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn theta_method_accepts_interval_endpoints() {
    assert_eq!(ThetaMethod::new(0.5), ThetaMethod::CRANK_NICOLSON);
    assert_eq!(ThetaMethod::new(1.0), ThetaMethod::BACKWARD_EULER);
  }

  #[test]
  #[should_panic(expected = "0.5 <= theta <= 1.0")]
  fn theta_method_rejects_values_below_half() {
    let _ = ThetaMethod::new(0.49);
  }

  #[test]
  #[should_panic(expected = "0.5 <= theta <= 1.0")]
  fn theta_method_rejects_values_above_one() {
    let _ = ThetaMethod::new(1.01);
  }

  #[test]
  fn validate_time_grid_accepts_strictly_increasing_grid() {
    validate_time_grid(&[0.0, 0.1, 0.4]);
  }

  #[test]
  #[should_panic(expected = "at least two entries")]
  fn validate_time_grid_requires_at_least_two_points() {
    validate_time_grid(&[0.0]);
  }

  #[test]
  #[should_panic(expected = "strictly increasing")]
  fn validate_time_grid_rejects_repeated_entries() {
    validate_time_grid(&[0.0, 0.0, 0.1]);
  }
}
