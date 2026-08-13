fn main() -> Result<(), Box<dyn std::error::Error>> {
  use formoniq::sparse_inverse_hodge_validation::{
    run_sparse_inverse_hodge_validation, SparseInverseHodgeValidationConfig,
  };
  tracing_subscriber::fmt::init();
  let mut config = SparseInverseHodgeValidationConfig::default();
  if let Ok(value) = std::env::var("FORMONIQ_MAX_REFINEMENT") {
    config.max_refinement = value.parse()?;
  }
  run_sparse_inverse_hodge_validation(&config)?;
  Ok(())
}
