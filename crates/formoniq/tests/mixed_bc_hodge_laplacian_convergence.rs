#![cfg(feature = "external-solver-tests")]

use formoniq::mixed_bc_hodge_laplacian_convergence::{
  convergence_csv_output_path, run_mixed_bc_hodge_laplacian_convergence,
};

use std::{
  fs,
  path::PathBuf,
  process,
  time::{SystemTime, UNIX_EPOCH},
};

fn unique_output_dir() -> PathBuf {
  let unique = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .expect("system clock before unix epoch")
    .as_nanos();
  std::env::temp_dir().join(format!(
    "formoniq_mixed_bc_hodge_laplacian_convergence_{}_{}",
    process::id(),
    unique
  ))
}

#[test]
fn mixed_bc_hodge_laplacian_convergence_writes_l2_hd_table(
) -> Result<(), Box<dyn std::error::Error>> {
  let output_dir = unique_output_dir();
  let results = run_mixed_bc_hodge_laplacian_convergence(&output_dir, &[1])?;
  assert_eq!(results.len(), 1);

  let record = &results[0];
  assert!(record.h.is_finite() && record.h > 0.0);
  assert!(record.l2_error.is_finite() && record.l2_error >= 0.0);
  assert!(record.hd_error.is_finite() && record.hd_error >= 0.0);
  assert!(record.hd_error >= record.l2_error);
  assert!(record.wall_seconds.is_finite() && record.wall_seconds >= 0.0);

  let csv_path = convergence_csv_output_path(&output_dir);
  assert!(csv_path.exists(), "missing convergence CSV: {csv_path:?}");
  let csv = fs::read_to_string(&csv_path)?;
  assert!(csv.contains("resolution,h,u_dofs,l2_error,l2_rate,hd_error,hd_rate,wall_seconds"));
  assert_eq!(csv.lines().count(), 2);

  let _ = fs::remove_dir_all(&output_dir);
  Ok(())
}
