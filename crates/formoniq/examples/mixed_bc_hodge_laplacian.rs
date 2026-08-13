use formoniq::mixed_bc_hodge_laplacian_convergence::{
  run_mixed_bc_hodge_laplacian_convergence, DEFAULT_RESOLUTIONS,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
  tracing_subscriber::fmt::init();

  println!(
    "| {:>10} | {:>10} | {:>8} | {:>10} | {:>8} | {:>10} | {:>8} | {:>8} |",
    "resolution", "h", "u dofs", "L2 error", "L2 rate", "Hd error", "Hd rate", "wall s",
  );
  println!(
    "| {:-<10} | {:-<10} | {:-<8} | {:-<10} | {:-<8} | {:-<10} | {:-<8} | {:-<8} |",
    "", "", "", "", "", "", "", "",
  );

  let rows = run_mixed_bc_hodge_laplacian_convergence(
    "out/examples/general_hodge_laplacian",
    DEFAULT_RESOLUTIONS,
  )?;
  for row in rows {
    println!(
      "| {:>10} | {:>10.3e} | {:>8} | {:>10.3e} | {:>8} | {:>10.3e} | {:>8} | {:>8.3} |",
      row.resolution,
      row.h,
      row.u_dofs,
      row.l2_error,
      format_rate(row.l2_rate),
      row.hd_error,
      format_rate(row.hd_rate),
      row.wall_seconds,
    );
  }

  Ok(())
}

fn format_rate(rate: f64) -> String {
  if rate.is_finite() {
    format!("{rate:.2}")
  } else {
    "-".to_string()
  }
}
