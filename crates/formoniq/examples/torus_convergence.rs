use formoniq::torus_convergence::run_torus_convergence;

fn main() -> Result<(), Box<dyn std::error::Error>> {
  tracing_subscriber::fmt::init();
  let records = run_torus_convergence("out/examples/torus_convergence")?;

  println!(
    "| {:>10} | {:>10} | {:>10} | {:>8} | {:>10} | {:>8} | {:>8} |",
    "resolution", "h", "L2 error", "L2 rate", "Hd error", "Hd rate", "wall s",
  );
  println!(
    "| {:-<10} | {:-<10} | {:-<10} | {:-<8} | {:-<10} | {:-<8} | {:-<8} |",
    "", "", "", "", "", "", "",
  );
  for record in records {
    println!(
      "| {:>10} | {:>10.3e} | {:>10.3e} | {:>8} | {:>10.3e} | {:>8} | {:>8.3} |",
      record.resolution,
      record.h,
      record.l2_error,
      format_rate(record.l2_rate),
      record.hd_error,
      format_rate(record.hd_rate),
      record.wall_seconds,
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
