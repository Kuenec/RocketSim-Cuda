use std::time::Instant;

use rocketsim::{Arena, ArenaConfig, CarBodyConfig, CarControls, GameMode, MutatorConfig, Team};

use crate::ffi;
use crate::scenario::{Backend, SplitMix, mode_code, to_ffi_controls, to_ffi_mutators};

pub enum Target {
    Upstream,
    Ours(Backend),
}

impl Target {
    pub fn name(&self) -> &'static str {
        match self {
            Target::Upstream => "upstream-rust",
            Target::Ours(b) => b.name(),
        }
    }
}

const TICKS_PER_ACTION: u32 = 8;
const CONTROL_POOL: usize = 32;

pub struct Stats {
    pub median: f64,
    pub min: f64,
    pub max: f64,
    pub samples: Vec<f64>,
}

fn stats(mut samples: Vec<f64>) -> Stats {
    let sorted = {
        let mut s = samples.clone();
        s.sort_by(|a, b| a.total_cmp(b));
        s
    };
    let median = if sorted.len() % 2 == 1 {
        sorted[sorted.len() / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    };
    samples.shrink_to_fit();
    Stats {
        median,
        min: sorted[0],
        max: sorted[sorted.len() - 1],
        samples,
    }
}

fn control_pool(arenas: usize, cars: usize) -> Vec<Vec<CarControls>> {
    let mut rng = SplitMix::new(0xbe7c_4a11);
    (0..CONTROL_POOL)
        .map(|_| {
            (0..arenas * cars)
                .map(|_| CarControls {
                    throttle: rng.range(-1.0, 1.0),
                    steer: rng.range(-1.0, 1.0),
                    pitch: rng.range(-1.0, 1.0),
                    yaw: rng.range(-1.0, 1.0),
                    roll: rng.range(-1.0, 1.0),
                    jump: rng.below(6) == 0,
                    boost: rng.below(3) == 0,
                    handbrake: rng.below(8) == 0,
                })
                .collect()
        })
        .collect()
}

pub fn run(
    meshes: &ffi::MeshSet,
    target: &Target,
    arenas: usize,
    reps: usize,
    ticks: u32,
    warmup_ticks: u32,
) -> Result<Stats, String> {
    let mode = GameMode::Soccar;
    let mutators = MutatorConfig::new(mode);
    let teams = [Team::Blue, Team::Orange];
    let pool = control_pool(arenas, teams.len());
    let chunks = ticks / TICKS_PER_ACTION;
    let warmup_chunks = warmup_ticks / TICKS_PER_ACTION;
    let mut samples = Vec::new();
    match target {
        Target::Upstream => {
            let mut sims: Vec<Arena> = (0..arenas as u64)
                .map(|seed| {
                    let config = ArenaConfig::new(mode)
                        .with_mutators(mutators)
                        .with_rng_seed(seed);
                    let mut arena = Arena::new_with_config(config);
                    for team in teams {
                        arena.add_car(team, CarBodyConfig::OCTANE);
                    }
                    arena.reset_to_random_kickoff(Some(seed));
                    arena
                })
                .collect();
            let mut next_chunk = 0usize;
            for rep in 0..=reps {
                let start = Instant::now();
                let count = if rep == 0 { warmup_chunks } else { chunks } as usize;
                for _ in 0..count {
                    let chunk = next_chunk;
                    next_chunk += 1;
                    let controls = &pool[chunk % CONTROL_POOL];
                    for (a, arena) in sims.iter_mut().enumerate() {
                        for c in 0..teams.len() {
                            arena.set_car_controls(c, controls[a * teams.len() + c]);
                        }
                        for _ in 0..TICKS_PER_ACTION {
                            arena.step_tick();
                        }
                    }
                }
                if rep > 0 {
                    samples.push(
                        (arenas as f64 * (chunks * TICKS_PER_ACTION) as f64)
                            / start.elapsed().as_secs_f64(),
                    );
                }
            }
        }
        Target::Ours(backend) => {
            let cars = teams
                .iter()
                .map(|&team| {
                    Ok(ffi::CarSpec {
                        team: team as u8,
                        config: ffi::car_body_config(0)?,
                    })
                })
                .collect::<Result<Vec<_>, String>>()?;
            let mut batch = ffi::Batch::new(
                meshes,
                &ffi::BatchDesc {
                    backend: backend.code(),
                    game_mode: mode_code(mode),
                    mutators: to_ffi_mutators(&mutators),
                    cars,
                    rng_seeds: (0..arenas as u64).collect(),
                },
            )?;
            for a in 0..arenas as u32 {
                batch.reset_kickoff(a, Some(a as u64))?;
            }
            let ffi_pool: Vec<Vec<ffi::CarControls>> = pool
                .iter()
                .map(|set| set.iter().map(to_ffi_controls).collect())
                .collect();
            let mut next_chunk = 0usize;
            for rep in 0..=reps {
                let start = Instant::now();
                let count = if rep == 0 { warmup_chunks } else { chunks } as usize;
                for _ in 0..count {
                    batch.set_all_controls(&ffi_pool[next_chunk % CONTROL_POOL])?;
                    next_chunk += 1;
                    batch.step(TICKS_PER_ACTION)?;
                }
                if rep > 0 {
                    samples.push(
                        (arenas as f64 * (chunks * TICKS_PER_ACTION) as f64)
                            / start.elapsed().as_secs_f64(),
                    );
                }
            }
            let errors = batch.snapshot(0)?.scalars.error_flags;
            if errors != 0 {
                return Err(format!(
                    "capacity error flags 0x{errors:x} during benchmark"
                ));
            }
        }
    }
    Ok(stats(samples))
}
