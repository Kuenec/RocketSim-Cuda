mod bench;
mod ffi;
mod flatten;
mod mirror;
mod scenario;

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use rocketsim::{CarControls, DemoMode, GameMode, Team};
use rocketsim_test::rlpr::Recording;

use scenario::{Action, Backend, ControlSource, MutatorChange, Scenario};

enum Command {
    Gate,
    Bench,
}

struct Options {
    command: Command,
    bench_targets: Vec<bench::Target>,
    bench_arenas: Vec<usize>,
    bench_reps: usize,
    bench_ticks: u32,
    bench_warmup: u32,
    backends: Vec<Backend>,
    filter: Vec<String>,
    quick: bool,
    meshes: PathBuf,
    recordings: PathBuf,
}

fn parse_args() -> Result<Options, String> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("..");
    let mut opts = Options {
        command: Command::Gate,
        bench_targets: vec![
            bench::Target::Upstream,
            bench::Target::Ours(Backend::Cpu),
            bench::Target::Ours(Backend::Cuda),
        ],
        bench_arenas: vec![100, 1000, 10000],
        bench_reps: 5,
        bench_ticks: 240,
        bench_warmup: 1200,
        backends: vec![Backend::Cpu, Backend::Cuda],
        filter: Vec::new(),
        quick: false,
        meshes: root.join("collision_meshes"),
        recordings: root.join("build/upstream/rs/rocketsim_test/recordings"),
    };
    let mut args = std::env::args().skip(1);
    opts.command = match args.next().as_deref() {
        Some("gate") => Command::Gate,
        Some("bench") => Command::Bench,
        other => {
            return Err(format!(
                "unknown command {other:?}; usage: rsv3_harness gate [--backend cpu|cuda|both] [--filter NAME]... [--quick] [--meshes DIR] [--recordings DIR] | rsv3_harness bench [--targets upstream,cpu,cpu-mt,cuda] [--arenas 100,1000,10000] [--reps N] [--ticks N] [--warmup N]"
            ));
        }
    };
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--backend" => {
                opts.backends = match args.next().as_deref() {
                    Some("cpu") => vec![Backend::Cpu],
                    Some("cuda") => vec![Backend::Cuda],
                    Some("both") => vec![Backend::Cpu, Backend::Cuda],
                    other => return Err(format!("invalid backend {other:?}")),
                }
            }
            "--filter" => opts
                .filter
                .push(args.next().ok_or("--filter needs a value")?),
            "--quick" => opts.quick = true,
            "--meshes" => opts.meshes = PathBuf::from(args.next().ok_or("--meshes needs a value")?),
            "--recordings" => {
                opts.recordings = PathBuf::from(args.next().ok_or("--recordings needs a value")?)
            }
            "--targets" => {
                opts.bench_targets = args
                    .next()
                    .ok_or("--targets needs a value")?
                    .split(',')
                    .map(|t| match t {
                        "upstream" => Ok(bench::Target::Upstream),
                        "cpu" => Ok(bench::Target::Ours(Backend::Cpu)),
                        "cuda" => Ok(bench::Target::Ours(Backend::Cuda)),
                        "cpu-mt" => Ok(bench::Target::CpuThreads),
                        other => Err(format!("invalid bench target {other}")),
                    })
                    .collect::<Result<_, _>>()?
            }
            "--arenas" => {
                opts.bench_arenas = args
                    .next()
                    .ok_or("--arenas needs a value")?
                    .split(',')
                    .map(|n| {
                        n.parse()
                            .map_err(|e| format!("invalid arena count {n}: {e}"))
                    })
                    .collect::<Result<_, _>>()?
            }
            "--reps" => {
                let v = args.next().ok_or("--reps needs a value")?;
                opts.bench_reps = v.parse().map_err(|e| format!("invalid --reps {v}: {e}"))?;
            }
            "--warmup" => {
                let v = args.next().ok_or("--warmup needs a value")?;
                opts.bench_warmup = v
                    .parse()
                    .map_err(|e| format!("invalid --warmup {v}: {e}"))?;
            }
            "--ticks" => {
                let v = args.next().ok_or("--ticks needs a value")?;
                opts.bench_ticks = v.parse().map_err(|e| format!("invalid --ticks {v}: {e}"))?;
            }
            other => return Err(format!("unknown argument {other}")),
        }
    }
    Ok(opts)
}

fn load_recording_controls(path: &Path) -> Result<Arc<Vec<Vec<CarControls>>>, String> {
    let rec = Recording::from_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let rows: Vec<Vec<CarControls>> = rec
        .ticks
        .iter()
        .filter(|t| !t.car_records.is_empty())
        .map(|t| {
            t.car_records
                .iter()
                .map(|c| c.prev_controls.into())
                .collect()
        })
        .collect();
    if rows.is_empty() {
        return Err(format!("{} has no car inputs", path.display()));
    }
    Ok(Arc::new(rows))
}

fn team_layout(blue: usize, orange: usize, body_offset: usize) -> Vec<(Team, usize)> {
    let mut cars = Vec::new();
    for i in 0..blue {
        cars.push((Team::Blue, (body_offset + i) % 7));
    }
    for i in 0..orange {
        cars.push((Team::Orange, (body_offset + blue + i) % 7));
    }
    cars
}

fn mode_name(mode: GameMode) -> &'static str {
    match mode {
        GameMode::Soccar => "soccar",
        GameMode::Hoops => "hoops",
        GameMode::Heatseeker => "heatseeker",
        GameMode::Snowday => "snowday",
        GameMode::Dropshot => "dropshot",
        GameMode::TheVoid => "void",
    }
}

fn scenarios(quick: bool, recordings: &[(String, Arc<Vec<Vec<CarControls>>>)]) -> Vec<Scenario> {
    let scale = |ticks: u32| if quick { ticks / 10 } else { ticks };
    let mut out = Vec::new();
    let base = |name: String, mode: GameMode, cars: Vec<(Team, usize)>| Scenario {
        name,
        game_mode: mode,
        cars,
        demo_mode: DemoMode::Normal,
        arenas: 1,
        ticks: 1000,
        controls: ControlSource::Random,
        compare_every: 1,
        ticks_per_step: 1,
        actions: Vec::new(),
        seed: 1,
    };

    let mut long = base(
        "soccar_1v1_20k".into(),
        GameMode::Soccar,
        team_layout(1, 1, 0),
    );
    long.ticks = scale(20_000);
    long.actions = vec![
        (scale(2500), Action::KickoffSeeded),
        (scale(5000), Action::KickoffUnseeded),
        (scale(7000), Action::RandomBall),
        (scale(9000), Action::RandomCars),
        (scale(11000), Action::Mutators(MutatorChange::Mixed)),
        (scale(13000), Action::PadStates),
        (scale(15000), Action::KickoffSeeded),
        (scale(17000), Action::TransplantCars),
        (scale(3000), Action::CarsCollide),
        (scale(10000), Action::CarsCollide),
        (scale(18000), Action::BallIntoGoal),
        (scale(19000), Action::SnapshotRoundTrip),
        (scale(6000), Action::KickoffMaskedSeeded),
    ];
    out.push(long);

    let mut batch = base(
        "soccar_1v1_batch128".into(),
        GameMode::Soccar,
        team_layout(1, 1, 0),
    );
    batch.arenas = 128;
    batch.ticks = scale(1500);
    batch.compare_every = 5;
    batch.seed = 2;
    batch.actions = vec![
        (scale(400), Action::CloneArena),
        (scale(500), Action::SnapshotRoundTrip),
        (scale(600), Action::CrossBatchCopy),
        (scale(800), Action::CrossBackendCopy),
        (scale(700), Action::TransplantCars),
        (scale(1000), Action::KickoffUnseeded),
        (scale(200), Action::CarsCollide),
        (scale(1200), Action::CarsCollide),
        (scale(900), Action::KickoffMaskedRng),
        (scale(1100), Action::KickoffMaskedSeeded),
    ];
    out.push(batch);

    for (name, rec) in recordings {
        let mut s = base(
            format!("soccar_recording_{name}"),
            GameMode::Soccar,
            team_layout(3, 3, 0),
        );
        s.arenas = 8;
        s.ticks = scale(4000);
        s.controls = ControlSource::Recording(rec.clone());
        s.compare_every = 2;
        s.seed = 3;
        s.actions = vec![(scale(2000), Action::KickoffSeeded)];
        out.push(s);
    }

    let layouts: [(usize, usize); 7] = [(0, 0), (1, 0), (1, 1), (2, 2), (3, 3), (2, 1), (4, 4)];
    for (mi, &mode) in scenario::GAME_MODES.iter().enumerate() {
        for (li, &(blue, orange)) in layouts.iter().enumerate() {
            let name = format!("{}_{}v{}", mode_name(mode), blue, orange);
            let mut s = base(name, mode, team_layout(blue, orange, li + mi));
            s.arenas = 4;
            s.ticks = scale(3000);
            s.compare_every = 2;
            s.seed = 100 + (mi * 10 + li) as u64;
            s.actions = vec![
                (scale(800), Action::RandomBall),
                (scale(1200), Action::KickoffSeeded),
                (scale(1800), Action::RandomCars),
                (scale(2400), Action::CloneArena),
                (scale(300), Action::CarsCollide),
                (scale(2000), Action::CarsCollide),
            ];
            s.actions.push((scale(2700), Action::KickoffMaskedRng));
            if mode == GameMode::Dropshot {
                s.actions.push((scale(1500), Action::Tiles));
            } else if mode != GameMode::TheVoid {
                s.actions.push((scale(1500), Action::PadStates));
            }
            out.push(s);
        }
    }

    for (demo, label) in [
        (DemoMode::Normal, "normal"),
        (DemoMode::OnContact, "on_contact"),
        (DemoMode::Disabled, "disabled"),
    ] {
        let mut s = base(
            format!("soccar_3v3_demos_{label}"),
            GameMode::Soccar,
            team_layout(3, 3, 0),
        );
        s.demo_mode = demo;
        s.arenas = 8;
        s.ticks = scale(3000);
        s.compare_every = 2;
        s.seed = 200;
        s.actions = (0..10)
            .map(|k| (scale(300) * k + 5, Action::CarsCollide))
            .collect();
        out.push(s);
    }

    for (change, label) in [
        (MutatorChange::Gravity, "gravity"),
        (MutatorChange::CarMass, "car_mass"),
        (MutatorChange::BallGeometry, "ball_geometry"),
        (MutatorChange::BoostAndPads, "boost"),
        (MutatorChange::Mixed, "mixed"),
    ] {
        for mode in [GameMode::Soccar, GameMode::Snowday, GameMode::Hoops] {
            let mut s = base(
                format!("{}_mutators_{label}", mode_name(mode)),
                mode,
                team_layout(2, 2, 3),
            );
            s.arenas = 4;
            s.ticks = scale(2000);
            s.compare_every = 2;
            s.seed = 300;
            s.actions = vec![
                (scale(300), Action::Mutators(change)),
                (scale(1200), Action::Mutators(change)),
            ];
            out.push(s);
        }
    }

    let mut multi = base(
        "soccar_2v2_multitick".into(),
        GameMode::Soccar,
        team_layout(2, 2, 1),
    );
    multi.arenas = 32;
    multi.ticks = scale(4000);
    multi.ticks_per_step = 8;
    multi.seed = 500;
    multi.actions = vec![
        (scale(1600), Action::KickoffUnseeded),
        (scale(2000), Action::CrossBackendCopy),
        (scale(3000), Action::SnapshotRoundTrip),
        (scale(3400), Action::KickoffMaskedSeeded),
        (scale(3600), Action::KickoffMaskedRng),
        (scale(2400), Action::Mutators(MutatorChange::Mixed)),
    ];
    out.push(multi);

    for mode in [
        GameMode::Soccar,
        GameMode::Heatseeker,
        GameMode::Snowday,
        GameMode::Hoops,
    ] {
        let mut goals = base(
            format!("{}_goals", mode_name(mode)),
            mode,
            team_layout(1, 1, 2),
        );
        goals.arenas = 8;
        goals.ticks = scale(2400);
        goals.controls = ControlSource::Idle;
        goals.seed = 600;
        goals.actions = (0..6)
            .map(|k| (scale(400) * k + 1, Action::BallIntoGoal))
            .collect();
        out.push(goals);
    }

    let mut idle = base(
        "soccar_kickoffs_idle".into(),
        GameMode::Soccar,
        team_layout(3, 3, 0),
    );
    idle.arenas = 16;
    idle.ticks = scale(1200);
    idle.controls = ControlSource::Idle;
    idle.seed = 400;
    idle.actions = vec![(scale(600), Action::KickoffUnseeded)];
    out.push(idle);
    out
}

fn run_bench(opts: &Options, meshes: &ffi::MeshSet) -> ExitCode {
    let mut ok = true;
    for &arenas in &opts.bench_arenas {
        for target in &opts.bench_targets {
            match bench::run(
                meshes,
                target,
                arenas,
                opts.bench_reps,
                opts.bench_ticks,
                opts.bench_warmup,
            ) {
                Ok(s) => println!(
                    "bench soccar_1v1 {:<14} arenas {:>6}  median {:>12.0}  min {:>12.0}  max {:>12.0} arena-ticks/s  samples {:?}",
                    target.name(),
                    arenas,
                    s.median,
                    s.min,
                    s.max,
                    s.samples
                        .iter()
                        .map(|v| v.round() as u64)
                        .collect::<Vec<_>>()
                ),
                Err(e) => {
                    ok = false;
                    println!(
                        "bench soccar_1v1 {:<14} arenas {:>6}  FAILED {e}",
                        target.name(),
                        arenas
                    );
                }
            }
        }
    }
    if ok {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn main() -> ExitCode {
    let opts = match parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::from(2);
        }
    };
    if let Err(e) = ffi::check_abi() {
        eprintln!("ABI check failed: {e}");
        return ExitCode::FAILURE;
    }
    if let Err(e) = rocketsim::init(&opts.meshes, true) {
        eprintln!("upstream init failed: {e}");
        return ExitCode::FAILURE;
    }
    let meshes = match ffi::MeshSet::load(opts.meshes.to_str().unwrap()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if let Command::Bench = opts.command {
        return run_bench(&opts, &meshes);
    }
    let mut recordings = Vec::new();
    for name in ["nexto_1v1_300s", "wisp_3v3_300s"] {
        let path = opts.recordings.join(format!("{name}.rlpr.zst"));
        match load_recording_controls(&path) {
            Ok(rec) => recordings.push((name.to_string(), rec)),
            Err(e) => {
                eprintln!("recording unavailable: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
    let dropshot_loaded = opts.meshes.join("dropshot").is_dir();

    let mut failures = 0usize;
    let mut blocked = 0usize;
    let mut passed = 0usize;
    for s in scenarios(opts.quick, &recordings) {
        if !opts.filter.is_empty() && !opts.filter.iter().any(|f| s.name.contains(f.as_str())) {
            continue;
        }
        if s.game_mode == GameMode::Dropshot && !dropshot_loaded {
            println!("BLOCKED {} (no dropshot collision meshes)", s.name);
            blocked += 1;
            continue;
        }
        for backend in &opts.backends {
            let start = Instant::now();
            match scenario::run(&s, backend, &meshes) {
                Ok(o) => {
                    passed += 1;
                    let c = &o.coverage;
                    println!(
                        "PASS    {:<34} {:<4} {:>7} arena-ticks {:>6} cmp {:>6.1}s  goals {} touches {} bumps {} demos {} pickups {} landings {}",
                        s.name,
                        backend.name(),
                        o.ticks,
                        o.comparisons,
                        start.elapsed().as_secs_f64(),
                        c.goals,
                        c.ball_touches,
                        c.bumps,
                        c.demos,
                        c.pickups,
                        c.landings
                    );
                }
                Err(e) => {
                    failures += 1;
                    println!("FAIL    {:<40} {:<4} {e}", s.name, backend.name());
                }
            }
        }
    }
    println!("summary: {passed} passed, {failures} failed, {blocked} blocked");
    if failures == 0 && passed > 0 {
        println!("GATES PASS");
        ExitCode::SUCCESS
    } else {
        println!("GATES FAIL");
        ExitCode::FAILURE
    }
}
