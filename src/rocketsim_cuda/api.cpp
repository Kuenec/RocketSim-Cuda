#include "rocketsim_cuda/rocketsim_cuda.hpp"

#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <thread>

#include "rocketsim_cuda/convert.hpp"
#include "rocketsim_cuda/cuda_batch.hpp"
#include "rsc/host/builder.hpp"

namespace rocketsim_cuda {

using namespace convert;

namespace {

constexpr uint64_t k_snapshot_magic = 0x5253435633415245ULL;

void require(bool condition, const char *message) {
  if (!condition) {
    throw std::invalid_argument(message);
  }
}

}

MutatorConfig default_mutators(GameMode mode) {
  return to_public(rsc::default_mutators(to_internal(mode)));
}

CarBodyConfig car_body_config(CarBody body) {
  return to_public(rsc::car_body_preset(static_cast<int>(body)));
}

CarCore car_core_default() {
  rsc::CarState d = rsc::car_state_default();
  CarState s = to_public(d);
  CarCore c{};
  c.phys = s.phys;
  c.controls = s.controls;
  c.flip_rel_torque = s.flip_rel_torque;
  c.boost = s.boost;
  return c;
}

struct MeshSet::Impl {
  rsc::MeshLibrary library;
};

MeshSet::MeshSet(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

MeshSet::~MeshSet() = default;

std::shared_ptr<const MeshSet>
MeshSet::load_directory(const std::string &collision_meshes_folder) {
  auto impl = std::make_unique<Impl>(
      Impl{rsc::MeshLibrary::load_directory(collision_meshes_folder)});
  return std::shared_ptr<const MeshSet>(new MeshSet(std::move(impl)));
}

bool MeshSet::has_game_mode(GameMode mode) const {
  if (mode == GameMode::TheVoid) {
    return true;
  }
  return impl_->library.has_mode(to_internal(mode));
}

class CpuStepPool {
public:
  CpuStepPool(const rsc::ArenaStatic &view, std::vector<rsc::Arena> &arenas,
              uint32_t threads)
      : view_(view), arenas_(arenas) {
    uint32_t n = static_cast<uint32_t>(arenas.size());
    uint32_t count = std::max(1u, std::min(threads, n));
    for (uint32_t w = 0; w < count; ++w) {
      scratch_.push_back(std::make_unique<rsc::StepScratch>());
    }
    for (uint32_t w = 1; w < count; ++w) {
      workers_.emplace_back([this, w] { run_worker(w); });
    }
  }

  CpuStepPool(const CpuStepPool &) = delete;
  CpuStepPool &operator=(const CpuStepPool &) = delete;

  ~CpuStepPool() {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stop_ = true;
    }
    start_.notify_all();
    for (std::thread &t : workers_) {
      t.join();
    }
  }

  void step(uint32_t ticks) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      ticks_ = ticks;
      pending_ = static_cast<uint32_t>(workers_.size());
      ++generation_;
    }
    start_.notify_all();
    step_range(0, ticks);
    std::unique_lock<std::mutex> lock(mutex_);
    done_.wait(lock, [this] { return pending_ == 0; });
  }

private:
  void step_range(uint32_t worker, uint32_t ticks) {
    size_t n = arenas_.size();
    size_t count = scratch_.size();
    size_t first = n * worker / count;
    size_t last = n * (worker + 1) / count;
    for (size_t i = first; i < last; ++i) {
      rsc::Arena &a = arenas_[i];
      a.counters = rsc::ArenaEventCounters{};
      for (uint32_t t = 0; t < ticks; ++t) {
        rsc::arena_step_tick(view_, a, *scratch_[worker]);
      }
    }
  }

  void run_worker(uint32_t worker) {
    uint64_t seen = 0;
    for (;;) {
      uint32_t ticks;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        start_.wait(lock, [&] { return stop_ || generation_ != seen; });
        if (stop_) {
          return;
        }
        seen = generation_;
        ticks = ticks_;
      }
      step_range(worker, ticks);
      {
        std::lock_guard<std::mutex> lock(mutex_);
        --pending_;
      }
      done_.notify_one();
    }
  }

  const rsc::ArenaStatic &view_;
  std::vector<rsc::Arena> &arenas_;
  std::vector<std::unique_ptr<rsc::StepScratch>> scratch_;
  std::vector<std::thread> workers_;
  std::mutex mutex_;
  std::condition_variable start_;
  std::condition_variable done_;
  uint64_t generation_ = 0;
  uint32_t ticks_ = 0;
  uint32_t pending_ = 0;
  bool stop_ = false;
};

struct ArenaBatch::Impl {
  Backend backend;
  GameMode mode;
  std::shared_ptr<const MeshSet> meshes;
  std::unique_ptr<rsc::HostStatic> st;
  uint32_t num_arenas;
  uint32_t num_cars;
  uint32_t num_pads;
  std::vector<CarInfo> car_infos;
  std::vector<rsc::MutatorConfig> mutators;
  std::vector<rsc::Arena> host;
  std::unique_ptr<CpuStepPool> cpu_pool;
  std::vector<CarControls> staged_controls;
  std::vector<uint8_t> staged_mask;
  bool any_staged = false;
#ifdef RSC_HAS_CUDA
  std::unique_ptr<rsc::CudaBatch> cuda;
#endif

  bool on_device() const {
#ifdef RSC_HAS_CUDA
    return cuda != nullptr;
#else
    return false;
#endif
  }

  void check_arena(uint32_t arena) const {
    if (arena >= num_arenas) {
      throw std::out_of_range("arena index out of range");
    }
  }

  void check_car(uint32_t car) const {
    if (car >= num_cars) {
      throw std::out_of_range("car index out of range");
    }
  }

  void require_device(const void *pointer) const {
    if (!on_device()) {
      throw std::invalid_argument(
          "device memory requires a batch on the CUDA backend");
    }
    if (pointer == nullptr) {
      throw std::invalid_argument("device pointer is null");
    }
  }

  void flush_controls() {
#ifdef RSC_HAS_CUDA
    if (cuda && any_staged) {
      cuda->apply_staged_controls(staged_controls.data(), staged_mask.data(),
                                  num_cars);
      std::fill(staged_mask.begin(), staged_mask.end(), 0);
      any_staged = false;
    }
#endif
  }

  void apply(uint32_t arena, const rsc::ArenaOp &op) {
    check_arena(arena);
#ifdef RSC_HAS_CUDA
    if (cuda) {
      flush_controls();
      cuda->apply_op(arena, op);
      return;
    }
#endif
    rsc::arena_apply_op(st->view(), host[arena], op);
  }

  void read_raw(uint32_t arena, size_t offset, size_t size, void *dst) {
    check_arena(arena);
#ifdef RSC_HAS_CUDA
    if (cuda) {
      flush_controls();
      cuda->read(arena, offset, size, dst);
      return;
    }
#endif
    std::memcpy(dst, reinterpret_cast<const char *>(&host[arena]) + offset,
                size);
  }

  void export_range(const StateBuffers &out, uint32_t first, uint32_t count,
                    MemorySpace space) {
    if (out.tiles && mode != GameMode::Dropshot) {
      throw std::invalid_argument("tile states exist only in Dropshot");
    }
#ifdef RSC_HAS_CUDA
    if (cuda) {
      flush_controls();
      cuda->export_state(out, first, count, num_cars, num_pads, space);
      return;
    }
#endif
    if (space == MemorySpace::Device) {
      require_device(&out);
    }
    for (uint32_t i = 0; i < count; ++i) {
      convert::export_arena(st->view(), host[first + i], i, out);
    }
  }

  ArenaInfo info(uint32_t arena) {
    check_arena(arena);
    ArenaInfo out;
    StateBuffers b;
    b.infos = &out;
    export_range(b, arena, 1, MemorySpace::Host);
    return out;
  }

  void stage_controls(uint32_t arena, uint32_t car, const CarControls &c) {
    check_arena(arena);
    check_car(car);
    if (on_device()) {
      size_t slot = static_cast<size_t>(arena) * num_cars + car;
      staged_controls[slot] = c;
      staged_mask[slot] = 1;
      any_staged = true;
      return;
    }
    host[arena].cars[car].state.controls = convert::to_internal(c);
  }

  void launch_step(uint32_t ticks) {
#ifdef RSC_HAS_CUDA
    if (cuda) {
      flush_controls();
      cuda->step(ticks);
      return;
    }
#endif
    cpu_pool->step(ticks);
  }

  void wait_idle() {
#ifdef RSC_HAS_CUDA
    if (cuda) {
      cuda->sync();
    }
#endif
  }

  rsc::Arena download(uint32_t arena) {
    check_arena(arena);
    rsc::Arena a;
#ifdef RSC_HAS_CUDA
    if (cuda) {
      flush_controls();
      cuda->download(arena, a);
      return a;
    }
#endif
    a = host[arena];
    return a;
  }

  void store(uint32_t arena, const rsc::Arena &a) {
    check_arena(arena);
    mutators[arena] = a.mutators;
#ifdef RSC_HAS_CUDA
    if (cuda) {
      flush_controls();
      cuda->upload(arena, 1, &a);
      return;
    }
#endif
    host[arena] = a;
  }
};

ArenaBatch::ArenaBatch(Backend backend, std::shared_ptr<const MeshSet> meshes,
                       const BatchConfig &config)
    : impl_(std::make_unique<Impl>()) {
  require(meshes != nullptr, "mesh set is null");
  require(config.cars.size() <= k_max_cars, "too many cars per arena");
  require(!config.rng_seeds.empty(), "rng_seeds must list one seed per arena");
  Impl &im = *impl_;
  im.backend = backend;
  im.mode = config.game_mode;
  im.meshes = std::move(meshes);
  im.num_arenas = static_cast<uint32_t>(config.rng_seeds.size());
  im.num_cars = static_cast<uint32_t>(config.cars.size());
  rsc::ArenaSetup setup;
  setup.mode = to_internal(config.game_mode);
  setup.mem_weight_mode = static_cast<rsc::MemWeightMode>(
      static_cast<uint8_t>(config.mem_weight_mode));
  setup.min_pos = to_internal(config.min_pos);
  setup.max_pos = to_internal(config.max_pos);
  setup.max_aabb_len = config.max_aabb_len;
  setup.no_ball_rot = config.no_ball_rot;
  setup.has_custom_pads = config.custom_boost_pads.has_value();
  if (config.custom_boost_pads) {
    for (const BoostPadConfig &p : *config.custom_boost_pads) {
      setup.custom_pads.push_back({to_internal(p.pos), p.is_big});
    }
  }
  im.st = std::make_unique<rsc::HostStatic>(im.meshes->impl().library, setup);
  const rsc::ArenaStatic &view = im.st->view();
  im.num_pads = view.has_pads ? view.pads.num_pads : 0;
  std::vector<rsc::CarSpec> cars;
  for (uint32_t c = 0; c < im.num_cars; ++c) {
    const CarSpec &spec = config.cars[c];
    rsc::CarSpec internal{to_internal(spec.team), to_internal(spec.config)};
    cars.push_back(internal);
    im.car_infos.push_back({c, spec.team, to_public(internal.config)});
  }
  rsc::MutatorConfig mutators = to_internal(config.mutators);
  im.mutators.assign(im.num_arenas, mutators);
  size_t num_slots = static_cast<size_t>(im.num_arenas) * im.num_cars;
  im.staged_controls.resize(num_slots);
  im.staged_mask.assign(num_slots, 0);
  if (backend == Backend::Cuda) {
#ifdef RSC_HAS_CUDA
    im.cuda = std::make_unique<rsc::CudaBatch>(*im.st, im.num_arenas);
    constexpr uint32_t chunk = 64;
    std::vector<rsc::Arena> staging(chunk);
    for (uint32_t first = 0; first < im.num_arenas; first += chunk) {
      uint32_t count = std::min(chunk, im.num_arenas - first);
      for (uint32_t i = 0; i < count; ++i) {
        rsc::create_arena(*im.st, mutators, config.rng_seeds[first + i], cars,
                          staging[i]);
      }
      im.cuda->upload(first, count, staging.data());
    }
#else
    throw std::runtime_error(
        "this build of rocketsim_cuda has no CUDA backend");
#endif
  } else {
    im.host.resize(im.num_arenas);
    for (uint32_t i = 0; i < im.num_arenas; ++i) {
      rsc::create_arena(*im.st, mutators, config.rng_seeds[i], cars,
                        im.host[i]);
    }
    uint32_t threads = config.cpu_threads != 0
                           ? config.cpu_threads
                           : std::max(1u, std::thread::hardware_concurrency());
    im.cpu_pool = std::make_unique<CpuStepPool>(view, im.host, threads);
  }
}

ArenaBatch::~ArenaBatch() = default;

Backend ArenaBatch::backend() const { return impl_->backend; }

GameMode ArenaBatch::game_mode() const { return impl_->mode; }

uint32_t ArenaBatch::num_arenas() const { return impl_->num_arenas; }

uint32_t ArenaBatch::num_cars() const { return impl_->num_cars; }

uint32_t ArenaBatch::num_boost_pads() const { return impl_->num_pads; }

void ArenaBatch::set_car_controls(uint32_t arena, uint32_t car,
                                  const CarControls &controls) {
  impl_->stage_controls(arena, car, controls);
}

void ArenaBatch::set_all_controls(std::span<const CarControls> controls) {
  Impl &im = *impl_;
  require(controls.size() == static_cast<size_t>(im.num_arenas) * im.num_cars,
          "controls must hold num_arenas * num_cars entries");
  for (uint32_t a = 0; a < im.num_arenas; ++a) {
    for (uint32_t c = 0; c < im.num_cars; ++c) {
      im.stage_controls(a, c,
                        controls[static_cast<size_t>(a) * im.num_cars + c]);
    }
  }
}

void ArenaBatch::set_all_controls_device(const CarControls *device_controls) {
  Impl &im = *impl_;
  im.require_device(device_controls);
#ifdef RSC_HAS_CUDA
  im.flush_controls();
  im.cuda->set_controls_device(device_controls, im.num_cars);
#endif
}

void ArenaBatch::reset_to_random_kickoff_masked(const uint8_t *mask,
                                                const uint64_t *seeds,
                                                MemorySpace space) {
  Impl &im = *impl_;
  require(mask != nullptr, "reset mask is null");
#ifdef RSC_HAS_CUDA
  if (im.cuda) {
    im.flush_controls();
    im.cuda->reset_kickoff_masked(mask, seeds, space);
    return;
  }
#endif
  if (space == MemorySpace::Device) {
    im.require_device(mask);
  }
  for (uint32_t a = 0; a < im.num_arenas; ++a) {
    if (mask[a]) {
      rsc::ArenaOp op{};
      op.kind = rsc::ArenaOpKind::Kickoff;
      op.kickoff = {seeds != nullptr, seeds ? seeds[a] : 0};
      im.apply(a, op);
    }
  }
}

void ArenaBatch::export_state(const StateBuffers &out, MemorySpace space) {
  impl_->export_range(out, 0, impl_->num_arenas, space);
}

void *ArenaBatch::cuda_stream() const {
#ifdef RSC_HAS_CUDA
  if (impl_->cuda) {
    return impl_->cuda->stream();
  }
#endif
  return nullptr;
}

void ArenaBatch::step(uint32_t ticks) {
  impl_->launch_step(ticks);
  impl_->wait_idle();
}

void ArenaBatch::step_async(uint32_t ticks) { impl_->launch_step(ticks); }

void ArenaBatch::sync() { impl_->wait_idle(); }

ArenaState ArenaBatch::get_arena_state(uint32_t arena) {
  Impl &im = *impl_;
  im.check_arena(arena);
  ArenaInfo info;
  BallState ball;
  TileStates tiles;
  std::vector<CarState> cars(im.num_cars);
  std::vector<PadStateView> pads(im.num_pads);
  StateBuffers b;
  b.infos = &info;
  b.balls = &ball;
  b.cars = cars.data();
  b.pads = pads.data();
  b.tiles = im.mode == GameMode::Dropshot ? &tiles : nullptr;
  im.export_range(b, arena, 1, MemorySpace::Host);
  const rsc::ArenaStatic &view = im.st->view();
  ArenaState s;
  s.game_mode = im.mode;
  s.tick_count = info.tick_count;
  s.car_infos = im.car_infos;
  s.cars = std::move(cars);
  s.ball = ball;
  for (uint32_t p = 0; p < im.num_pads; ++p) {
    s.boost_pad_configs.push_back(
        {to_public(view.pads.configs[p].pos), view.pads.configs[p].is_big});
    BoostPadState ps;
    ps.cooldown = pads[p].cooldown;
    ps.touching_car_idxs.assign(pads[p].touching,
                                pads[p].touching + pads[p].num_touching);
    s.boost_pads.push_back(std::move(ps));
  }
  if (b.tiles) {
    s.tile_states = tiles;
  }
  return s;
}

ArenaInfo ArenaBatch::get_arena_info(uint32_t arena) {
  return impl_->info(arena);
}

CarState ArenaBatch::get_car_state(uint32_t arena, uint32_t car) {
  Impl &im = *impl_;
  im.check_car(car);
  rsc::CarState s;
  im.read_raw(arena,
              offsetof(rsc::Arena, cars) + car * sizeof(rsc::Car) +
                  offsetof(rsc::Car, state),
              sizeof(s), &s);
  return to_public(s);
}

BallState ArenaBatch::get_ball_state(uint32_t arena) {
  rsc::BallState s;
  impl_->read_raw(arena,
                  offsetof(rsc::Arena, ball) + offsetof(rsc::Ball, state),
                  sizeof(s), &s);
  return to_public(s);
}

BoostPadState ArenaBatch::get_boost_pad_state(uint32_t arena, uint32_t pad) {
  Impl &im = *impl_;
  if (pad >= im.num_pads) {
    throw std::out_of_range("boost pad index out of range");
  }
  std::vector<PadStateView> pads(im.num_pads);
  StateBuffers b;
  b.pads = pads.data();
  im.check_arena(arena);
  im.export_range(b, arena, 1, MemorySpace::Host);
  BoostPadState s;
  s.cooldown = pads[pad].cooldown;
  s.touching_car_idxs.assign(pads[pad].touching,
                             pads[pad].touching + pads[pad].num_touching);
  return s;
}

TileStates ArenaBatch::get_tile_states(uint32_t arena) {
  Impl &im = *impl_;
  if (im.mode != GameMode::Dropshot) {
    throw std::logic_error("this arena has no dropshot tiles");
  }
  rsc::TileStates t;
  im.read_raw(arena, offsetof(rsc::Arena, tiles), sizeof(t), &t);
  return to_public(t);
}

MutatorConfig ArenaBatch::get_mutator_config(uint32_t arena) {
  impl_->check_arena(arena);
  return to_public(impl_->mutators[arena]);
}

std::vector<ArenaEvent> ArenaBatch::get_last_step_events(uint32_t arena) {
  Impl &im = *impl_;
  uint32_t count = im.info(arena).num_events;
  std::vector<rsc::ArenaEvent> events(count);
  if (count != 0) {
    im.read_raw(arena, offsetof(rsc::Arena, events),
                sizeof(rsc::ArenaEvent) * count, events.data());
  }
  rsc::LandedWheels landed[rsc::k_max_cars];
  im.read_raw(arena, offsetof(rsc::Arena, landed), sizeof(landed), landed);
  std::vector<ArenaEvent> out;
  for (const rsc::ArenaEvent &e : events) {
    out.push_back(to_public(e, landed[e.car_idx]));
  }
  return out;
}

EventCounts ArenaBatch::get_event_counts(uint32_t arena) {
  rsc::ArenaEventCounters c;
  impl_->read_raw(arena, offsetof(rsc::Arena, counters), sizeof(c), &c);
  return to_public(c);
}

void ArenaBatch::set_car_state(uint32_t arena, uint32_t car,
                               const CarCore &core) {
  impl_->check_car(car);
  rsc::ArenaOp op{};
  op.kind = rsc::ArenaOpKind::SetCarCore;
  op.index = car;
  op.car = core_to_internal(core);
  impl_->apply(arena, op);
}

void ArenaBatch::set_car_state_full(uint32_t arena, uint32_t car,
                                    const CarState &state) {
  impl_->check_car(car);
  rsc::ArenaOp op{};
  op.kind = rsc::ArenaOpKind::SetCarFull;
  op.index = car;
  op.car = to_internal(state);
  impl_->apply(arena, op);
}

void ArenaBatch::set_ball_state(uint32_t arena, const BallState &state) {
  rsc::ArenaOp op{};
  op.kind = rsc::ArenaOpKind::SetBall;
  op.ball = to_internal(state);
  impl_->apply(arena, op);
}

void ArenaBatch::set_boost_pad_state(uint32_t arena, uint32_t pad,
                                     const BoostPadState &state) {
  Impl &im = *impl_;
  rsc::ArenaOp op{};
  op.kind = rsc::ArenaOpKind::SetPad;
  op.index = pad;
  op.pad = rsc::make_pad_op(*im.st, im.num_cars, pad, state.cooldown,
                            state.touching_car_idxs);
  im.apply(arena, op);
}

void ArenaBatch::set_tile_states(uint32_t arena, const TileStates &states) {
  Impl &im = *impl_;
  if (im.mode != GameMode::Dropshot) {
    throw std::logic_error("this arena has no dropshot tiles");
  }
  rsc::ArenaOp op{};
  op.kind = rsc::ArenaOpKind::SetTiles;
  op.tiles = to_internal(states);
  im.apply(arena, op);
}

void ArenaBatch::set_mutator_config(uint32_t arena,
                                    const MutatorConfig &mutators) {
  Impl &im = *impl_;
  im.check_arena(arena);
  rsc::MutatorConfig next = to_internal(mutators);
  rsc::ArenaOp op{};
  op.kind = rsc::ArenaOpKind::SetMutators;
  op.mutators = rsc::make_mutator_op(im.mutators[arena], next);
  im.apply(arena, op);
  im.mutators[arena] = next;
}

void ArenaBatch::reset_to_random_kickoff(uint32_t arena,
                                         std::optional<uint64_t> seed) {
  rsc::ArenaOp op{};
  op.kind = rsc::ArenaOpKind::Kickoff;
  op.kickoff = {seed.has_value(), seed.value_or(0)};
  impl_->apply(arena, op);
}

void ArenaBatch::copy_arenas(const ArenaBatch &source,
                             std::span<const uint32_t> source_arenas,
                             std::span<const uint32_t> dest_arenas) {
  Impl &dst = *impl_;
  Impl &src = *source.impl_;
  require(source_arenas.size() == dest_arenas.size(),
          "source and destination index lists differ in length");
  require(src.meshes == dst.meshes && src.mode == dst.mode &&
              src.num_cars == dst.num_cars && src.num_pads == dst.num_pads &&
              src.st->view().world.num_static_bodies ==
                  dst.st->view().world.num_static_bodies,
          "arena batches have different static configurations");
  for (size_t k = 0; k < source_arenas.size(); ++k) {
    src.check_arena(source_arenas[k]);
    dst.check_arena(dest_arenas[k]);
  }
  std::vector<rsc::MutatorConfig> copied;
  for (uint32_t s : source_arenas) {
    copied.push_back(src.mutators[s]);
  }
#ifdef RSC_HAS_CUDA
  if (src.cuda && dst.cuda) {
    src.flush_controls();
    dst.flush_controls();
    dst.cuda->copy_from(*src.cuda, source_arenas.data(), dest_arenas.data(),
                        static_cast<uint32_t>(source_arenas.size()));
    for (size_t k = 0; k < dest_arenas.size(); ++k) {
      dst.mutators[dest_arenas[k]] = copied[k];
    }
    return;
  }
#endif
  std::vector<rsc::Arena> staged;
  staged.reserve(source_arenas.size());
  for (uint32_t s : source_arenas) {
    staged.push_back(src.download(s));
  }
  for (size_t k = 0; k < dest_arenas.size(); ++k) {
    dst.store(dest_arenas[k], staged[k]);
  }
}

std::vector<uint8_t> ArenaBatch::save_arena(uint32_t arena) {
  rsc::Arena a = impl_->download(arena);
  std::vector<uint8_t> out(sizeof(uint64_t) * 2 + sizeof(rsc::Arena));
  uint64_t header[2] = {k_snapshot_magic, sizeof(rsc::Arena)};
  std::memcpy(out.data(), header, sizeof(header));
  std::memcpy(out.data() + sizeof(header), &a, sizeof(rsc::Arena));
  return out;
}

void ArenaBatch::load_arena(uint32_t arena, std::span<const uint8_t> snapshot) {
  uint64_t header[2];
  require(snapshot.size() == sizeof(header) + sizeof(rsc::Arena),
          "arena snapshot has the wrong size");
  std::memcpy(header, snapshot.data(), sizeof(header));
  require(header[0] == k_snapshot_magic && header[1] == sizeof(rsc::Arena),
          "arena snapshot header does not match this build");
  rsc::Arena a;
  std::memcpy(&a, snapshot.data() + sizeof(header), sizeof(rsc::Arena));
  impl_->store(arena, a);
}

}
