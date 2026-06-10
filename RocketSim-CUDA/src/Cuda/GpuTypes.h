#pragma once
#include <cstdint>

// 2026-06-10: Plain-POD state mirrors shared between host bridge code and CUDA kernels.
// No RocketSim/Bullet/CUDA includes allowed here - this header must compile everywhere.
// Field set mirrors what the simulation actually persists tick-to-tick; it grows as
// phases are ported. All floats: the CPU reference is compiled -ffp-contract=off and
// kernels -fmad=false so both sides compute plain IEEE mul+add.

namespace rscuda {

struct GpuVec3 {
	float x, y, z;
};

// Row-major 3x3 basis, same layout as btMatrix3x3 rows
struct GpuMat3 {
	GpuVec3 row[3];
};

// Rigid body dynamic state in BULLET UNITS (BT), mirroring btRigidBody fields that
// persist tick-to-tick. basis/pos = worldTransform; invInertiaWorld is derived but
// carried so phase compares can catch divergence at its source.
struct GpuBodyState {
	GpuMat3 basis;           // btTransform basis rows (row-major, btMatrix3x3 m_el)
	GpuVec3 pos;             // btTransform origin (BT units)
	GpuVec3 linVel;          // m_linearVelocity (BT units)
	GpuVec3 angVel;          // m_angularVelocity (rad/s)
	GpuVec3 totalForce;      // m_totalForce (incl. gravity, applied pre-step)
	GpuVec3 totalTorque;     // m_totalTorque
	GpuMat3 invInertiaWorld; // m_invInertiaTensorWorld
	int activationState;     // bullet ACTIVE_TAG / ISLAND_SLEEPING / etc.
	float deactivationTime;
};

// Per-body-slot constants (identical across arenas; slot 0 = ball, 1.. = cars)
struct GpuBodyConsts {
	float invMass;
	GpuVec3 invInertiaLocal;
	float linDampFactor;     // btPow(1 - linearDamping, dt), host-precomputed
	float angDampFactor;     // btPow(1 - angularDamping, dt), host-precomputed
	float linSleepThreshSq;  // m_linearSleepingThreshold^2 (host-precomputed product)
	float angSleepThreshSq;
	uint8_t noRot;           // btRigidBody::m_noRot (RocketSim extension)
};

// Phase IDs for the monolithic kernel, in CPU reference order within a tick.
// Dev-only phase windows let --gpu-compare validate one phase at a time; production
// runs the full range.
enum GpuPhase : int {
	GPU_PHASE_PREDICT = 0,    // btDiscreteDynamicsWorld::predictUnconstraintMotion
	GPU_PHASE_COLLISION = 1,  // performDiscreteCollisionDetection (not ported yet)
	GPU_PHASE_SOLVE = 2,      // islands + solveConstraints (not ported yet)
	GPU_PHASE_INTEGRATE = 3,  // integrateTransforms
	GPU_PHASE_ACTIVATION = 4, // updateActivationState
	GPU_PHASE_COUNT
};

struct GpuCarControls {
	float throttle, steer, pitch, yaw, roll;
	uint8_t jump, boost, handbrake;
};

struct GpuBallHitInfo {
	uint8_t isValid;
	GpuVec3 relativePosOnBall;
	GpuVec3 ballPos;
	GpuVec3 extraHitVel;
	uint64_t tickCountWhenHit;
	uint64_t tickCountWhenExtraImpulseApplied;
};

// Persistent per-wheel state (refined during the vehicle-dynamics port)
struct GpuWheelState {
	float rotation;
	float deltaRotation;
	float steerAngle;
	uint8_t hasContact;
};

// Mirror of RocketSim CarState game fields + car-internal persistent extras
struct GpuCarGameState {
	uint64_t tickCountSinceUpdate;
	uint8_t isOnGround;
	uint8_t wheelsWithContact[4];
	uint8_t hasJumped, hasDoubleJumped, hasFlipped;
	GpuVec3 flipRelTorque;
	float jumpTime, flipTime;
	uint8_t isFlipping, isJumping;
	float airTime, airTimeSinceJump;
	float boost;
	float timeSinceBoosted;
	uint8_t isBoosting;
	float boostingTime;
	uint8_t isSupersonic;
	float supersonicTime;
	float handbrakeVal;
	uint8_t isAutoFlipping;
	float autoFlipTimer;
	float autoFlipTorqueScale;
	uint8_t worldContact_hasContact;
	GpuVec3 worldContact_normal;
	uint32_t carContact_otherCarID;
	float carContact_cooldownTimer;
	uint8_t isDemoed;
	float demoRespawnTimer;
	GpuBallHitInfo ballHitInfo;
	GpuCarControls lastControls;
	GpuVec3 velocityImpulseCache;   // Car::_velocityImpulseCache
	GpuWheelState wheels[4];
	uint32_t carID;
	uint8_t team;                   // 0 blue, 1 orange
};

struct GpuPadState {
	uint8_t isActive;
	float cooldown;
	uint32_t prevLockedCarID;
	int32_t lockedCarIdx;           // -1 = none (mirrors BoostPadState::curLockedCar)
};

} // namespace rscuda
