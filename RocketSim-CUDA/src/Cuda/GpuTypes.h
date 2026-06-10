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
// NOTE (2026-06-10): m_interpolationWorldTransform needs no slot here. RS-modified
// predictUnconstraintMotion overwrites it UNCONDITIONALLY each tick (all dyn bodies,
// no isActive check) as predictIntegratedTransform(world, damped vels, dt), and the
// only reader (updateAabbs) runs right after - so the collide phase recomputes it.

// Per-body-slot constants (identical across arenas; slot 0 = ball, 1.. = cars)
struct GpuBodyConsts {
	float invMass;
	GpuVec3 invInertiaLocal;
	float linDampFactor;     // btPow(1 - linearDamping, dt), host-precomputed
	float angDampFactor;     // btPow(1 - angularDamping, dt), host-precomputed
	float linSleepThreshSq;  // m_linearSleepingThreshold^2 (host-precomputed product)
	float angSleepThreshSq;
	uint8_t noRot;           // btRigidBody::m_noRot (RocketSim extension)
	GpuVec3 gravityForce;    // btRigidBody m_gravity (FORCE = acceleration * mass)
};

// Phase IDs for the monolithic kernel, in CPU reference order within a tick.
// Dev-only phase windows let --gpu-compare validate one phase at a time; production
// runs the full range.
enum GpuPhase : int {
	GPU_PHASE_GAME_PRE = 0,   // ball sleep check + car pre-ticks + pad pre + applyGravity
	GPU_PHASE_PREDICT = 1,    // btDiscreteDynamicsWorld::predictUnconstraintMotion
	GPU_PHASE_COLLISION = 2,  // performDiscreteCollisionDetection
	GPU_PHASE_SOLVE = 3,      // islands + solveConstraints
	GPU_PHASE_INTEGRATE = 4,  // integrateTransforms
	GPU_PHASE_ACTIVATION = 5, // updateActivationState
	GPU_PHASE_GAME_POST = 6,  // clearForces + car post/finish + pads + ball finish
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

// Persistent per-wheel state: fields written in tick N and read in tick N+1
// (calcFrictionImpulses runs BEFORE _UpdateWheels within a tick, so it consumes the
// previous tick's engine/brake/steer/friction values; updateWheelTransform's origin
// uses the previous tick's suspensionLength)
struct GpuWheelState {
	float steerAngle;
	float engineForce;
	float brake;
	float latFriction, longFriction;
	float suspensionLength;
	float extraPushback; // NOT reset on uncompressed contact - carries across ticks
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

// LinearPieceCurve mirror (std::map iterates sorted -> arrays in same order)
constexpr int GPU_CURVE_MAX = 12;
struct GpuCurve {
	float xs[GPU_CURVE_MAX];
	float ys[GPU_CURVE_MAX];
	int n;
};

struct GpuWheelConsts {
	GpuVec3 connectionPointCS;   // chassis connection (BT units, post addWheel)
	float suspensionRestLength;  // btWheelInfo m_suspensionRestLength1
	float wheelRadius;
	float maxSuspensionTravelCm;
	float suspensionStiffness;
	float dampingCompression, dampingRelaxation;
	float suspensionForceScale;
};

// Everything constant the car pre-tick phase needs, extracted host-side from the live
// arena/car/RLConst so no transcription can drift. Products that the CPU compiler
// constant-folds (parenthesized const subexpressions) are extracted pre-folded; chained
// multiplies stay raw and the device encodes the same trees.
struct GpuCarConsts {
	GpuWheelConsts wheels[4];
	GpuVec3 wheelDirectionCS, wheelAxleCS;
	float massBT;
	float frictionScale;          // chassis getMass()/3, host-evaluated
	float dodgeDeadzone;
	int rayCastGroup;             // DefaultFilter | vehicle raycaster addedFilterMask
	float erp;                    // solverInfo.m_erp (extraPushback)
	float uuToBT, btToUU;

	// mutator config
	float jumpAccel, jumpImmediateForce;
	float jumpImmediateForceRL; // RLConst::JUMP_IMMEDIATE_FORCE (double jump uses the CONST, not the mutator)
	float boostAccelGround, boostAccelAir, boostUsedPerSecond;
	uint8_t rechargeBoostEnabled;
	float rechargeBoostDelay, rechargeBoostPerSecond;
	uint8_t unlimitedFlips, unlimitedDoubleJumps;

	// RLConst scalars (raw)
	float powerslideRiseRate, powerslideFallRate;
	float throttleDeadzone, stoppingForwardVel, coastingBrakeFactor, brakingNoThrottleSpeedThresh;
	float throttleAirAccel;
	float jumpMinTime, jumpMaxTime, jumpResetTimePad, doublejumpMaxDelay;
	float flipZDamp120, flipZDampStart, flipZDampEnd, flipTorqueTime, flipPitchlockExtraTime;
	float flipInitialVelScale, flipTorqueX, flipTorqueY;
	float flipForwardImpulseMaxSpeedScale, flipSideImpulseMaxSpeedScale;
	float flipBackwardImpulseMaxSpeedScale, flipBackwardImpulseScaleX;
	float carTorqueScale;
	GpuVec3 carAirControlTorque, carAirControlDamping;
	float carAutoflipImpulse, carAutoflipTorque, carAutoflipTime;
	float carAutoflipNormZThresh, carAutoflipRollThresh;
	float carAutorollForce, carAutorollTorque;
	float boostMinTime, boostMax;
	float carMaxSpeed;
	float suspensionSubtraction; // RLConst::BTVehicle::SUSPENSION_SUBTRACTION

	// pre-folded constant subexpressions (host-evaluated with identical expressions)
	float throttleTorqueBT;       // THROTTLE_TORQUE_AMOUNT * UU_TO_BT
	float brakeTorqueBT;          // BRAKE_TORQUE_AMOUNT * UU_TO_BT
	float gravityZBT;             // GRAVITY_Z * UU_TO_BT
	float flipZDampFactor;        // powf(1 - FLIP_Z_DAMP_120, tickTimeScale=1)

	// curves (4-wheel variants; gates use octane)
	GpuCurve driveSpeedTorqueFactor;
	GpuCurve steerAngleFromSpeed;
	GpuCurve powerslideSteerAngleFromSpeed;
	GpuCurve nonStickyFrictionFactor;
	GpuCurve latFriction, longFriction;
	GpuCurve handbrakeLatFrictionFactor, handbrakeLongFrictionFactor;
};

// Dynamic ray-candidate body (ball sphere / car compound box) - shape descriptor plus
// its current world transform and filters; production kernels build these from
// GpuBodyState + shape consts, the ray-fuzz tool fills them host-side.
struct GpuDynBody {
	int kind;                 // 1 = sphere, 2 = box-compound (CU_SHAPE_* values)
	float sphereRadius;
	GpuVec3 boxHalfExtWithMargin;
	GpuVec3 childOffset;      // compound child transform origin (identity rotation)
	GpuMat3 basis;
	GpuVec3 origin;
	GpuVec3 aabbMin;          // broadphase proxy AABB min (drives voxel cell membership)
	int group, mask;
	int hasContactResponse;
	// dynamics (friction ground-body math; zero for ray-only uses)
	GpuVec3 linVel, angVel;
	GpuMat3 invInertiaWorld;
	GpuVec3 invInertiaDiagLocal;
	float invMass;
};

// Host<->device car pre-tick state bundle (compare harness + production phase I/O)
struct GpuCarTickState {
	GpuMat3 basis;
	GpuVec3 origin, linVel, angVel;
	GpuVec3 totalForce, totalTorque;
	GpuMat3 invInertiaWorld;
	GpuVec3 invInertiaDiagLocal;
	float invMass;
	GpuCarGameState st;
	GpuWheelState wheels[4];
	GpuCarControls controls;
	float solverTimeStep;
	float dbgForwardSpeedUU; // out: device-computed forwardSpeed_UU (compare diagnostics)
};

// Wheel intra-tick scratch results, downloadable for compare localization
struct GpuWheelScratchOut {
	GpuMat3 worldBasis;
	GpuVec3 hardPointWS, contactPointWS, contactNormalWS;
	uint8_t isInContact, isInContactWithWorld;
	int32_t groundIdx; // -2 none, -1 static, >=0 dyn index
	float suspensionRelativeVelocity, clippedInvContactDotSuspension, extraPushback;
	GpuVec3 impulse;
	float dbgWheelTraceLen, dbgPushThresh, dbgResolveRaw; // pushback-branch diagnostics
};

// btRSBroadphase voxel grid parameters. Dynamic proxies register in the 3x3x3 cells
// around their aabbMin's cell; short rays (len^2 < cellSizeSq) only test proxies in
// rayFrom's cell. Statics are smeared +-1 cell, so for them brute force == reference
// for any ray length; only dynamics need the cell gate.
struct GpuGridParams {
	GpuVec3 minPos;
	float cellSize;
	float cellSizeSq;
	int cellsX, cellsY, cellsZ;
};

// ---- Static world geometry (host-extracted from the arena's bullet world) ----

struct GpuPlane {
	GpuVec3 normal;     // shape normal (unit)
	float planeConst;   // shape plane constant (0 for RocketSim arenas)
	GpuVec3 origin;     // object world transform origin (basis = identity)
	int group, mask;    // broadphase proxy filters
};

struct GpuMeshInfo {
	int triStart;       // first triangle index in the tri vertex buffer
	int triCount;
	int group, mask;
};

// Quantized BVH mirror (btQuantizedBvhNode / btBvhSubtreeInfo layouts)
struct GpuBvhNode {
	uint16_t qmin[3], qmax[3];
	int32_t escapeIndexOrTriangleIndex; // >=0 leaf (triangleIndex), <0 internal (-escape)
};
struct GpuBvhSubtree {
	uint16_t qmin[3], qmax[3];
	int32_t rootNodeIndex, subtreeSize;
};
struct GpuMeshBvh {
	int nodeStart;      // into shared node buffer
	int subtreeStart;   // into shared subtree buffer
	int numSubtrees;
	GpuVec3 bvhAabbMin, bvhAabbMax, bvhQuantization;
};

// Per-triangle internal-edge info (btTriangleInfo; absent entries = zero angles)
struct GpuTriInfo {
	int flags;
	float edgeV0V1Angle, edgeV1V2Angle, edgeV2V0Angle;
	uint8_t hasInfo;
};

// ---- collision orchestration (btRSBroadphase + dispatcher replication) ----

// Persistent broadphase proxy state for one dyn body (ball, cars), per arena
struct GpuDynProxy {
	GpuVec3 aabbMin, aabbMax;
	int cellIdx;       // GetCellIdx(aabbMin), updated on every aabb change
	int i, j, k;       // registered 3x3x3 center (updated only on cell move w/ >1 dyn)
	int regSeq;        // cell re-registration sequence; cell dyn-list order = ascending
};

// Static collision object, broadphase handle order (constants; statics never move)
struct GpuStaticObj {
	int kind;                  // 0 = triangle mesh, 1 = static plane
	int shapeIdx;              // into the uploaded mesh / plane arrays
	// proxy aabb (pair-gen): shape aabb +-0.02 after the first updateAabbs
	GpuVec3 aabbMin, aabbMax;
	// raw shape->getAabb (compound ProcessChildShape tests; NO expansion)
	GpuVec3 shapeAabbMin, shapeAabbMax;
	int group, mask;
	float friction, restitution;  // rigid body material (combine inputs)
	float breakingThreshold;      // shape->getContactBreakingThreshold(0.02)
};

// Manifold download mirror (device CuManifold/CuManifoldPoint)
struct GpuContactOut {
	GpuVec3 localPointA, localPointB;
	GpuVec3 positionWorldOnA, positionWorldOnB, normalWorldOnB;
	GpuVec3 lateralFrictionDir1, lateralFrictionDir2;
	float distance, combinedFriction, combinedRestitution;
	int partId0, partId1, index0, index1;
	uint8_t isSpecial;
};
struct GpuManifoldOut {
	int bodyA, bodyB;          // >= 0: dyn slot (0 = ball); < 0: static handle -1-v
	int numContacts;
	GpuContactOut pts[4];
};
// Capacity == pair capacity (<= 1 manifold per pair, so overflow is impossible)
#define GPU_MAX_MANIFOLDS 96

// Ball game state touched by collision callbacks + pre-tick + finish
struct GpuBallGameState {
	GpuVec3 velocityImpulseCache;  // Ball::_velocityImpulseCache (BT units)
	uint64_t tickCountSinceUpdate; // BallState::tickCountSinceUpdate (finish increments)
	// HeatseekerInfo
	float hsYTargetDir;            // 0 = no net
	float hsCurTargetSpeed;        // UU
	float hsTimeSinceHit;
};

// Boost pad constants (pad order = arena _boostPads order, identical all arenas)
struct GpuPadConfig {
	GpuVec3 posBT;             // BoostPad::_posBT
	GpuVec3 boxMinBT, boxMaxBT;
	float cylRadBT;            // (CYL_RAD_BIG/SMALL) * UU_TO_BT
	uint8_t isBig;
};

// Game-tail constants (Arena::Step phases outside bullet; shared by all arenas)
struct GpuTailConsts {
	float supersonicStartSpeed, supersonicMaintainMinSpeed, supersonicMaintainMaxTime;
	float carMaxSpeedBT, carMaxAngSpeed;
	float ballMaxSpeedBT, ballMaxAngSpeed;
	float boostMax, boostAmountBig, boostAmountSmall;
	float padCooldownBig, padCooldownSmall;
	// BoostPadGrid (soccar: 8x10 cells)
	float padGridExtentZ;
	int padGridCellSizeX, padGridCellSizeY; // int consts (float division by int on CPU)
	int padGridCellsX, padGridCellsY;
	int16_t padGridMap[8 * 10];             // [ix * cellsY + iy] -> pad index or -1
	float cylHeightBT;                       // CYL_HEIGHT * UU_TO_BT
};


// Constants for the collide phase (shared by all arenas)
struct GpuCollideConsts {
	// btCompoundShape::getAabb inputs (m_localAabb includes the child box margin)
	GpuVec3 carLocalAabbMin, carLocalAabbMax;
	float carCompoundMargin;
	float ballRadius;
	GpuVec3 carBoxHalfExtNoMargin;
	float carBoxMargin;
	GpuVec3 carChildOffset;            // compound child transform origin (basis = I)
	float ballBreakingThreshold, carBreakingThreshold; // getAngularMotionDisc()*0.02
	float ballFriction, ballRestitution, carFriction, carRestitution;
	int ballFilterGroup, ballFilterMask, carFilterGroup, carFilterMask;
	int numStatics;
	int gameMode;                      // RocketSim GameMode (0 = soccar)
	int demoMode;                      // MutatorConfig (gates: DISABLED)
	uint8_t enableTeamDemos;
	float respawnDelay, bumpCooldownTime, bumpForceScale, ballHitExtraForceScale;
	float carWorldFriction, carWorldRestitution;
	float carBallFriction, carBallRestitution;   // CARBALL_COLLISION_*
	float carCarFriction, carCarRestitution;     // CARCAR_COLLISION_*
	float bumpMinForwardDist;                    // BUMP_MIN_FORWARD_DIST (UU)
	float ballCarExtraZScale, ballCarExtraForwardScale, ballCarExtraMaxDeltaVelUU;
	GpuCurve ballExtraImpulseFactorCurve;
	GpuCurve bumpVelGroundCurve, bumpVelAirCurve, bumpUpwardVelCurve;
	// Heatseeker (RLConst::Heatseeker + arena extent; the contact callback and the
	// ball pre-tick both read these)
	float hsTargetY, hsTargetZ;             // goal target point (UU)
	float hsHorizontalBlend, hsVerticalBlend, hsSpeedBlend;
	float hsMaxTurnPitch;
	float hsMaxSpeed;                       // seek speed cap (4600)
	float hsTargetSpeedIncrement, hsMinSpeedupInterval;
	float hsWallBounceChangeYThresh, hsWallBounceChangeYNormal;
	float hsWallBounceForceScale, hsWallBounceUpFrac;
	float arenaExtentY;
};

} // namespace rscuda
