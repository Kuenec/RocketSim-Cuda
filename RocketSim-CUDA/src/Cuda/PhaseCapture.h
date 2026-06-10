#pragma once
#include <cstdint>
#include <vector>

class btDiscreteDynamicsWorld;
namespace RocketSim { class Car; }

// 2026-06-10: CPU-side phase-boundary state capture (work copy only, diagnostic).
// Used by rocketsim_bench --gpu-compare to validate each ported GPU phase bit-exactly:
// the bench uploads a captured CPU state, runs one GPU phase, and compares against the
// next capture point. Zero overhead unless enabled.
namespace rscap {

enum Point : int {
	PRE_BULLET,       // entry of internalSingleStepSimulation (forces incl. gravity applied)
	AFTER_PREDICT,
	AFTER_COLLISION,
	AFTER_ISLANDS,
	AFTER_SOLVE,
	AFTER_INTEGRATE,
	AFTER_ACTIVATION,
	POINT_COUNT
};
extern const char* const POINT_NAMES[POINT_COUNT];

struct BodySnap {
	const void* rb; // btRigidBody* identity for index mapping
	float basis[9], pos[3];
	float linVel[3], angVel[3];
	float totalForce[3], totalTorque[3];
	float invInertiaWorld[9];
	int activationState;
	float deactivationTime;
};

extern bool enabled;
extern std::vector<BodySnap> points[POINT_COUNT]; // refreshed every captured tick

void Capture(Point p, btDiscreteDynamicsWorld* world);

// ---- collision-phase capture (at AFTER_COLLISION) ----
// Dispatcher manifolds in creation order + broadphase proxy state of each nonstatic
// body. This is the bit-exact validation target for the GPU collision orchestrator.

struct ContactSnap {
	float localPointA[3], localPointB[3];
	float positionWorldOnA[3], positionWorldOnB[3];
	float normalWorldOnB[3];
	float distance, combinedFriction, combinedRestitution;
	int partId0, index0, partId1, index1;
	uint8_t isSpecial;
};
struct ManifoldSnap {
	const void* body0, *body1;   // btCollisionObject* identity (may be a static)
	int numContacts;
	ContactSnap pts[4];
};
struct ProxySnap {
	const void* rb;
	float aabbMin[3], aabbMax[3];
	int cellIdx, iIdx, jIdx, kIdx;
};
extern std::vector<ManifoldSnap> manifolds;  // refreshed at AFTER_COLLISION
extern std::vector<ProxySnap> proxies;       // refreshed at AFTER_COLLISION

// ---- per-car pre-tick capture (hooks in Car::_PreTickUpdate) ----
// One entry per car per tick, in CPU execution order (unordered_set iteration), so the
// GPU compare can replay each car's pre-tick from its exact input state - including
// the post-pre-tick state of cars processed earlier in the same tick.

struct WheelSnap {
	float steerAngle, engineForce, brake, latFriction, longFriction, suspensionLength;
	// intra-tick scratch after updateVehicleFirst:
	float worldBasis[9];
	float hardPointWS[3], contactPointWS[3], contactNormalWS[3];
	uint8_t isInContact, isInContactWithWorld;
	int groundIsStatic;     // 1 static, 0 dynamic, -1 none
	const void* groundRb;   // identity of ground object (for dyn mapping)
	float suspensionRelativeVelocity, clippedInvContactDotSuspension, extraPushback;
	float impulse[3];
};

// Dynamic-body view at capture time (ray candidates + friction ground bodies)
struct DynSnap {
	const void* rb;
	float basis[9], origin[3], linVel[3], angVel[3];
	float invInertiaWorld[9], invInertiaDiagLocal[3];
	float invMass;
	float aabbMin[3];       // broadphase proxy (voxel cell gate)
	uint8_t hasContactResponse;
};

struct CarPreTickSnap {
	uint32_t carId;
	int stage;              // 0 = entry, 1 = after updateVehicleFirst, 2 = exit
	BodySnap body;          // own chassis (forces included)
	DynSnap dyn[8];         // all nonstatic bodies at this moment (entry stage)
	int numDyn;
	WheelSnap wheels[4];
	// game-state fields (CarState scalars the phase touches)
	float handbrakeVal, boost, boostingTime, timeSinceBoosted, jumpTime, flipTime;
	float airTime, airTimeSinceJump, autoFlipTimer, autoFlipTorqueScale;
	float flipRelTorque[3], worldContactNormal[3];
	uint8_t isOnGround, wheelsWithContact[4], hasJumped, hasDoubleJumped, hasFlipped;
	uint8_t isFlipping, isJumping, isBoosting, isAutoFlipping, worldContactHasContact;
	uint8_t isDemoed;
	float solverTimeStep;   // solverInfo.m_timeStep at this moment
	// inputs (entry stage): raw controls + last tick's controls (jumpPressed edge)
	float ctrlThrottle, ctrlSteer, ctrlPitch, ctrlYaw, ctrlRoll;
	uint8_t ctrlJump, ctrlBoost, ctrlHandbrake;
	uint8_t lastCtrlJump;
	float forwardSpeedUU; // getForwardSpeed()*BT_TO_UU as the CPU computes it
};

extern std::vector<CarPreTickSnap> carSnaps; // cleared at PRE car loop, appended per car/stage

void CaptureCarPreTick(RocketSim::Car* car, int stage, float solverTimeStep);

} // namespace rscap
