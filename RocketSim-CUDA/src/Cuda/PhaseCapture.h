#pragma once
#include <cstdint>
#include <vector>

class btDiscreteDynamicsWorld;

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

} // namespace rscap
