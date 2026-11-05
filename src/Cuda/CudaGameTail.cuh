#pragma once
// 2026-06-10: Device ports of the Arena::Step phases OUTSIDE the bullet step:
// ball zero-vel sleep check, applyGravity/clearForces (stepSimulation wrappers),
// Car::_PostTickUpdate + _FinishPhysicsTick, BoostPadGrid::CheckCollision +
// BoostPad::_PreTickUpdate/_PostTickUpdate, Ball::_FinishPhysicsTick.
// Soccar scope (gates): dropshot tile logic / goal-score callback (null in the
// bench) / demolition respawns (DemoMode::DISABLED) are not modeled.

#include "GpuTypes.h"
#include "CudaMath.cuh"
#include "CudaCollideWorld.cuh" // activation constants + CuSetActivationState

namespace rscuda {

// Arena::Step head: ball zero-vel sleeping (exact ==0 compares; btVector3::length2)
__device__ __forceinline__ void CuBallSleepCheck(GpuBodyState& ball) {
	float lv = CuLength2(GV(ball.linVel));
	float av = CuLength2(GV(ball.angVel));
	if (lv == 0.0f && av == 0.0f)
		CuSetActivationState(ball.activationState, CU_ISLAND_SLEEPING);
	else
		CuSetActivationState(ball.activationState, CU_ACTIVE_TAG);
}

// btDiscreteDynamicsWorld::applyGravity: active bodies only; m_totalForce += m_gravity
__device__ __forceinline__ void CuApplyGravity(GpuBodyState& b, const GpuBodyConsts& c) {
	if (!CuIsActive(b.activationState))
		return;
	CuVec3 f = GV(b.totalForce) + GV(c.gravityForce);
	b.totalForce = { f.x, f.y, f.z };
}

// btDiscreteDynamicsWorld::clearForces (unconditional)
__device__ __forceinline__ void CuClearForces(GpuBodyState& b) {
	b.totalForce = { 0, 0, 0 };
	b.totalTorque = { 0, 0, 0 };
}

// Car::_PostTickUpdate. The rotMat cache is refreshed from the live basis here,
// which means it goes STALE while demoed (the early return skips it) - that stale
// value is the CPU's hash-visible CarState::rotMat, so it is carried in st.
__device__ __forceinline__ void CuCarPostTick(const GpuBodyState& b, GpuCarGameState& st,
	const GpuTailConsts& tc, const GpuCarControls& controls, float tickTime) {
	if (st.isDemoed)
		return;
	st.rotMat = b.basis;
	{ // supersonic
		CuVec3 velUU = GV(b.linVel) * 50.0f; // * BT_TO_UU
		float speedSquared = CuLength2(velUU);
		if (st.isSupersonic && st.supersonicTime < tc.supersonicMaintainMaxTime) {
			st.isSupersonic =
				speedSquared >= tc.supersonicMaintainMinSpeed * tc.supersonicMaintainMinSpeed;
		} else {
			st.isSupersonic =
				speedSquared >= tc.supersonicStartSpeed * tc.supersonicStartSpeed;
		}
		if (st.isSupersonic)
			st.supersonicTime += tickTime;
		else
			st.supersonicTime = 0;
	}
	if (st.carContact_cooldownTimer > 0) {
		float t = st.carContact_cooldownTimer - tickTime;
		st.carContact_cooldownTimer = t > 0.0f ? t : 0.0f; // RS_MAX(t, 0)
	}
	st.lastControls = controls;
}

// shared velocity-clamp tail (Car/Ball::_FinishPhysicsTick): normalized() is the
// RSQRT normalize path
__device__ __forceinline__ void CuClampVelocities(GpuBodyState& b,
	float maxSpeedBT, float maxAngSpeed) {
	CuVec3 vel = GV(b.linVel);
	CuVec3 angVel = GV(b.angVel);
	if (CuLength2(vel) > maxSpeedBT * maxSpeedBT)
		vel = CuNormalize3(vel) * maxSpeedBT;
	if (CuLength2(angVel) > maxAngSpeed * maxAngSpeed)
		angVel = CuNormalize3(angVel) * maxAngSpeed;
	b.linVel = { vel.x, vel.y, vel.z };
	b.angVel = { angVel.x, angVel.y, angVel.z };
}

__device__ __forceinline__ bool CuVecIsZero(GpuVec3 v) {
	return v.x == 0.0f && v.y == 0.0f && v.z == 0.0f;
}

// Car::_FinishPhysicsTick
__device__ __forceinline__ void CuCarFinish(GpuBodyState& b, GpuCarGameState& st,
	const GpuTailConsts& tc) {
	if (st.isDemoed)
		return;
	if (!CuVecIsZero(st.velocityImpulseCache)) {
		CuVec3 v = GV(b.linVel) + GV(st.velocityImpulseCache);
		b.linVel = { v.x, v.y, v.z };
		st.velocityImpulseCache = { 0, 0, 0 };
	}
	CuClampVelocities(b, tc.carMaxSpeedBT, tc.carMaxAngSpeed);
	st.tickCountSinceUpdate++;
}

// Ball::_FinishPhysicsTick
__device__ __forceinline__ void CuBallFinish(GpuBodyState& b, GpuBallGameState& st,
	const GpuTailConsts& tc) {
	if (!CuVecIsZero(st.velocityImpulseCache)) {
		CuVec3 v = GV(b.linVel) + GV(st.velocityImpulseCache);
		b.linVel = { v.x, v.y, v.z };
		st.velocityImpulseCache = { 0, 0, 0 };
	}
	CuClampVelocities(b, tc.ballMaxSpeedBT, tc.ballMaxAngSpeed);
	st.tickCountSinceUpdate++;
}

// Ball::_PreTickUpdate hoops/dropshot branch: launch the kickoff-frozen ball with an
// upward impulse once tickCountSinceUpdate crosses the launch delay
__device__ __forceinline__ void CuBallPreTickLaunch(GpuBodyState& b, GpuBallGameState& st,
	const GpuCollideConsts& cc, const GpuBodyConsts& bc, float tickTime) {
	float curKickoffTime = (float)st.tickCountSinceUpdate * tickTime;
	float prevKickoffTime = curKickoffTime - tickTime;
	if (prevKickoffTime < cc.ballLaunchDelay && curKickoffTime >= cc.ballLaunchDelay) {
		// GetState(): vel/pos in UU, angVel raw
		CuVec3 velUU = GV(b.linVel) * CU_BT_TO_UU;
		CuVec3 angVel = GV(b.angVel);
		CuVec3 posUU = GV(b.pos) * CU_BT_TO_UU;
		if (velUU.x == 0.0f && velUU.y == 0.0f && velUU.z == 0.0f &&
			angVel.x == 0.0f && angVel.y == 0.0f && angVel.z == 0.0f &&
			posUU.x == 0.0f && posUU.y == 0.0f) {
			// applyCentralImpulse(Vec(0,0,launchVelZ) * GetMass() * UU_TO_BT):
			// m_linearVelocity += impulse * m_linearFactor(1,1,1) * m_inverseMass
			float impZ = (cc.ballLaunchZVel * cc.ballMassBT) * CU_UU_TO_BT;
			CuVec3 v = GV(b.linVel);
			v.x = v.x + (0.0f * 1.0f) * bc.invMass;
			v.y = v.y + (0.0f * 1.0f) * bc.invMass;
			v.z = v.z + (impZ * 1.0f) * bc.invMass;
			b.linVel = { v.x, v.y, v.z };
			CuSetActivationState(b.activationState, CU_ACTIVE_TAG);
		}
	}
}

// ---- demolition respawn (Car::Respawn + Car::SetState) -----------------------------

// libstdc++ std::default_random_engine = minstd_rand0: x = x * 16807 % 2147483647
__device__ __forceinline__ uint32_t CuMinstdNext(uint32_t& s) {
	s = (uint32_t)(((uint64_t)s * 16807ull) % 2147483647ull);
	return s;
}

// Car::Respawn(gameMode, seed=-1, boost) followed by Car::SetState(CarState reset).
// Wheels (btVehicleRL persists) are NOT reset - CPU quirk.
__device__ inline void CuCarRespawn(GpuBodyState& body, GpuCarGameState& st,
	uint32_t& rngState, const GpuBodyConsts& bc, const GpuTailConsts& tc) {
	// Math::RandInt(0, 4, -1) = engine() % 4
	int spawnPosIndex = (int)(CuMinstdNext(rngState) % 4u);
	float ySign = st.team == 0 /*BLUE*/ ? 1.0f : -1.0f;
	CuVec3 posUU = { tc.respawnX[spawnPosIndex],
		tc.respawnY[spawnPosIndex] * ySign, tc.carRespawnZ };
	// Angle(yawAng + (blue ? 0 : M_PI), 0, 0): the M_PI add promotes to double
	float yaw = st.team == 0 ? tc.respawnYaw[spawnPosIndex]
		: (float)((double)tc.respawnYaw[spawnPosIndex] + 0x1.921fb54442d18p+1);
	// ToRotMat = setEulerYPR(yaw, -0.f, -0.f) = setEulerZYX(eulerX=-0, eulerY=-0,
	// eulerZ=yaw); signed zeros flow through the products
	float ci = CuCosf(-0.0f), cj = CuCosf(-0.0f), ch = CuCosf(yaw);
	float si = CuSinf(-0.0f), sj = CuSinf(-0.0f), sh = CuSinf(yaw);
	float cc = ci * ch, cs = ci * sh, sc = si * ch, ss = si * sh;
	CuMat3 basis = { { { cj * ch, sj * sc - cs, sj * cc + ss },
	                   { cj * sh, sj * ss + cc, sj * cs - sc },
	                   { -sj, cj * si, cj * ci } } };
	// Car::SetState
	CuVec3 posBT = posUU * (1.0f / 50.0f); // * UU_TO_BT
	body.pos = { posBT.x, posBT.y, posBT.z };
	for (int r = 0; r < 3; r++)
		body.basis.row[r] = { basis.r[r].x, basis.r[r].y, basis.r[r].z };
	GpuMat3 rotMat = body.basis; // _internalState.rotMat = state.rotMat
	body.linVel = { 0, 0, 0 };
	body.angVel = { 0, 0, 0 };
	CuMat3 invIW = CuUpdateInertiaTensor(basis, GV(bc.invInertiaLocal));
	for (int r = 0; r < 3; r++)
		body.invInertiaWorld.row[r] = { invIW.r[r].x, invIW.r[r].y, invIW.r[r].z };
	// _internalState = CarState() defaults (keep carID/team/wheels)
	uint32_t carID = st.carID;
	uint8_t team = st.team;
	GpuWheelState wheels[4];
	for (int w = 0; w < 4; w++)
		wheels[w] = st.wheels[w];
	st = {};
	st.carID = carID;
	st.team = team;
	for (int w = 0; w < 4; w++)
		st.wheels[w] = wheels[w];
	st.rotMat = rotMat;
	st.isOnGround = 1;
	st.boost = tc.carSpawnBoostAmount;
	st.ballHitInfo.tickCountWhenHit = ~0ull;                  // BallHitInfo defaults
	st.ballHitInfo.tickCountWhenExtraImpulseApplied = ~0ull;
	// (velocityImpulseCache / timers / flags all zero from the {} init)
}

// ---- RocketSim Angle math (heatseeker ball steering) -------------------------------

struct CuAngle { float yaw, pitch, roll; };

// Math::WrapNormalizeFloat: C fmod promotes float args to DOUBLE (exact for the
// remainder of two floats), result truncated back via the float assignment
__device__ __forceinline__ float CuWrapNormalizeFloat(float val, float minmax) {
	float m2 = minmax * 2;
	float result = (float)fmod((double)val, (double)m2);
	if (result > minmax)
		result -= m2;
	else if (result < -minmax)
		result += m2;
	return result;
}

__device__ __forceinline__ void CuAngleNormalizeFix(CuAngle& a) {
	a.yaw = CuWrapNormalizeFloat(a.yaw, (float)(0x1.921fb54442d18p+1));      // M_PI
	a.pitch = CuWrapNormalizeFloat(a.pitch, (float)(0x1.921fb54442d18p+0)); // M_PI/2
	a.roll = CuWrapNormalizeFloat(a.roll, (float)(0x1.921fb54442d18p+1));
}

// Angle::FromVec (atan2f = the glibc-exact CuAtan2f)
__device__ __forceinline__ CuAngle CuAngleFromVec(CuVec3 forward) {
	float yaw, pitch;
	if (fabsf(forward.y) > CU_SIMD_EPSILON || fabsf(forward.x) > CU_SIMD_EPSILON) {
		yaw = CuAtan2f(forward.y, forward.x);
		float dist2D = sqrtf(forward.x * forward.x + forward.y * forward.y);
		pitch = -CuAtan2f(-forward.z, dist2D);
	} else {
		yaw = 0;
		if (forward.z > CU_SIMD_EPSILON)
			pitch = (float)(0x1.921fb54442d18p+0);  // M_PI / 2
		else if (forward.z < -CU_SIMD_EPSILON)
			pitch = (float)(-0x1.921fb54442d18p+0);
		else
			pitch = 0;
	}
	return { yaw, pitch, 0 };
}

// Math::RoundAngleUE3 (UE3 rotator quantization; roll asserted 0)
__device__ __forceinline__ CuAngle CuRoundAngleUE3(CuAngle ang) {
	const float TO_INTS = (float)((double)(1 << 15) / 0x1.921fb54442d18p+1);
	const float BACK_TO_RADIANS = (1.0f / TO_INTS) * 4.0f;
	const int ROUNDING_MASK = 0x4000 - 1;
	int rYaw = ((int)(ang.yaw * TO_INTS) >> 2) & ROUNDING_MASK;
	int rPitch = ((int)(ang.pitch * TO_INTS) >> 2) & ROUNDING_MASK;
	ang.yaw = rYaw * BACK_TO_RADIANS;
	ang.pitch = rPitch * BACK_TO_RADIANS;
	return ang;
}

// Angle::GetForwardVec (full-range glibc-exact CuSinf/CuCosf)
__device__ __forceinline__ CuVec3 CuAngleForwardVec(const CuAngle& a) {
	float cp = CuCosf(-a.pitch);
	float cy = CuCosf(a.yaw);
	float sy = CuSinf(a.yaw);
	float sp = CuSinf(-a.pitch);
	return { cp * cy, cp * sy, -sp };
}

// Ball::_PreTickUpdate, HEATSEEKER branch (runs after pad pre-ticks)
__device__ inline void CuBallPreTickHS(GpuBodyState& ball, GpuBallGameState& bs,
	const GpuCollideConsts& cc, float tickTime) {
	if (bs.hsYTargetDir == 0.0f)
		return;
	// GetState(): live conversions to UU (RocketSim Vec math from here)
	CuVec3 pos = GV(ball.pos) * 50.0f;     // BT_TO_UU
	CuVec3 vel = GV(ball.linVel) * 50.0f;
	float yTargetDir = bs.hsYTargetDir;

	CuAngle velAngle = CuAngleFromVec(vel);
	CuVec3 goalTargetPos = { 0, cc.hsTargetY * yTargetDir, cc.hsTargetZ };
	CuAngle angleToGoal = CuAngleFromVec(goalTargetPos - pos);
	// angleToGoal - velAngle = velAngle.GetDeltaTo(angleToGoal)
	CuAngle deltaAngle = { angleToGoal.yaw - velAngle.yaw,
		angleToGoal.pitch - velAngle.pitch, angleToGoal.roll - velAngle.roll };
	CuAngleNormalizeFix(deltaAngle);

	// Vec::Length
	float curSpeedSq = vel.x * vel.x + vel.y * vel.y + vel.z * vel.z;
	float curSpeed = curSpeedSq > 0 ? sqrtf(curSpeedSq) : 0.0f;
	float speedRatio = curSpeed / cc.hsMaxSpeed;

	CuAngle newAngle = velAngle;
	float baseInterpFactor = speedRatio * tickTime;
	newAngle.yaw += deltaAngle.yaw * baseInterpFactor * cc.hsHorizontalBlend;
	newAngle.pitch += deltaAngle.pitch * baseInterpFactor * cc.hsVerticalBlend;
	CuAngleNormalizeFix(newAngle);
	// RS_CLAMP(v, lo, hi) = RS_MAX(RS_MIN(v, hi), lo)
	float pmin = -cc.hsMaxTurnPitch, pmax = cc.hsMaxTurnPitch;
	float pcl = newAngle.pitch < pmax ? newAngle.pitch : pmax;
	newAngle.pitch = pcl > pmin ? pcl : pmin;
	newAngle = CuRoundAngleUE3(newAngle);

	float newSpeed = curSpeed + ((bs.hsCurTargetSpeed - curSpeed) * cc.hsSpeedBlend);
	CuVec3 newDir = CuAngleForwardVec(newAngle);
	CuVec3 newVel = newDir * newSpeed;
	CuVec3 lv = newVel * (1.0f / 50.0f); // * UU_TO_BT
	ball.linVel = { lv.x, lv.y, lv.z };
	bs.hsTimeSinceHit += tickTime;
}

// BoostPad::_PreTickUpdate
__device__ __forceinline__ void CuPadPreTick(GpuPadState& p, float tickTime) {
	if (p.cooldown > 0) {
		float t = p.cooldown - tickTime;
		p.cooldown = t > 0.0f ? t : 0.0f;
	}
	p.isActive = (p.cooldown == 0.0f);
	p.lockedCarIdx = -1;
}

struct CuPadGridCar {
	CuVec3 posBT;
	CuVec3 aabbMin, aabbMax;
	uint32_t carID;
	int i0, i1, j0, j1;
};

__device__ __forceinline__ void CuPadGridCarStage(const GpuBodyState& carBody,
	const GpuCarGameState& carSt, const GpuTailConsts& tc, const GpuCollideConsts& cc,
	CuPadGridCar& g) {
	g.carID = carSt.carID;
	g.i0 = 1;
	g.i1 = 0;
	g.j0 = 1;
	g.j1 = 0;
	if (carSt.isDemoed || carSt.boost >= tc.boostMax)
		return;
	CuVec3 carPosBT = GV(carBody.pos);
	CuVec3 carPosUU = carPosBT * 50.0f;
	if (carPosUU.z > tc.padGridExtentZ)
		return;
	int indexX = (int)(carPosUU.x / (float)tc.padGridCellSizeX + (float)(tc.padGridCellsX / 2));
	int indexY = (int)(carPosUU.y / (float)tc.padGridCellSizeY + (float)(tc.padGridCellsY / 2));
	g.i0 = indexX - 1 > 0 ? indexX - 1 : 0;
	g.i1 = indexX + 1 < tc.padGridCellsX - 1 ? indexX + 1 : tc.padGridCellsX - 1;
	g.j0 = indexY - 1 > 0 ? indexY - 1 : 0;
	g.j1 = indexY + 1 < tc.padGridCellsY - 1 ? indexY + 1 : tc.padGridCellsY - 1;
	g.posBT = carPosBT;
	CuCompoundAabbAt(cc, CuLoadBasis(carBody.basis), carPosBT, g.aabbMin, g.aabbMax);
}

__device__ __forceinline__ bool CuPadCollidesCar(const GpuPadState& p, const GpuPadConfig& pc,
	const CuPadGridCar& g, const GpuTailConsts& tc) {
	if (pc.cellX < g.i0 || pc.cellX > g.i1 || pc.cellY < g.j0 || pc.cellY > g.j1)
		return false;
	if (p.prevLockedCarID == g.carID) {
		CuVec3 bmin = GV(pc.boxMinBT), bmax = GV(pc.boxMaxBT);
		return (bmax.x > g.aabbMin.x && bmax.y > g.aabbMin.y && bmax.z > g.aabbMin.z) &&
			(bmin.x < g.aabbMax.x && bmin.y < g.aabbMax.y && bmin.z < g.aabbMax.z);
	}
	CuVec3 pos = GV(pc.posBT);
	float dx = g.posBT.x - pos.x;
	float dy = g.posBT.y - pos.y;
	if (dx * dx + dy * dy < pc.cylRadBT * pc.cylRadBT)
		return fabsf(g.posBT.z - pos.z) < tc.cylHeightBT;
	return false;
}

// BoostPad::_PostTickUpdate, split for the parallel kernel (2026-06-11):
// the per-pad state advance touches only its own pad (cars are only READ for
// carID), so all pads can run concurrently; the boost grant writes car.boost,
// whose boostMax clamp is order-sensitive when several pads lock one car, so
// grants are applied afterwards in pad order. The grant DECISION (locked +
// active) does not depend on car.boost, so the split is exact.

// phase 1, parallel-safe per pad: returns the car index to grant boost to (-1 none)
__device__ __forceinline__ int CuPadPostTickPad(GpuPadState& p, const GpuPadConfig& pc,
	const GpuCarGameState* cars, const GpuTailConsts& tc) {
	uint32_t lockedCarID = 0;
	int grantTo = -1;
	if (p.lockedCarIdx >= 0) {
		lockedCarID = cars[p.lockedCarIdx].carID;
		if (p.isActive) {
			grantTo = p.lockedCarIdx;
			p.isActive = 0;
			p.cooldown = pc.isBig ? tc.padCooldownBig : tc.padCooldownSmall;
		}
	}
	p.prevLockedCarID = lockedCarID;
	return grantTo;
}

// phase 2, sequential in pad order: the boost grant with the boostMax clamp
__device__ __forceinline__ void CuPadGrantBoost(const GpuPadConfig& pc,
	GpuCarGameState& car, const GpuTailConsts& tc) {
	float boostToAdd = pc.isBig ? tc.boostAmountBig : tc.boostAmountSmall;
	float nb = car.boost + boostToAdd;
	car.boost = nb < tc.boostMax ? nb : tc.boostMax; // RS_MIN
}

__device__ __forceinline__ void CuPadPostTick(GpuPadState& p, const GpuPadConfig& pc,
	GpuCarGameState* cars, const GpuTailConsts& tc) {
	int grantTo = CuPadPostTickPad(p, pc, cars, tc);
	if (grantTo >= 0)
		CuPadGrantBoost(pc, cars[grantTo], tc);
}

} // namespace rscuda
