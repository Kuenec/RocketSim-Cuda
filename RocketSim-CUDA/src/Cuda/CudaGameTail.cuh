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

// Car::_PostTickUpdate (rotMat cache equals the live basis - the callbacks read
// columns of the live basis already; demoed cars would keep a stale rotMat, but
// demolitions are not modeled)
__device__ __forceinline__ void CuCarPostTick(const GpuBodyState& b, GpuCarGameState& st,
	const GpuTailConsts& tc, const GpuCarControls& controls, float tickTime) {
	if (st.isDemoed)
		return;
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

// BoostPadGrid::CheckCollision(car) + BoostPad::_CheckCollide (soccar grid)
__device__ inline void CuPadGridCheck(const GpuBodyState& carBody,
	const GpuCarGameState& carSt, int carIdx,
	GpuPadState* pads, const GpuPadConfig* padConfigs,
	const GpuTailConsts& tc, const GpuCollideConsts& cc) {
	if (carSt.isDemoed || carSt.boost >= tc.boostMax)
		return;
	CuVec3 carPosBT = GV(carBody.pos);
	CuVec3 carPosUU = carPosBT * 50.0f; // * BT_TO_UU (Vec scalar mul)
	if (carPosUU.z > tc.padGridExtentZ)
		return;
	int indexX = (int)(carPosUU.x / (float)tc.padGridCellSizeX + (float)(tc.padGridCellsX / 2));
	int indexY = (int)(carPosUU.y / (float)tc.padGridCellSizeY + (float)(tc.padGridCellsY / 2));
	int i0 = indexX - 1 > 0 ? indexX - 1 : 0;
	int i1 = indexX + 1 < tc.padGridCellsX - 1 ? indexX + 1 : tc.padGridCellsX - 1;
	int j0 = indexY - 1 > 0 ? indexY - 1 : 0;
	int j1 = indexY + 1 < tc.padGridCellsY - 1 ? indexY + 1 : tc.padGridCellsY - 1;
	for (int i = i0; i <= i1; i++) {
		for (int j = j0; j <= j1; j++) {
			int padIdx = tc.padGridMap[i * tc.padGridCellsY + j];
			if (padIdx < 0)
				continue;
			GpuPadState& p = pads[padIdx];
			const GpuPadConfig& pc = padConfigs[padIdx];
			bool colliding = false;
			if (p.prevLockedCarID == carSt.carID) {
				// car->_rigidBody.getAabb = compound shape aabb at current transform
				CuVec3 mn, mx;
				CuCompoundAabbAt(cc, CuLoadBasis(carBody.basis), carPosBT, mn, mx);
				CuVec3 bmin = GV(pc.boxMinBT), bmax = GV(pc.boxMaxBT);
				// Vec operator> / operator<: ALL components
				colliding = (bmax.x > mn.x && bmax.y > mn.y && bmax.z > mn.z) &&
					(bmin.x < mx.x && bmin.y < mx.y && bmin.z < mx.z);
			} else {
				// Vec::DistSq2D
				CuVec3 pos = GV(pc.posBT);
				float dx = carPosBT.x - pos.x;
				float dy = carPosBT.y - pos.y;
				if (dx * dx + dy * dy < pc.cylRadBT * pc.cylRadBT)
					colliding = fabsf(carPosBT.z - pos.z) < tc.cylHeightBT;
			}
			if (colliding)
				p.lockedCarIdx = carIdx;
		}
	}
}

// BoostPad::_PostTickUpdate
__device__ __forceinline__ void CuPadPostTick(GpuPadState& p, const GpuPadConfig& pc,
	GpuCarGameState* cars, const GpuTailConsts& tc) {
	uint32_t lockedCarID = 0;
	if (p.lockedCarIdx >= 0) {
		GpuCarGameState& car = cars[p.lockedCarIdx];
		lockedCarID = car.carID;
		if (p.isActive) {
			float boostToAdd = pc.isBig ? tc.boostAmountBig : tc.boostAmountSmall;
			float nb = car.boost + boostToAdd;
			car.boost = nb < tc.boostMax ? nb : tc.boostMax; // RS_MIN
			p.isActive = 0;
			p.cooldown = pc.isBig ? tc.padCooldownBig : tc.padCooldownSmall;
		}
	}
	p.prevLockedCarID = lockedCarID;
}

} // namespace rscuda
