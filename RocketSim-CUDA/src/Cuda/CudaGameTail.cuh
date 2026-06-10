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

// Ball::_FinishPhysicsTick (ballState tick counter lives in GpuBallGameState? -
// only the impulse cache is modeled; tickCountSinceUpdate not consumed by sim)
__device__ __forceinline__ void CuBallFinish(GpuBodyState& b, GpuBallGameState& st,
	const GpuTailConsts& tc) {
	if (!CuVecIsZero(st.velocityImpulseCache)) {
		CuVec3 v = GV(b.linVel) + GV(st.velocityImpulseCache);
		b.linVel = { v.x, v.y, v.z };
		st.velocityImpulseCache = { 0, 0, 0 };
	}
	CuClampVelocities(b, tc.ballMaxSpeedBT, tc.ballMaxAngSpeed);
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
