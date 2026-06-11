// 2026-06-10: GpuArenaBridge implementation. The static helpers below are the
// bench driver's validated extraction/snapshot/writeback code (tools/bench_work/
// main.cpp), generalized from bench CarSlot lists to plain Car* lists. Validated
// end-to-end by the in-framework CPU-vs-GPU bit-exactness comparison.

#include "GpuArenaBridge.h"

#ifdef RS_CUDA

#include "../RocketSim.h"
#include "../../libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btStridingMeshInterface.h"
#include "../../libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btTriangleInfoMap.h"
#include "../../libsrc/bullet3-3.24/BulletCollision/BroadphaseCollision/btRSBroadphase.h"
#include "../../libsrc/bullet3-3.24/BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
#include "../../libsrc/bullet3-3.24/BulletCollision/CollisionShapes/btConvexHullShape.h"

#include <cassert>

using namespace RocketSim;

namespace {

constexpr float TICK_TIME = 1.0f / 120.0f;

static rscuda::GpuVec3 ToGpu(const Vec& v) { return { v.x, v.y, v.z }; }
static Vec FromGpu(const rscuda::GpuVec3& v) { return Vec(v.x, v.y, v.z); }

static rscuda::GpuVec3 ToGpuB(const btVector3& v) { return { v.x(), v.y(), v.z() }; }
static btVector3 FromGpuB(const rscuda::GpuVec3& v) { return btVector3(v.x, v.y, v.z); }

// GpuMat3 rows = btMatrix3x3 rows (row-major, BULLET basis - not RocketSim RotMat)
static rscuda::GpuMat3 ToGpuB(const btMatrix3x3& m) {
	return { { ToGpuB(m[0]), ToGpuB(m[1]), ToGpuB(m[2]) } };
}
static btMatrix3x3 FromGpuB(const rscuda::GpuMat3& m) {
	return btMatrix3x3(
		m.row[0].x, m.row[0].y, m.row[0].z,
		m.row[1].x, m.row[1].y, m.row[1].z,
		m.row[2].x, m.row[2].y, m.row[2].z);
}

// Full bullet-unit dynamic state of a rigid body -> GPU mirror
static void SnapshotBody(const btRigidBody& rb, rscuda::GpuBodyState& b) {
	const btTransform& t = rb.getWorldTransform();
	b.basis = ToGpuB(t.getBasis());
	b.pos = ToGpuB(t.getOrigin());
	b.linVel = ToGpuB(rb.getLinearVelocity());
	b.angVel = ToGpuB(rb.getAngularVelocity());
	b.totalForce = ToGpuB(rb.getTotalForce());
	b.totalTorque = ToGpuB(rb.getTotalTorque());
	b.invInertiaWorld = ToGpuB(rb.getInvInertiaTensorWorld());
	b.activationState = rb.getActivationState();
	b.deactivationTime = rb.getDeactivationTime();
}

static void WritebackBody(btRigidBody& rb, const rscuda::GpuBodyState& b) {
	btTransform t;
	t.setBasis(FromGpuB(b.basis));
	t.setOrigin(FromGpuB(b.pos));
	rb.setWorldTransform(t);
	rb.setLinearVelocity(FromGpuB(b.linVel));
	rb.setAngularVelocity(FromGpuB(b.angVel));
	rb.updateInertiaTensor(); // same result as GPU's carried tensor when basis matches
	rb.forceActivationState(b.activationState);
	rb.setDeactivationTime(b.deactivationTime);
}

static rscuda::GpuBodyConsts MakeBodyConsts(const btRigidBody& rb, float dt) {
	rscuda::GpuBodyConsts c = {};
	c.invMass = rb.getInvMass();
	c.invInertiaLocal = ToGpuB(rb.getInvInertiaDiagLocal());
	// CPU recomputes btPow(1-damping, dt) every applyDamping call with constant args;
	// computing it once here with the same expression gives the identical float
	c.linDampFactor = powf(1.0f - rb.getLinearDamping(), dt);
	c.angDampFactor = powf(1.0f - rb.getAngularDamping(), dt);
	c.linSleepThreshSq = rb.getLinearSleepingThreshold() * rb.getLinearSleepingThreshold();
	c.angSleepThreshSq = rb.getAngularSleepingThreshold() * rb.getAngularSleepingThreshold();
	c.noRot = rb.m_noRot;
	// btRigidBody::setGravity: m_gravity (force) = acceleration * (1/invMass)
	c.gravityForce = ToGpuB(rb.getGravity() * (1.0f / rb.getInvMass()));
	return c;
}

static rscuda::GpuCarControls ToGpu(const CarControls& c) {
	return { c.throttle, c.steer, c.pitch, c.yaw, c.roll,
		(uint8_t)c.jump, (uint8_t)c.boost, (uint8_t)c.handbrake };
}
static CarControls FromGpu(const rscuda::GpuCarControls& g) {
	CarControls c = {};
	c.throttle = g.throttle; c.steer = g.steer;
	c.pitch = g.pitch; c.yaw = g.yaw; c.roll = g.roll;
	c.jump = g.jump; c.boost = g.boost; c.handbrake = g.handbrake;
	return c;
}

static void SnapshotCar(Car* car, rscuda::GpuBodyState& body, rscuda::GpuCarGameState& g) {
	CarState cs = car->GetState();
	SnapshotBody(car->_rigidBody, body);

	g = {};
	g.tickCountSinceUpdate = cs.tickCountSinceUpdate;
	g.isOnGround = cs.isOnGround;
	for (int w = 0; w < 4; w++) g.wheelsWithContact[w] = cs.wheelsWithContact[w];
	g.hasJumped = cs.hasJumped; g.hasDoubleJumped = cs.hasDoubleJumped; g.hasFlipped = cs.hasFlipped;
	g.flipRelTorque = ToGpu(cs.flipRelTorque);
	g.jumpTime = cs.jumpTime; g.flipTime = cs.flipTime;
	g.isFlipping = cs.isFlipping; g.isJumping = cs.isJumping;
	g.airTime = cs.airTime; g.airTimeSinceJump = cs.airTimeSinceJump;
	g.boost = cs.boost; g.timeSinceBoosted = cs.timeSinceBoosted;
	g.isBoosting = cs.isBoosting; g.boostingTime = cs.boostingTime;
	g.isSupersonic = cs.isSupersonic; g.supersonicTime = cs.supersonicTime;
	g.handbrakeVal = cs.handbrakeVal;
	g.isAutoFlipping = cs.isAutoFlipping;
	g.autoFlipTimer = cs.autoFlipTimer; g.autoFlipTorqueScale = cs.autoFlipTorqueScale;
	g.worldContact_hasContact = cs.worldContact.hasContact;
	g.worldContact_normal = ToGpu(cs.worldContact.contactNormal);
	g.carContact_otherCarID = cs.carContact.otherCarID;
	g.carContact_cooldownTimer = cs.carContact.cooldownTimer;
	g.isDemoed = cs.isDemoed; g.demoRespawnTimer = cs.demoRespawnTimer;
	g.rotMat = ToGpuB((btMatrix3x3)cs.rotMat);
	g.noContactResponse = car->_rigidBody.hasContactResponse() ? 0 : 1;
	g.ballHitInfo.isValid = cs.ballHitInfo.isValid;
	g.ballHitInfo.relativePosOnBall = ToGpu(cs.ballHitInfo.relativePosOnBall);
	g.ballHitInfo.ballPos = ToGpu(cs.ballHitInfo.ballPos);
	g.ballHitInfo.extraHitVel = ToGpu(cs.ballHitInfo.extraHitVel);
	g.ballHitInfo.tickCountWhenHit = cs.ballHitInfo.tickCountWhenHit;
	g.ballHitInfo.tickCountWhenExtraImpulseApplied = cs.ballHitInfo.tickCountWhenExtraImpulseApplied;
	g.lastControls = ToGpu(cs.lastControls);
	g.velocityImpulseCache = ToGpu(car->_velocityImpulseCache);
	g.carID = car->id;
	g.team = (uint8_t)car->team;
	for (int w = 0; w < 4; w++) {
		const btWheelInfoRL& wi = car->_bulletVehicle.m_wheelInfo[w];
		g.wheels[w].steerAngle = wi.m_steerAngle;
		g.wheels[w].engineForce = wi.m_engineForce;
		g.wheels[w].brake = wi.m_brake;
		g.wheels[w].latFriction = wi.m_latFriction;
		g.wheels[w].longFriction = wi.m_longFriction;
		g.wheels[w].suspensionLength = wi.m_raycastInfo.m_suspensionLength;
		g.wheels[w].extraPushback = wi.m_extraPushback;
	}
}

static void WritebackCar(Car* car, const rscuda::GpuBodyState& body, const rscuda::GpuCarGameState& g) {
	WritebackBody(car->_rigidBody, body);
	CarState& cs = car->_internalState;
	// CarState::rotMat is the device-carried post-tick cache, NOT the live basis -
	// they differ for a car that is demoed right now (post-tick skips the refresh)
	cs.rotMat = (RotMat)FromGpuB(g.rotMat);
	{ // mirror noContactResponse back into the body's collision flags
		int flags = car->_rigidBody.getCollisionFlags();
		if (g.noContactResponse)
			flags |= btCollisionObject::CF_NO_CONTACT_RESPONSE;
		else
			flags &= ~btCollisionObject::CF_NO_CONTACT_RESPONSE;
		car->_rigidBody.setCollisionFlags(flags);
	}
	cs.isOnGround = g.isOnGround;
	for (int w = 0; w < 4; w++) cs.wheelsWithContact[w] = g.wheelsWithContact[w];
	cs.hasJumped = g.hasJumped; cs.hasDoubleJumped = g.hasDoubleJumped; cs.hasFlipped = g.hasFlipped;
	cs.flipRelTorque = FromGpu(g.flipRelTorque);
	cs.jumpTime = g.jumpTime; cs.flipTime = g.flipTime;
	cs.isFlipping = g.isFlipping; cs.isJumping = g.isJumping;
	cs.airTime = g.airTime; cs.airTimeSinceJump = g.airTimeSinceJump;
	cs.boost = g.boost; cs.timeSinceBoosted = g.timeSinceBoosted;
	cs.isBoosting = g.isBoosting; cs.boostingTime = g.boostingTime;
	cs.isSupersonic = g.isSupersonic; cs.supersonicTime = g.supersonicTime;
	cs.handbrakeVal = g.handbrakeVal;
	cs.isAutoFlipping = g.isAutoFlipping;
	cs.autoFlipTimer = g.autoFlipTimer; cs.autoFlipTorqueScale = g.autoFlipTorqueScale;
	cs.worldContact.hasContact = g.worldContact_hasContact;
	cs.worldContact.contactNormal = FromGpu(g.worldContact_normal);
	cs.carContact.otherCarID = g.carContact_otherCarID;
	cs.carContact.cooldownTimer = g.carContact_cooldownTimer;
	cs.isDemoed = g.isDemoed; cs.demoRespawnTimer = g.demoRespawnTimer;
	cs.ballHitInfo.isValid = g.ballHitInfo.isValid;
	cs.ballHitInfo.relativePosOnBall = FromGpu(g.ballHitInfo.relativePosOnBall);
	cs.ballHitInfo.ballPos = FromGpu(g.ballHitInfo.ballPos);
	cs.ballHitInfo.extraHitVel = FromGpu(g.ballHitInfo.extraHitVel);
	cs.ballHitInfo.tickCountWhenHit = g.ballHitInfo.tickCountWhenHit;
	cs.ballHitInfo.tickCountWhenExtraImpulseApplied = g.ballHitInfo.tickCountWhenExtraImpulseApplied;
	cs.lastControls = FromGpu(g.lastControls);
	cs.tickCountSinceUpdate = g.tickCountSinceUpdate;
	car->_velocityImpulseCache = FromGpu(g.velocityImpulseCache);
	for (int w = 0; w < 4; w++) {
		btWheelInfoRL& wi = car->_bulletVehicle.m_wheelInfo[w];
		wi.m_steerAngle = g.wheels[w].steerAngle;
		wi.m_engineForce = g.wheels[w].engineForce;
		wi.m_brake = g.wheels[w].brake;
		wi.m_latFriction = g.wheels[w].latFriction;
		wi.m_longFriction = g.wheels[w].longFriction;
		wi.m_raycastInfo.m_suspensionLength = g.wheels[w].suspensionLength;
		wi.m_extraPushback = g.wheels[w].extraPushback;
	}
}

static void ExtractWorldGeo(Arena* arena,
	std::vector<rscuda::GpuPlane>& planes,
	std::vector<rscuda::GpuMeshInfo>& meshes,
	std::vector<rscuda::GpuVec3>& tris) {
	for (btRigidBody* rb : arena->_worldCollisionRBs) {
		const btCollisionShape* shape = rb->getCollisionShape();
		const btBroadphaseProxy* proxy = rb->getBroadphaseProxy();
		if (shape->getShapeType() == TRIANGLE_MESH_SHAPE_PROXYTYPE) {
			auto* meshShape = (const btBvhTriangleMeshShape*)shape;
			const unsigned char* vertBase;
			int numVerts, vertStride;
			const unsigned char* idxBase;
			int idxStride, numFaces;
			meshShape->getMeshInterface()->getLockedReadOnlyVertexIndexBase(
				&vertBase, numVerts, vertStride, &idxBase, idxStride, numFaces);
			rscuda::GpuMeshInfo mi;
			mi.triStart = (int)(tris.size() / 3);
			mi.triCount = numFaces;
			mi.group = proxy->m_collisionFilterGroup;
			mi.mask = proxy->m_collisionFilterMask;
			meshes.push_back(mi);
			for (int f = 0; f < numFaces; f++) {
				const int* idx = (const int*)(idxBase + (size_t)f * idxStride);
				for (int c = 0; c < 3; c++) {
					const float* v = (const float*)(vertBase + (size_t)idx[c] * vertStride);
					tris.push_back({ v[0], v[1], v[2] });
				}
			}
		} else if (shape->getShapeType() == STATIC_PLANE_PROXYTYPE) {
			auto* planeShape = (const btStaticPlaneShape*)shape;
			rscuda::GpuPlane pl;
			pl.normal = ToGpuB(planeShape->getPlaneNormal());
			pl.planeConst = planeShape->getPlaneConstant();
			pl.origin = ToGpuB(rb->getWorldTransform().getOrigin());
			pl.group = proxy->m_collisionFilterGroup;
			pl.mask = proxy->m_collisionFilterMask;
			planes.push_back(pl);
		}
	}
}

static void ExtractMeshBvhs(Arena* arena,
	std::vector<rscuda::GpuMeshBvh>& bvhs,
	std::vector<rscuda::GpuBvhNode>& nodes,
	std::vector<rscuda::GpuBvhSubtree>& subtrees,
	std::vector<rscuda::GpuTriInfo>& triInfos,
	std::vector<float>& margins) {
	for (btRigidBody* rb : arena->_worldCollisionRBs) {
		btCollisionShape* shape = rb->getCollisionShape();
		if (shape->getShapeType() != TRIANGLE_MESH_SHAPE_PROXYTYPE)
			continue;
		auto* meshShape = (btBvhTriangleMeshShape*)shape;
		btOptimizedBvh* bvh = meshShape->getOptimizedBvh();

		rscuda::GpuMeshBvh mb;
		mb.nodeStart = (int)nodes.size();
		mb.subtreeStart = (int)subtrees.size();
		mb.numNodes = bvh->getQuantizedNodeArray().size();
		mb.bvhAabbMin = ToGpuB(bvh->getBvhAabbMin());
		mb.bvhAabbMax = ToGpuB(bvh->getBvhAabbMax());
		mb.bvhQuantization = ToGpuB(bvh->getBvhQuantization());

		const QuantizedNodeArray& na = bvh->getQuantizedNodeArray();
		for (int i = 0; i < na.size(); i++) {
			rscuda::GpuBvhNode n;
			for (int c = 0; c < 3; c++) {
				n.qmin[c] = na[i].m_quantizedAabbMin[c];
				n.qmax[c] = na[i].m_quantizedAabbMax[c];
			}
			n.escapeIndexOrTriangleIndex = na[i].m_escapeIndexOrTriangleIndex;
			nodes.push_back(n);
		}
		const BvhSubtreeInfoArray& sa = bvh->getSubtreeInfoArray();
		mb.numSubtrees = sa.size();
		for (int i = 0; i < sa.size(); i++) {
			rscuda::GpuBvhSubtree s;
			for (int c = 0; c < 3; c++) {
				s.qmin[c] = sa[i].m_quantizedAabbMin[c];
				s.qmax[c] = sa[i].m_quantizedAabbMax[c];
			}
			s.rootNodeIndex = sa[i].m_rootNodeIndex;
			s.subtreeSize = sa[i].m_subtreeSize;
			subtrees.push_back(s);
		}
		bvhs.push_back(mb);
		margins.push_back(meshShape->getMargin());

		const unsigned char* vertBase;
		int numVerts, vertStride;
		const unsigned char* idxBase;
		int idxStride, numFaces;
		meshShape->getMeshInterface()->getLockedReadOnlyVertexIndexBase(
			&vertBase, numVerts, vertStride, &idxBase, idxStride, numFaces);
		const btTriangleInfoMap* infoMap = meshShape->getTriangleInfoMap();
		for (int f = 0; f < numFaces; f++) {
			rscuda::GpuTriInfo ti = {};
			// btGetHash(partId=0, triIdx) = (0 << 21) | triIdx = triIdx
			const btTriangleInfo* info = infoMap ? infoMap->find(f) : nullptr;
			if (info) {
				ti.hasInfo = 1;
				ti.flags = info->m_flags;
				ti.edgeV0V1Angle = info->m_edgeV0V1Angle;
				ti.edgeV1V2Angle = info->m_edgeV1V2Angle;
				ti.edgeV2V0Angle = info->m_edgeV2V0Angle;
			}
			triInfos.push_back(ti);
		}
	}
}

static rscuda::GpuCurve ExtractCurve(const LinearPieceCurve& c);

static void ExtractCollideData(Arena* arena, const std::vector<Car*>& cars,
	rscuda::GpuCollideConsts& cc,
	std::vector<rscuda::GpuStaticObj>& statics,
	std::vector<int>& cellStart, std::vector<int>& cellList,
	std::vector<rscuda::GpuDynProxy>& proxies, int& regCounter) {
	using namespace RLConst;
	auto* bp = (btRSBroadphase*)arena->_bulletWorldParams.broadphase;

	// static table (handle order) + handle -> static index map
	std::vector<int> staticIdxOfHandle(bp->m_LastHandleIndex + 1, -1);
	int meshCount = 0, planeCount = 0;
	for (int h = 0; h <= bp->m_LastHandleIndex; h++) {
		btRSBroadphaseProxy* p = &bp->m_pHandles[h];
		if (!p->m_clientObject || !p->isStatic)
			continue;
		auto* obj = (btCollisionObject*)p->m_clientObject;
		rscuda::GpuStaticObj so = {};
		if (p->shapeType == TRIANGLE_MESH_SHAPE_PROXYTYPE) {
			so.kind = 0;
			so.shapeIdx = meshCount++;
		} else if (p->shapeType == CONVEX_HULL_SHAPE_PROXYTYPE) {
			so.kind = 2; // dropshot tile
			so.shapeIdx = obj->getUserIndex2(); // total tile index 0..139
		} else {
			so.kind = 1;
			so.shapeIdx = planeCount++;
		}
		so.aabbMin = ToGpuB(p->m_aabbMin);
		so.aabbMax = ToGpuB(p->m_aabbMax);
		{
			btVector3 smn, smx;
			obj->getCollisionShape()->getAabb(obj->getWorldTransform(), smn, smx);
			so.shapeAabbMin = ToGpuB(smn);
			so.shapeAabbMax = ToGpuB(smx);
		}
		so.group = p->m_collisionFilterGroup;
		so.mask = p->m_collisionFilterMask;
		so.friction = obj->getFriction();
		so.restitution = obj->getRestitution();
		so.breakingThreshold = obj->getCollisionShape()->getContactBreakingThreshold(gContactBreakingThreshold);
		staticIdxOfHandle[h] = (int)statics.size();
		statics.push_back(so);
	}

	// per-cell static lists (CSR), preserving CPU smear insertion order
	cellStart.resize(bp->totalCells + 1);
	int total = 0;
	for (int c = 0; c < bp->totalCells; c++) {
		cellStart[c] = total;
		total += (int)bp->cells[c].staticHandles.size();
	}
	cellStart[bp->totalCells] = total;
	cellList.reserve(total);
	for (int c = 0; c < bp->totalCells; c++)
		for (auto* sh : bp->cells[c].staticHandles)
			cellList.push_back(staticIdxOfHandle[(int)(sh - bp->m_pHandles)]);

	// initial dyn proxy state (creation order: ball, cars). regSeq = creation order.
	regCounter = 0;
	auto addProxy = [&](btRigidBody* rb) {
		auto* p = (btRSBroadphaseProxy*)rb->getBroadphaseProxy();
		rscuda::GpuDynProxy dp = {};
		dp.aabbMin = ToGpuB(p->m_aabbMin);
		dp.aabbMax = ToGpuB(p->m_aabbMax);
		dp.cellIdx = p->cellIdx;
		dp.i = p->iIdx;
		dp.j = p->jIdx;
		dp.k = p->kIdx;
		dp.regSeq = ++regCounter;
		proxies.push_back(dp);
	};
	addProxy(&arena->ball->_rigidBody);
	for (Car* car : cars)
		addProxy(&car->_rigidBody);

	// collide consts
	cc = {};
	Car* car = cars[0];
	const btCompoundShape& comp = car->_compoundShape;
	cc.carLocalAabbMin = ToGpuB(comp.getLocalAabbMin());
	cc.carLocalAabbMax = ToGpuB(comp.getLocalAabbMax());
	cc.carCompoundMargin = comp.getMargin();
	btCollisionShape* ballShape = arena->ball->_rigidBody.getCollisionShape();
	if (arena->ball->IsSphere()) {
		cc.ballRadius = ((btSphereShape*)ballShape)->getRadius();
	} else {
		// snowday puck: btConvexHullShape (points uploaded via UploadBallHull).
		// m_localAabb is protected; recompute it exactly as recalcLocalAabb does
		// (6-direction batched support extrema +- margin, margin included ONCE here
		// and added AGAIN by getAabb - bullet quirk, replicated on device)
		auto* hull = (btConvexHullShape*)ballShape;
		cc.ballIsHull = 1;
		cc.ballHullMargin = hull->getMargin();
		static const btVector3 dirs[6] = {
			btVector3(1, 0, 0), btVector3(0, 1, 0), btVector3(0, 0, 1),
			btVector3(-1, 0, 0), btVector3(0, -1, 0), btVector3(0, 0, -1) };
		btVector3 sup[6] = {};
		hull->batchedUnitVectorGetSupportingVertexWithoutMargin(dirs, sup, 6);
		btVector3 lmn, lmx;
		for (int i = 0; i < 3; i++) {
			lmx[i] = sup[i][i] + hull->getMargin();
			lmn[i] = sup[i + 3][i] - hull->getMargin();
		}
		cc.ballHullLocalAabbMin = ToGpuB(lmn);
		cc.ballHullLocalAabbMax = ToGpuB(lmx);
	}
	cc.puckGroundStickForce = Snowday::PUCK_GROUND_STICK_FORCE;
	cc.ballLaunchDelay = (arena->gameMode == GameMode::DROPSHOT)
		? Dropshot::BALL_LAUNCH_DELAY : BALL_HOOPS_LAUNCH_DELAY;
	cc.ballLaunchZVel = (arena->gameMode == GameMode::DROPSHOT)
		? Dropshot::BALL_LAUNCH_Z_VEL : BALL_HOOPS_LAUNCH_Z_VEL;
	cc.ballMassBT = arena->ball->GetMass();
	cc.hoopsZScaleGround = BALL_CAR_EXTRA_IMPULSE_Z_SCALE_HOOPS_GROUND;
	cc.hoopsZScaleThresh = BALL_CAR_EXTRA_IMPULSE_Z_SCALE_HOOPS_NORMAL_Z_THRESH;
	cc.dsMinDownwardSpeedToDamage = Dropshot::MIN_DOWNWARD_SPEED_TO_DAMAGE;
	cc.dsMinChargeHitSpeed = Dropshot::MIN_CHARGE_HIT_SPEED;
	cc.dsMinAbsorbedForCharge = Dropshot::MIN_ABSORBED_FORCE_FOR_CHARGE;
	cc.dsMinAbsorbedForSupercharge = Dropshot::MIN_ABSORBED_FORCE_FOR_SUPERCHARGE;
	cc.dsMinDamageInterval = Dropshot::MIN_DAMAGE_INTERVAL;
	if (arena->gameMode == GameMode::DROPSHOT) {
		cc.numTiles = (int)arena->_worldDropshotTileRBs.size();
		cc.tileMargin = arena->_worldDropshotTileRBs[0]->getCollisionShape()->getMargin();
	}
	const btBoxShape& box = car->_childHitboxShape;
	cc.carBoxHalfExtNoMargin = ToGpuB(box.getHalfExtentsWithoutMargin());
	cc.carBoxMargin = box.getMargin();
	cc.carChildOffset = ToGpuB(car->_compoundShape.getChildTransform(0).getOrigin());
	cc.ballBreakingThreshold = ballShape->getContactBreakingThreshold(gContactBreakingThreshold);
	cc.carBreakingThreshold = comp.getContactBreakingThreshold(gContactBreakingThreshold);
	cc.ballFriction = arena->ball->_rigidBody.getFriction();
	cc.ballRestitution = arena->ball->_rigidBody.getRestitution();
	cc.carFriction = car->_rigidBody.getFriction();
	cc.carRestitution = car->_rigidBody.getRestitution();
	cc.ballFilterGroup = arena->ball->_rigidBody.getBroadphaseProxy()->m_collisionFilterGroup;
	cc.ballFilterMask = arena->ball->_rigidBody.getBroadphaseProxy()->m_collisionFilterMask;
	cc.carFilterGroup = car->_rigidBody.getBroadphaseProxy()->m_collisionFilterGroup;
	cc.carFilterMask = car->_rigidBody.getBroadphaseProxy()->m_collisionFilterMask;
	cc.numStatics = (int)statics.size();
	cc.gameMode = (int)arena->gameMode;
	const MutatorConfig& mc = arena->GetMutatorConfig();
	cc.demoMode = (int)mc.demoMode;
	cc.enableTeamDemos = mc.enableTeamDemos;
	cc.respawnDelay = mc.respawnDelay;
	cc.bumpCooldownTime = mc.bumpCooldownTime;
	cc.bumpForceScale = mc.bumpForceScale;
	cc.ballHitExtraForceScale = mc.ballHitExtraForceScale;
	cc.carWorldFriction = mc.carWorldFriction;
	cc.carWorldRestitution = mc.carWorldRestitution;
	cc.carBallFriction = CARBALL_COLLISION_FRICTION;
	cc.carBallRestitution = CARBALL_COLLISION_RESTITUTION;
	cc.carCarFriction = CARCAR_COLLISION_FRICTION;
	cc.carCarRestitution = CARCAR_COLLISION_RESTITUTION;
	cc.bumpMinForwardDist = BUMP_MIN_FORWARD_DIST;
	cc.ballCarExtraZScale = BALL_CAR_EXTRA_IMPULSE_Z_SCALE;
	cc.ballCarExtraForwardScale = BALL_CAR_EXTRA_IMPULSE_FORWARD_SCALE;
	cc.ballCarExtraMaxDeltaVelUU = BALL_CAR_EXTRA_IMPULSE_MAXDELTAVEL_UU;
	cc.ballExtraImpulseFactorCurve = ExtractCurve(BALL_CAR_EXTRA_IMPULSE_FACTOR_CURVE);
	cc.bumpVelGroundCurve = ExtractCurve(BUMP_VEL_AMOUNT_GROUND_CURVE);
	cc.bumpVelAirCurve = ExtractCurve(BUMP_VEL_AMOUNT_AIR_CURVE);
	cc.bumpUpwardVelCurve = ExtractCurve(BUMP_UPWARD_VEL_AMOUNT_CURVE);
	cc.hsTargetY = Heatseeker::TARGET_Y;
	cc.hsTargetZ = Heatseeker::TARGET_Z;
	cc.hsHorizontalBlend = Heatseeker::HORIZONTAL_BLEND;
	cc.hsVerticalBlend = Heatseeker::VERTICAL_BLEND;
	cc.hsSpeedBlend = Heatseeker::SPEED_BLEND;
	cc.hsMaxTurnPitch = Heatseeker::MAX_TURN_PITCH;
	cc.hsMaxSpeed = Heatseeker::MAX_SPEED;
	cc.hsTargetSpeedIncrement = Heatseeker::TARGET_SPEED_INCREMENT;
	cc.hsMinSpeedupInterval = Heatseeker::MIN_SPEEDUP_INTERVAL;
	cc.hsWallBounceChangeYThresh = Heatseeker::WALL_BOUNCE_CHANGE_Y_THRESH;
	cc.hsWallBounceChangeYNormal = Heatseeker::WALL_BOUNCE_CHANGE_Y_NORMAL;
	cc.hsWallBounceForceScale = Heatseeker::WALL_BOUNCE_FORCE_SCALE;
	cc.hsWallBounceUpFrac = Heatseeker::WALL_BOUNCE_UP_FRAC;
	cc.arenaExtentY = ARENA_EXTENT_Y;
}

static rscuda::GpuCurve ExtractCurve(const LinearPieceCurve& c) {
	rscuda::GpuCurve g = {};
	int i = 0;
	for (auto& p : c.valueMappings) {
		g.xs[i] = p.first;
		g.ys[i] = p.second;
		i++;
	}
	g.n = i;
	return g;
}

static rscuda::GpuCarConsts ExtractCarConsts(Arena* arena, Car* car) {
	using namespace RLConst;
	rscuda::GpuCarConsts cc = {};
	for (int i = 0; i < 4; i++) {
		auto& wi = car->_bulletVehicle.m_wheelInfo[i];
		auto& wc = cc.wheels[i];
		wc.connectionPointCS = ToGpuB(wi.m_chassisConnectionPointCS);
		wc.suspensionRestLength = wi.getSuspensionRestLength();
		wc.wheelRadius = wi.m_wheelsRadius;
		wc.maxSuspensionTravelCm = wi.m_maxSuspensionTravelCm;
		wc.suspensionStiffness = wi.m_suspensionStiffness;
		wc.dampingCompression = wi.m_wheelsDampingCompression;
		wc.dampingRelaxation = wi.m_wheelsDampingRelaxation;
		wc.suspensionForceScale = wi.m_suspensionForceScale;
	}
	cc.wheelDirectionCS = ToGpuB(car->_bulletVehicle.m_wheelInfo[0].m_wheelDirectionCS);
	cc.wheelAxleCS = ToGpuB(car->_bulletVehicle.m_wheelInfo[0].m_wheelAxleCS);
	cc.massBT = CAR_MASS_BT;
	cc.frictionScale = car->_rigidBody.getMass() / 3; // calcFrictionImpulses expression
	cc.dodgeDeadzone = car->config.dodgeDeadzone;
	cc.rayCastGroup = btBroadphaseProxy::DefaultFilter | CollisionMasks::DROPSHOT_FLOOR;
	cc.erp = arena->_bulletWorld.getSolverInfo().m_erp;
	cc.uuToBT = UU_TO_BT;
	cc.btToUU = BT_TO_UU;

	const MutatorConfig& mc = arena->GetMutatorConfig();
	cc.jumpAccel = mc.jumpAccel;
	cc.jumpImmediateForce = mc.jumpImmediateForce;
	cc.jumpImmediateForceRL = JUMP_IMMEDIATE_FORCE;
	cc.boostAccelGround = mc.boostAccelGround;
	cc.boostAccelAir = mc.boostAccelAir;
	cc.boostUsedPerSecond = mc.boostUsedPerSecond;
	cc.rechargeBoostEnabled = mc.rechargeBoostEnabled;
	cc.rechargeBoostDelay = mc.rechargeBoostDelay;
	cc.rechargeBoostPerSecond = mc.rechargeBoostPerSecond;
	cc.unlimitedFlips = mc.unlimitedFlips;
	cc.unlimitedDoubleJumps = mc.unlimitedDoubleJumps;

	cc.powerslideRiseRate = POWERSLIDE_RISE_RATE;
	cc.powerslideFallRate = POWERSLIDE_FALL_RATE;
	cc.throttleDeadzone = THROTTLE_DEADZONE;
	cc.stoppingForwardVel = STOPPING_FORWARD_VEL;
	cc.coastingBrakeFactor = COASTING_BRAKE_FACTOR;
	cc.brakingNoThrottleSpeedThresh = BRAKING_NO_THROTTLE_SPEED_THRESH;
	cc.throttleAirAccel = THROTTLE_AIR_ACCEL;
	cc.jumpMinTime = JUMP_MIN_TIME;
	cc.jumpMaxTime = JUMP_MAX_TIME;
	cc.jumpResetTimePad = JUMP_RESET_TIME_PAD;
	cc.doublejumpMaxDelay = DOUBLEJUMP_MAX_DELAY;
	cc.flipZDamp120 = FLIP_Z_DAMP_120;
	cc.flipZDampStart = FLIP_Z_DAMP_START;
	cc.flipZDampEnd = FLIP_Z_DAMP_END;
	cc.flipTorqueTime = FLIP_TORQUE_TIME;
	cc.flipPitchlockExtraTime = FLIP_PITCHLOCK_EXTRA_TIME;
	cc.flipInitialVelScale = FLIP_INITIAL_VEL_SCALE;
	cc.flipTorqueX = FLIP_TORQUE_X;
	cc.flipTorqueY = FLIP_TORQUE_Y;
	cc.flipForwardImpulseMaxSpeedScale = FLIP_FORWARD_IMPULSE_MAX_SPEED_SCALE;
	cc.flipSideImpulseMaxSpeedScale = FLIP_SIDE_IMPULSE_MAX_SPEED_SCALE;
	cc.flipBackwardImpulseMaxSpeedScale = FLIP_BACKWARD_IMPULSE_MAX_SPEED_SCALE;
	cc.flipBackwardImpulseScaleX = FLIP_BACKWARD_IMPULSE_SCALE_X;
	cc.carTorqueScale = CAR_TORQUE_SCALE;
	cc.carAirControlTorque = ToGpu(CAR_AIR_CONTROL_TORQUE);
	cc.carAirControlDamping = ToGpu(CAR_AIR_CONTROL_DAMPING);
	cc.carAutoflipImpulse = CAR_AUTOFLIP_IMPULSE;
	cc.carAutoflipTorque = CAR_AUTOFLIP_TORQUE;
	cc.carAutoflipTime = CAR_AUTOFLIP_TIME;
	cc.carAutoflipNormZThresh = CAR_AUTOFLIP_NORMZ_THRESH;
	cc.carAutoflipRollThresh = CAR_AUTOFLIP_ROLL_THRESH;
	cc.carAutorollForce = CAR_AUTOROLL_FORCE;
	cc.carAutorollTorque = CAR_AUTOROLL_TORQUE;
	cc.boostMinTime = BOOST_MIN_TIME;
	cc.boostMax = BOOST_MAX;
	cc.carMaxSpeed = CAR_MAX_SPEED;
	cc.suspensionSubtraction = BTVehicle::SUSPENSION_SUBTRACTION;

	cc.throttleTorqueBT = THROTTLE_TORQUE_AMOUNT * UU_TO_BT;
	cc.brakeTorqueBT = BRAKE_TORQUE_AMOUNT * UU_TO_BT;
	cc.gravityZBT = GRAVITY_Z * UU_TO_BT;
	cc.flipZDampFactor = powf(1 - FLIP_Z_DAMP_120, TICK_TIME / (1 / 120.f));

	cc.driveSpeedTorqueFactor = ExtractCurve(DRIVE_SPEED_TORQUE_FACTOR_CURVE);
	cc.steerAngleFromSpeed = ExtractCurve(STEER_ANGLE_FROM_SPEED_CURVE);
	cc.powerslideSteerAngleFromSpeed = ExtractCurve(POWERSLIDE_STEER_ANGLE_FROM_SPEED_CURVE);
	cc.nonStickyFrictionFactor = ExtractCurve(NON_STICKY_FRICTION_FACTOR_CURVE);
	cc.latFriction = ExtractCurve(LAT_FRICTION_CURVE);
	cc.longFriction = ExtractCurve(LONG_FRICTION_CURVE);
	cc.handbrakeLatFrictionFactor = ExtractCurve(HANDBRAKE_LAT_FRICTION_FACTOR_CURVE);
	cc.handbrakeLongFrictionFactor = ExtractCurve(HANDBRAKE_LONG_FRICTION_FACTOR_CURVE);
	return cc;
}

static void ExtractTailData(Arena* arena, const std::vector<Car*>& cars,
	rscuda::GpuTailConsts& tc, std::vector<rscuda::GpuPadConfig>& padConfigs,
	std::vector<rscuda::GpuDynBody>& dynTemplates) {
	using namespace RLConst;
	tc = {};
	tc.supersonicStartSpeed = SUPERSONIC_START_SPEED;
	tc.supersonicMaintainMinSpeed = SUPERSONIC_MAINTAIN_MIN_SPEED;
	tc.supersonicMaintainMaxTime = SUPERSONIC_MAINTAIN_MAX_TIME;
	tc.carMaxSpeedBT = CAR_MAX_SPEED * UU_TO_BT;
	tc.carMaxAngSpeed = CAR_MAX_ANG_SPEED;
	const MutatorConfig& mc = arena->GetMutatorConfig();
	tc.ballMaxSpeedBT = mc.ballMaxSpeed * UU_TO_BT;
	tc.ballMaxAngSpeed = BALL_MAX_ANG_SPEED;
	tc.boostMax = BOOST_MAX;
	tc.boostAmountBig = BoostPads::BOOST_AMOUNT_BIG;
	tc.boostAmountSmall = BoostPads::BOOST_AMOUNT_SMALL;
	tc.padCooldownBig = mc.boostPadCooldown_Big;
	tc.padCooldownSmall = mc.boostPadCooldown_Small;
	tc.padGridExtentZ = BoostPadGrid::EXTENT_Z;
	tc.padGridCellSizeX = BoostPadGrid::CELL_SIZE_X;
	tc.padGridCellSizeY = BoostPadGrid::CELL_SIZE_Y;
	tc.padGridCellsX = BoostPadGrid::CELLS_X;
	tc.padGridCellsY = BoostPadGrid::CELLS_Y;
	tc.cylHeightBT = BoostPads::CYL_HEIGHT * UU_TO_BT;

	{ // Car::Respawn spawn table (mode-appropriate) + spawn constants
		const CarSpawnPos* respawns;
		switch (arena->gameMode) {
		case GameMode::HOOPS: respawns = CAR_RESPAWN_LOCATIONS_HOOPS; break;
		case GameMode::DROPSHOT: respawns = CAR_RESPAWN_LOCATIONS_DROPSHOT; break;
		default: respawns = CAR_RESPAWN_LOCATIONS_SOCCAR;
		}
		for (int i = 0; i < CAR_RESPAWN_LOCATION_AMOUNT; i++) {
			tc.respawnX[i] = respawns[i].x;
			tc.respawnY[i] = respawns[i].y;
			tc.respawnYaw[i] = respawns[i].yawAng;
		}
		tc.carRespawnZ = CAR_RESPAWN_Z;
		tc.carSpawnBoostAmount = mc.carSpawnBoostAmount;
	}

	const auto& padList = arena->GetBoostPads();
	for (BoostPad* pad : padList) {
		rscuda::GpuPadConfig pc = {};
		pc.posBT = ToGpu(pad->_posBT);
		pc.boxMinBT = ToGpu(pad->_boxMinBT);
		pc.boxMaxBT = ToGpu(pad->_boxMaxBT);
		pc.cylRadBT = (pad->config.isBig ? BoostPads::CYL_RAD_BIG : BoostPads::CYL_RAD_SMALL) * UU_TO_BT;
		pc.isBig = pad->config.isBig;
		padConfigs.push_back(pc);
	}
	for (int i = 0; i < BoostPadGrid::CELLS_X; i++) {
		for (int j = 0; j < BoostPadGrid::CELLS_Y; j++) {
			BoostPad* p = arena->_boostPadGrid.pads[i][j];
			int idx = -1;
			if (p)
				for (size_t k = 0; k < padList.size(); k++)
					if (padList[k] == p) { idx = (int)k; break; }
			tc.padGridMap[i * BoostPadGrid::CELLS_Y + j] = (int16_t)idx;
		}
	}

	{ // dyn shape/filter templates (transforms filled per tick on device)
		rscuda::GpuDynBody t = {};
		btRigidBody& rb = arena->ball->_rigidBody;
		if (arena->ball->IsSphere()) {
			t.kind = 1;
			t.sphereRadius = ((btSphereShape*)rb.getCollisionShape())->getRadius();
		} else {
			t.kind = 4; // CU_SHAPE_HULL; sphereRadius slot = collision margin
			t.sphereRadius = rb.getCollisionShape()->getMargin();
		}
		t.group = rb.getBroadphaseProxy()->m_collisionFilterGroup;
		t.mask = rb.getBroadphaseProxy()->m_collisionFilterMask;
		t.invMass = rb.getInvMass();
		t.invInertiaDiagLocal = ToGpuB(rb.getInvInertiaDiagLocal());
		t.hasContactResponse = rb.hasContactResponse();
		dynTemplates.push_back(t);
		for (Car* slotCar : cars) {
			rscuda::GpuDynBody c = {};
			btRigidBody& crb = slotCar->_rigidBody;
			c.kind = 2;
			const btBoxShape& box = slotCar->_childHitboxShape;
			btVector3 he = box.getHalfExtentsWithoutMargin();
			float m = box.getMargin();
			c.boxHalfExtWithMargin = { he.x() + m, he.y() + m, he.z() + m };
			c.childOffset = ToGpuB(slotCar->_compoundShape.getChildTransform(0).getOrigin());
			c.group = crb.getBroadphaseProxy()->m_collisionFilterGroup;
			c.mask = crb.getBroadphaseProxy()->m_collisionFilterMask;
			c.invMass = crb.getInvMass();
			c.invInertiaDiagLocal = ToGpuB(crb.getInvInertiaDiagLocal());
			c.hasContactResponse = crb.hasContactResponse();
			dynTemplates.push_back(c);
		}
	}
}

static void ExtractTiles(Arena* arena,
	std::vector<rscuda::GpuTileShape>& tiles, rscuda::GpuTileTables& tables) {
	for (btRigidBody* rb : arena->_worldDropshotTileRBs) {
		auto* hull = (btConvexHullShape*)rb->getCollisionShape();
		rscuda::GpuTileShape ts = {};
		for (int i = 0; i < 6; i++)
			ts.pts[i] = ToGpuB(hull->getUnscaledPoints()[i]);
		static const btVector3 dirs[6] = {
			btVector3(1, 0, 0), btVector3(0, 1, 0), btVector3(0, 0, 1),
			btVector3(-1, 0, 0), btVector3(0, -1, 0), btVector3(0, 0, -1) };
		btVector3 sup[6] = {};
		hull->batchedUnitVectorGetSupportingVertexWithoutMargin(dirs, sup, 6);
		btVector3 lmn, lmx;
		for (int i = 0; i < 3; i++) {
			lmx[i] = sup[i][i] + hull->getMargin();
			lmn[i] = sup[i + 3][i] - hull->getMargin();
		}
		ts.localAabbMin = ToGpuB(lmn);
		ts.localAabbMax = ToGpuB(lmx);
		ts.origin = ToGpuB(rb->getWorldTransform().getOrigin());
		tiles.push_back(ts);
	}
	tables = {};
	for (int i = 0; i < RLConst::Dropshot::NUM_TILES_PER_TEAM; i++) {
		auto n1 = DropshotTiles::GetNeighborIndices(i, 2);
		auto n2 = DropshotTiles::GetNeighborIndices(i, 3);
		tables.n1Cnt[i] = (int8_t)n1.size();
		for (size_t k = 0; k < n1.size(); k++)
			tables.n1[i][k] = (int8_t)n1[k];
		tables.n2Cnt[i] = (int8_t)n2.size();
		for (size_t k = 0; k < n2.size(); k++)
			tables.n2[i][k] = (int8_t)n2[k];
	}
}

// libstdc++ minstd_rand0 post-.seed() state for a per-arena demo-respawn stream
static uint32_t MinstdSeedState(uint64_t s) {
	uint32_t st = (uint32_t)(s % 2147483647ull);
	return st == 0 ? 1u : st;
}

static uint64_t SplitMix64(uint64_t z) {
	z += 0x9E3779B97F4A7C15ull;
	z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
	z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
	return z ^ (z >> 31);
}

} // namespace

namespace rscuda {

GpuArenaBatch::GpuArenaBatch(const std::vector<Arena*>& arenasIn, uint64_t rngSeed) {
	arenas = arenasIn;
	assert(!arenas.empty());

	GameMode gameMode = arenas[0]->gameMode;
	numCars = (int)arenas[0]->_cars.size();
	numPads = (int)arenas[0]->GetBoostPads().size();
	for (Arena* arena : arenas) {
		assert(arena->gameMode == gameMode);
		assert((int)arena->_cars.size() == numCars);
		assert(arena->tickCount == arenas[0]->tickCount);
	}

	carLists.resize(arenas.size());
	for (size_t a = 0; a < arenas.size(); a++)
		for (Car* car : arenas[a]->_cars)
			carLists[a].push_back(car);

	batch = CreateBatch((int)arenas.size(), numCars, numPads);

	{ // body + car constants (shared across arenas; default car config assumed)
		std::vector<GpuBodyConsts> consts(1 + numCars);
		consts[0] = MakeBodyConsts(arenas[0]->ball->_rigidBody, TICK_TIME);
		for (int c = 0; c < numCars; c++)
			consts[1 + c] = MakeBodyConsts(carLists[0][c]->_rigidBody, TICK_TIME);
		UploadBodyConsts(batch, consts.data());
		UploadCarConsts(batch, ExtractCarConsts(arenas[0], carLists[0][0]));
	}

	// statics reach their steady-state proxy aabbs/cell smear before extraction
	for (Arena* arena : arenas)
		arena->_bulletWorld.updateAabbs();

	{ // world geometry + collide + tail data (identical across arenas)
		std::vector<GpuPlane> planes;
		std::vector<GpuMeshInfo> meshes;
		std::vector<GpuVec3> tris;
		ExtractWorldGeo(arenas[0], planes, meshes, tris);
		UploadWorldGeo(batch, planes.data(), (int)planes.size(),
			meshes.data(), (int)meshes.size(), tris.data(), (int)(tris.size() / 3));

		std::vector<GpuMeshBvh> bvhs;
		std::vector<GpuBvhNode> bvhNodes;
		std::vector<GpuBvhSubtree> bvhSubtrees;
		std::vector<GpuTriInfo> triInfos;
		std::vector<float> meshMargins;
		ExtractMeshBvhs(arenas[0], bvhs, bvhNodes, bvhSubtrees, triInfos, meshMargins);
		UploadMeshBvhs(batch, bvhs.data(), (int)bvhs.size(),
			bvhNodes.data(), (int)bvhNodes.size(),
			bvhSubtrees.data(), (int)bvhSubtrees.size(),
			triInfos.data(), (int)triInfos.size(), meshMargins.data());

		GpuGridParams grid = {};
		auto* bp = (btRSBroadphase*)arenas[0]->_bulletWorldParams.broadphase;
		grid.minPos = ToGpuB(bp->minPos);
		grid.cellSize = bp->cellSize;
		grid.cellSizeSq = bp->cellSizeSq;
		grid.cellsX = bp->cellsX;
		grid.cellsY = bp->cellsY;
		grid.cellsZ = bp->cellsZ;
		GpuCollideConsts ccons;
		std::vector<GpuStaticObj> statics;
		std::vector<int> cellStart, cellList;
		std::vector<GpuDynProxy> proxies0;
		int regCounter0 = 0;
		ExtractCollideData(arenas[0], carLists[0], ccons, statics, cellStart, cellList,
			proxies0, regCounter0);
		UploadCollideData(batch, ccons, statics.data(), (int)statics.size(),
			cellStart.data(), (int)cellStart.size() - 1,
			cellList.data(), (int)cellList.size(), grid);

		GpuTailConsts tc;
		std::vector<GpuPadConfig> padConfigs;
		std::vector<GpuDynBody> dynTemplates;
		ExtractTailData(arenas[0], carLists[0], tc, padConfigs, dynTemplates);
		UploadTailData(batch, tc, padConfigs.data(), numPads, dynTemplates.data());

		if (!arenas[0]->ball->IsSphere()) { // snowday puck hull points
			auto* hull = (btConvexHullShape*)arenas[0]->ball->_collisionShape;
			std::vector<GpuVec3> pts(hull->getNumPoints());
			for (int i = 0; i < hull->getNumPoints(); i++)
				pts[i] = ToGpuB(hull->getUnscaledPoints()[i]);
			UploadBallHull(batch, pts.data(), (int)pts.size());
		}
		if (gameMode == GameMode::DROPSHOT) {
			std::vector<GpuTileShape> tiles;
			GpuTileTables tables;
			ExtractTiles(arenas[0], tiles, tables);
			UploadTiles(batch, tiles.data(), (int)tiles.size(), tables);
		}
	}

	// per-arena: dyn proxies, identity car order (slots ARE the _cars iteration
	// order), demo-respawn RNG stream
	{
		std::vector<int> identityOrder(numCars);
		for (int c = 0; c < numCars; c++)
			identityOrder[c] = c;
		for (int a = 0; a < (int)arenas.size(); a++) {
			GpuCollideConsts ccUnused;
			std::vector<GpuStaticObj> sUnused;
			std::vector<int> aUnused, bUnused;
			std::vector<GpuDynProxy> proxies;
			int regCounter = 0;
			ExtractCollideData(arenas[a], carLists[a], ccUnused, sUnused, aUnused, bUnused,
				proxies, regCounter);
			UploadDynProxies(batch, a, proxies.data(), regCounter);
			UploadCarOrder(batch, a, identityOrder.data());
			UploadRngState(batch, a, MinstdSeedState(SplitMix64(rngSeed ^ (uint64_t)a)));
		}
	}

	SetTickCount(batch, arenas[0]->tickCount);

	bodiesScratch.resize(arenas.size() * (size_t)(1 + numCars));
	curCars.resize(arenas.size() * (size_t)numCars);
	prevCars.resize(arenas.size() * (size_t)numCars);
	padsScratch.resize(arenas.size() * (size_t)numPads);
	ballsScratch.resize(arenas.size());
	controlsScratch.resize(arenas.size() * (size_t)numCars);

	for (int a = 0; a < (int)arenas.size(); a++)
		UploadArena(a);
	prevCars = curCars;
}

GpuArenaBatch::~GpuArenaBatch() {
	if (batch)
		DestroyBatch(batch);
}

void GpuArenaBatch::UploadArena(int idx) {
	Arena* arena = arenas[idx];
	GpuBodyState* bodies = bodiesScratch.data() + (size_t)idx * (1 + numCars);
	GpuCarGameState* cars = curCars.data() + (size_t)idx * numCars;
	GpuPadState* pads = padsScratch.data() + (size_t)idx * numPads;

	SnapshotBody(arena->ball->_rigidBody, bodies[0]);
	for (int c = 0; c < numCars; c++)
		SnapshotCar(carLists[idx][c], bodies[1 + c], cars[c]);
	auto& padList = arena->GetBoostPads();
	for (int p = 0; p < numPads; p++) {
		BoostPadState ps = padList[p]->GetState();
		pads[p].isActive = ps.isActive;
		pads[p].cooldown = ps.cooldown;
		pads[p].prevLockedCarID = ps.prevLockedCarID;
		pads[p].lockedCarIdx = -1;
		for (int c = 0; c < numCars; c++)
			if (ps.curLockedCar == carLists[idx][c])
				pads[p].lockedCarIdx = c;
	}
	UploadArenaState(batch, idx, bodies, cars, pads);

	GpuBallGameState bs = {};
	Ball* ball = arena->ball;
	bs.velocityImpulseCache = ToGpu(ball->_velocityImpulseCache);
	bs.tickCountSinceUpdate = ball->_internalState.tickCountSinceUpdate;
	bs.hsYTargetDir = ball->_internalState.hsInfo.yTargetDir;
	bs.hsCurTargetSpeed = ball->_internalState.hsInfo.curTargetSpeed;
	bs.hsTimeSinceHit = ball->_internalState.hsInfo.timeSinceHit;
	bs.groundStickApplied = ball->_groundStickApplied;
	const auto& dsi = ball->_internalState.dsInfo;
	bs.dsChargeLevel = dsi.chargeLevel;
	bs.dsAccumulatedHitForce = dsi.accumulatedHitForce;
	bs.dsYTargetDir = dsi.yTargetDir;
	bs.dsHasDamaged = dsi.hasDamaged;
	bs.dsLastDamageTick = dsi.hasDamaged ? dsi.lastDamageTick : 0;
	UploadBallState(batch, idx, bs);
}

void GpuArenaBatch::Step(int numTicks) {
	for (int a = 0; a < (int)arenas.size(); a++)
		for (int c = 0; c < numCars; c++)
			controlsScratch[(size_t)a * numCars + c] = ToGpu(carLists[a][c]->controls);
	UploadConstantControls(batch, controlsScratch.data());
	StepTicks(batch, numTicks);
	pendingTicks += numTicks;
}

void GpuArenaBatch::Writeback() {
	std::swap(prevCars, curCars);
	DownloadAllArenaStates(batch, bodiesScratch.data(), curCars.data(), padsScratch.data());
	DownloadAllBallStates(batch, ballsScratch.data());

	for (int a = 0; a < (int)arenas.size(); a++) {
		Arena* arena = arenas[a];
		const GpuBodyState* bodies = bodiesScratch.data() + (size_t)a * (1 + numCars);
		const GpuCarGameState* cars = curCars.data() + (size_t)a * numCars;
		const GpuPadState* pads = padsScratch.data() + (size_t)a * numPads;

		WritebackBody(arena->ball->_rigidBody, bodies[0]);
		for (int c = 0; c < numCars; c++)
			WritebackCar(carLists[a][c], bodies[1 + c], cars[c]);
		auto& padList = arena->GetBoostPads();
		for (int p = 0; p < numPads; p++) {
			BoostPadState ps = padList[p]->GetState();
			ps.isActive = pads[p].isActive;
			ps.cooldown = pads[p].cooldown;
			ps.prevLockedCarID = pads[p].prevLockedCarID;
			ps.curLockedCar = (pads[p].lockedCarIdx >= 0)
				? carLists[a][pads[p].lockedCarIdx] : nullptr;
			padList[p]->SetState(ps);
		}

		Ball* ball = arena->ball;
		const GpuBallGameState& bs = ballsScratch[a];
		ball->_velocityImpulseCache = FromGpu(bs.velocityImpulseCache);
		ball->_internalState.tickCountSinceUpdate = bs.tickCountSinceUpdate;
		ball->_internalState.hsInfo.yTargetDir = bs.hsYTargetDir;
		ball->_internalState.hsInfo.curTargetSpeed = bs.hsCurTargetSpeed;
		ball->_internalState.hsInfo.timeSinceHit = bs.hsTimeSinceHit;
		ball->_groundStickApplied = bs.groundStickApplied;
		auto& dsi = ball->_internalState.dsInfo;
		dsi.chargeLevel = bs.dsChargeLevel;
		dsi.accumulatedHitForce = bs.dsAccumulatedHitForce;
		dsi.yTargetDir = bs.dsYTargetDir;
		dsi.hasDamaged = bs.dsHasDamaged;
		if (bs.dsHasDamaged)
			dsi.lastDamageTick = bs.dsLastDamageTick;

		if (arena->gameMode == GameMode::DROPSHOT) {
			constexpr int NT = RLConst::Dropshot::TEAM_AMOUNT * RLConst::Dropshot::NUM_TILES_PER_TEAM;
			uint8_t dmg[NT], nr[NT];
			DownloadTileState(batch, a, dmg, nr);
			for (int t = 0; t < RLConst::Dropshot::TEAM_AMOUNT; t++)
				for (int i = 0; i < RLConst::Dropshot::NUM_TILES_PER_TEAM; i++)
					arena->_dropshotTilesState.states[t][i].damageState =
						dmg[t * RLConst::Dropshot::NUM_TILES_PER_TEAM + i];
			for (int i = 0; i < NT; i++) {
				auto* rb = arena->_worldDropshotTileRBs[i];
				if (nr[i])
					rb->m_collisionFlags |= btCollisionObject::CF_NO_CONTACT_RESPONSE;
				else
					rb->m_collisionFlags &= ~btCollisionObject::CF_NO_CONTACT_RESPONSE;
			}
		}

		arena->tickCount += pendingTicks;
	}
	pendingTicks = 0;
}

} // namespace rscuda

#endif // RS_CUDA
