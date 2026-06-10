#include "PhaseCapture.h"

#include "../../libsrc/bullet3-3.24/BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"
#include "../../libsrc/bullet3-3.24/BulletDynamics/Dynamics/btRigidBody.h"
#include "../../libsrc/bullet3-3.24/BulletCollision/CollisionDispatch/btCollisionDispatcher.h"
#include "../../libsrc/bullet3-3.24/BulletCollision/NarrowPhaseCollision/btPersistentManifold.h"
#include "../../libsrc/bullet3-3.24/BulletCollision/BroadphaseCollision/btRSBroadphase.h"
#include "../Sim/Car/Car.h"

namespace rscap {

bool enabled = false;
std::vector<BodySnap> points[POINT_COUNT];
std::vector<CarPreTickSnap> carSnaps;
std::vector<ManifoldSnap> manifolds;
std::vector<ProxySnap> proxies;

const char* const POINT_NAMES[POINT_COUNT] = {
	"pre_bullet", "after_predict", "after_collision", "after_islands",
	"after_solve", "after_integrate", "after_activation",
};

static void CopyVec(float* dst, const btVector3& v) {
	dst[0] = v.x(); dst[1] = v.y(); dst[2] = v.z();
}
static void CopyMat(float* dst, const btMatrix3x3& m) {
	for (int i = 0; i < 3; i++)
		CopyVec(dst + i * 3, m[i]);
}

void Capture(Point p, btDiscreteDynamicsWorld* world) {
	auto& bodies = world->getNonStaticRigidBodies();
	auto& out = points[p];
	out.clear();
	out.resize(bodies.size());
	for (int i = 0; i < bodies.size(); i++) {
		btRigidBody* rb = bodies[i];
		BodySnap& s = out[i];
		s.rb = rb;
		CopyMat(s.basis, rb->getWorldTransform().getBasis());
		CopyVec(s.pos, rb->getWorldTransform().getOrigin());
		CopyVec(s.linVel, rb->getLinearVelocity());
		CopyVec(s.angVel, rb->getAngularVelocity());
		CopyVec(s.totalForce, rb->getTotalForce());
		CopyVec(s.totalTorque, rb->getTotalTorque());
		CopyMat(s.invInertiaWorld, rb->getInvInertiaTensorWorld());
		s.activationState = rb->getActivationState();
		s.deactivationTime = rb->getDeactivationTime();
	}

	if (p == AFTER_COLLISION) {
		auto* disp = world->getDispatcher();
		manifolds.clear();
		int n = disp->getNumManifolds();
		manifolds.resize(n);
		for (int i = 0; i < n; i++) {
			const btPersistentManifold* m = disp->getManifoldByIndexInternal(i);
			ManifoldSnap& ms = manifolds[i];
			ms.body0 = m->getBody0();
			ms.body1 = m->getBody1();
			ms.numContacts = m->getNumContacts();
			for (int c = 0; c < m->getNumContacts() && c < 4; c++) {
				const btManifoldPoint& cp = m->getContactPoint(c);
				ContactSnap& cs = ms.pts[c];
				CopyVec(cs.localPointA, cp.m_localPointA);
				CopyVec(cs.localPointB, cp.m_localPointB);
				CopyVec(cs.positionWorldOnA, cp.m_positionWorldOnA);
				CopyVec(cs.positionWorldOnB, cp.m_positionWorldOnB);
				CopyVec(cs.normalWorldOnB, cp.m_normalWorldOnB);
				cs.distance = cp.m_distance1;
				cs.combinedFriction = cp.m_combinedFriction;
				cs.combinedRestitution = cp.m_combinedRestitution;
				cs.partId0 = cp.m_partId0;
				cs.partId1 = cp.m_partId1;
				cs.index0 = cp.m_index0;
				cs.index1 = cp.m_index1;
				cs.isSpecial = cp.m_isSpecial;
			}
		}
		proxies.clear();
		proxies.resize(bodies.size());
		for (int i = 0; i < bodies.size(); i++) {
			btRigidBody* rb = bodies[i];
			ProxySnap& ps = proxies[i];
			ps.rb = rb;
			auto* proxy = (btRSBroadphaseProxy*)rb->getBroadphaseProxy();
			CopyVec(ps.aabbMin, proxy->m_aabbMin);
			CopyVec(ps.aabbMax, proxy->m_aabbMax);
			ps.cellIdx = proxy->cellIdx;
			ps.iIdx = proxy->iIdx;
			ps.jIdx = proxy->jIdx;
			ps.kIdx = proxy->kIdx;
		}
	}
}

static void SnapBody(BodySnap& s, btRigidBody* rb) {
	s.rb = rb;
	CopyMat(s.basis, rb->getWorldTransform().getBasis());
	CopyVec(s.pos, rb->getWorldTransform().getOrigin());
	CopyVec(s.linVel, rb->getLinearVelocity());
	CopyVec(s.angVel, rb->getAngularVelocity());
	CopyVec(s.totalForce, rb->getTotalForce());
	CopyVec(s.totalTorque, rb->getTotalTorque());
	CopyMat(s.invInertiaWorld, rb->getInvInertiaTensorWorld());
	s.activationState = rb->getActivationState();
	s.deactivationTime = rb->getDeactivationTime();
}

void CaptureCarPreTick(RocketSim::Car* car, int stage, float solverTimeStep) {
	using namespace RocketSim;
	CarPreTickSnap s = {};
	s.carId = car->id;
	s.stage = stage;
	s.solverTimeStep = solverTimeStep;
	SnapBody(s.body, &car->_rigidBody);

	{ // all nonstatic bodies (ray candidates / ground bodies), live at this moment
		auto* world = (btDiscreteDynamicsWorld*)car->_bulletVehicle.m_dynamicsWorld;
		auto& bodies = world->getNonStaticRigidBodies();
		s.numDyn = 0;
		for (int i = 0; i < bodies.size() && s.numDyn < 8; i++) {
			btRigidBody* rb = bodies[i];
			DynSnap& d = s.dyn[s.numDyn++];
			d.rb = rb;
			CopyMat(d.basis, rb->getWorldTransform().getBasis());
			CopyVec(d.origin, rb->getWorldTransform().getOrigin());
			CopyVec(d.linVel, rb->getLinearVelocity());
			CopyVec(d.angVel, rb->getAngularVelocity());
			CopyMat(d.invInertiaWorld, rb->getInvInertiaTensorWorld());
			CopyVec(d.invInertiaDiagLocal, rb->getInvInertiaDiagLocal());
			d.invMass = rb->getInvMass();
			CopyVec(d.aabbMin, rb->getBroadphaseProxy()->m_aabbMin);
			d.hasContactResponse = rb->hasContactResponse();
		}
	}

	for (int i = 0; i < 4; i++) {
		const btWheelInfoRL& wi = car->_bulletVehicle.m_wheelInfo[i];
		WheelSnap& w = s.wheels[i];
		w.steerAngle = wi.m_steerAngle;
		w.engineForce = wi.m_engineForce;
		w.brake = wi.m_brake;
		w.latFriction = wi.m_latFriction;
		w.longFriction = wi.m_longFriction;
		w.suspensionLength = wi.m_raycastInfo.m_suspensionLength;
		CopyMat(w.worldBasis, wi.m_worldTransform.getBasis());
		CopyVec(w.hardPointWS, wi.m_raycastInfo.m_hardPointWS);
		CopyVec(w.contactPointWS, wi.m_raycastInfo.m_contactPointWS);
		CopyVec(w.contactNormalWS, wi.m_raycastInfo.m_contactNormalWS);
		w.isInContact = wi.m_raycastInfo.m_isInContact;
		w.isInContactWithWorld = wi.m_isInContactWithWorld;
		w.groundRb = wi.m_raycastInfo.m_groundObject;
		w.groundIsStatic = wi.m_raycastInfo.m_groundObject
			? (((btCollisionObject*)wi.m_raycastInfo.m_groundObject)->isStaticObject() ? 1 : 0)
			: -1;
		w.suspensionRelativeVelocity = wi.m_suspensionRelativeVelocity;
		w.clippedInvContactDotSuspension = wi.m_clippedInvContactDotSuspension;
		w.extraPushback = wi.m_extraPushback;
		CopyVec(w.impulse, wi.m_impulse);
	}

	const CarState& st = car->_internalState;
	s.handbrakeVal = st.handbrakeVal;
	s.boost = st.boost;
	s.boostingTime = st.boostingTime;
	s.timeSinceBoosted = st.timeSinceBoosted;
	s.jumpTime = st.jumpTime;
	s.flipTime = st.flipTime;
	s.airTime = st.airTime;
	s.airTimeSinceJump = st.airTimeSinceJump;
	s.autoFlipTimer = st.autoFlipTimer;
	s.autoFlipTorqueScale = st.autoFlipTorqueScale;
	s.flipRelTorque[0] = st.flipRelTorque.x;
	s.flipRelTorque[1] = st.flipRelTorque.y;
	s.flipRelTorque[2] = st.flipRelTorque.z;
	s.worldContactNormal[0] = st.worldContact.contactNormal.x;
	s.worldContactNormal[1] = st.worldContact.contactNormal.y;
	s.worldContactNormal[2] = st.worldContact.contactNormal.z;
	s.isOnGround = st.isOnGround;
	for (int i = 0; i < 4; i++) s.wheelsWithContact[i] = st.wheelsWithContact[i];
	s.hasJumped = st.hasJumped;
	s.hasDoubleJumped = st.hasDoubleJumped;
	s.hasFlipped = st.hasFlipped;
	s.isFlipping = st.isFlipping;
	s.isJumping = st.isJumping;
	s.isBoosting = st.isBoosting;
	s.isAutoFlipping = st.isAutoFlipping;
	s.worldContactHasContact = st.worldContact.hasContact;
	s.isDemoed = st.isDemoed;

	s.ctrlThrottle = car->controls.throttle;
	s.ctrlSteer = car->controls.steer;
	s.ctrlPitch = car->controls.pitch;
	s.ctrlYaw = car->controls.yaw;
	s.ctrlRoll = car->controls.roll;
	s.ctrlJump = car->controls.jump;
	s.ctrlBoost = car->controls.boost;
	s.ctrlHandbrake = car->controls.handbrake;
	s.lastCtrlJump = st.lastControls.jump;
	s.forwardSpeedUU = car->_bulletVehicle.getForwardSpeed() * BT_TO_UU;

	carSnaps.push_back(s);
}

} // namespace rscap
