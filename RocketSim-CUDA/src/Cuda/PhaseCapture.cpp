#include "PhaseCapture.h"

#include "../../libsrc/bullet3-3.24/BulletDynamics/Dynamics/btDiscreteDynamicsWorld.h"
#include "../../libsrc/bullet3-3.24/BulletDynamics/Dynamics/btRigidBody.h"

namespace rscap {

bool enabled = false;
std::vector<BodySnap> points[POINT_COUNT];

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
}

} // namespace rscap
