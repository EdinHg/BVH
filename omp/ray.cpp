#pragma once
#include <algorithm>
#include <cstdint>

#include "geometry.cpp"

struct Ray { 
    Vector3 O, D; 
    float t = std::numeric_limits<float>::max();
    int hitTriangleIdx = -1;   
};

void IntersectTri(Ray& ray, const Triangle& tri, int triIdx) {
	const Vector3 edge1 = tri.v1 - tri.v0;
	const Vector3 edge2 = tri.v2 - tri.v0;
	const Vector3 h = ray.D.cross(edge2);
	const float a = edge1.dot(h);
	if (a > -0.0001f && a < 0.0001f) return; 
	const float f = 1 / a;
	const Vector3 s = ray.O - tri.v0;
	const float u = f * s.dot(h);
	if (u < 0 || u > 1) return;
	const Vector3 q = s.cross(edge1);
	const float v = f * ray.D.dot(q);
	if (v < 0 || u + v > 1) return;
	const float t = f * edge2.dot(q);
	if (t > 0.0001f && t < ray.t) {
        ray.t = t;
        ray.hitTriangleIdx = triIdx;
    }
}

bool IntersectAABB(Ray &ray, const Vector3 &bmin, const Vector3 &bmax) {
	float tx1 = (bmin.x - ray.O.x) / ray.D.x, tx2 = (bmax.x - ray.O.x) / ray.D.x;
	float tmin = std::min( tx1, tx2 ), tmax = std::max( tx1, tx2 );
	float ty1 = (bmin.y - ray.O.y) / ray.D.y, ty2 = (bmax.y - ray.O.y) / ray.D.y;
	tmin = std::max( tmin, std::min( ty1, ty2 ) ), tmax = std::min( tmax, std::max( ty1, ty2 ) );
	float tz1 = (bmin.z - ray.O.z) / ray.D.z, tz2 = (bmax.z - ray.O.z) / ray.D.z;
	tmin = std::max( tmin, std::min( tz1, tz2 ) ), tmax = std::min( tmax, std::max( tz1, tz2 ) );
	return tmax >= tmin && tmin < ray.t && tmax > 0;
}

bool IntersectAABBLBVH(const Ray& ray, const AABB& bbox) {
    float tmin = 0.0f, tmax = ray.t;
    
    for (int axis = 0; axis < 3; axis++) {
        float rayPos = (axis == 0) ? ray.O.x : (axis == 1) ? ray.O.y : ray.O.z;
        float rayD = (axis == 0) ? ray.D.x : (axis == 1) ? ray.D.y : ray.D.z;
        float bmin = (axis == 0) ? bbox.min.x : (axis == 1) ? bbox.min.y : bbox.min.z;
        float bmax = (axis == 0) ? bbox.max.x : (axis == 1) ? bbox.max.y : bbox.max.z;
        
        if (std::abs(rayD) < 1e-6f) {
            if (rayPos < bmin || rayPos > bmax) return false;
        } else {
            float t1 = (bmin - rayPos) / rayD;
            float t2 = (bmax - rayPos) / rayD;
            if (t1 > t2) std::swap(t1, t2);
            tmin = std::max(tmin, t1);
            tmax = std::min(tmax, t2);
            if (tmin > tmax) return false;
        }
    }
    return true;
}