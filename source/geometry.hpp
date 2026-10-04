#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <vector>

namespace geometry {

	struct Vertex {
		float position[3];
		float color[3];
	};

	struct Mesh {
		std::vector<Vertex> vertices;
		std::vector<uint16_t> indices;
	};

	inline Mesh make_truncated_tetrahedron() {
		using V3 = std::array<float, 3>;

		auto dot = [](const V3& a, const V3& b) {
			return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
			};
		auto sub = [](const V3& a, const V3& b) {
			return V3{ a[0] - b[0], a[1] - b[1], a[2] - b[2] };
			};
		auto cross = [](const V3& a, const V3& b) {
			return V3{ a[1] * b[2] - a[2] * b[1],
					  a[2] * b[0] - a[0] * b[2],
					  a[0] * b[1] - a[1] * b[0] };
			};
		auto normalize = [&](const V3& a) {
			float l = std::sqrt(dot(a, a));
			return V3{ a[0] / l, a[1] / l, a[2] / l };
			};

		std::vector<V3> p;
		const float base[3][3] = { {3, 1, 1}, {1, 3, 1}, {1, 1, 3} };
		for (const auto& b : base) {
			for (int sx = -1; sx <= 1; sx += 2)
				for (int sy = -1; sy <= 1; sy += 2)
					for (int sz = -1; sz <= 1; sz += 2) {
						if (sx * sy * sz != 1) continue; 
						p.push_back({ sx * b[0] / 3.0f, sy * b[1] / 3.0f, sz * b[2] / 3.0f });
					}
		}

		Mesh mesh;
		for (const auto& v : p) {
			mesh.vertices.push_back({
				{v[0], v[1], v[2]},
				{v[0] * 0.5f + 0.5f, v[1] * 0.5f + 0.5f, v[2] * 0.5f + 0.5f}
				});
		}

		const V3 dirs[4] = { {1, 1, 1}, {1, -1, -1}, {-1, 1, -1}, {-1, -1, 1} };
		for (int s = 1; s >= -1; s -= 2) {
			for (const auto& d : dirs) {
				V3 n = normalize({ d[0] * s, d[1] * s, d[2] * s });

				float best = -1e9f;
				for (const auto& v : p) best = std::max(best, dot(v, n));

				std::vector<int> face;
				for (int i = 0; i < (int)p.size(); ++i)
					if (dot(p[i], n) > best - 1e-4f) face.push_back(i);

				V3 c{ 0, 0, 0 };
				for (int i : face)
					for (int k = 0; k < 3; ++k) c[k] += p[i][k] / face.size();
				V3 u = normalize(sub(p[face[0]], c));
				V3 w = cross(n, u);

				std::sort(face.begin(), face.end(), [&](int a, int b) {
					V3 pa = sub(p[a], c), pb = sub(p[b], c);
					return std::atan2(dot(pa, w), dot(pa, u)) <
						std::atan2(dot(pb, w), dot(pb, u));
					});

				for (size_t i = 1; i + 1 < face.size(); ++i) {
					mesh.indices.push_back((uint16_t)face[0]);
					mesh.indices.push_back((uint16_t)face[i]);
					mesh.indices.push_back((uint16_t)face[i + 1]);
				}
			}
		}
		return mesh;
	}

} // namespace geometry