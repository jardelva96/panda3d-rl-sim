// 2-D ray caster against axis-aligned bounding boxes.
//
// Mirrors panda3d_rl_sim.sensors.ray_cast_aabb and returns identical results
// (up to float rounding), ~10x faster on dense workloads thanks to tight
// loops, early-out in the slab test, and no per-ray Python overhead.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace py = pybind11;

namespace {

inline float ray_aabb(float ox, float oy, float dx, float dy,
                      float xmin, float ymin, float xmax, float ymax,
                      float best) {
    float tmin = 0.0f;
    float tmax = best;

    const float o[2]  = {ox, oy};
    const float d[2]  = {dx, dy};
    const float lo[2] = {xmin, ymin};
    const float hi[2] = {xmax, ymax};

    for (int k = 0; k < 2; ++k) {
        if (std::fabs(d[k]) < 1e-9f) {
            if (o[k] < lo[k] || o[k] > hi[k]) return best;
            continue;
        }
        const float inv = 1.0f / d[k];
        float t1 = (lo[k] - o[k]) * inv;
        float t2 = (hi[k] - o[k]) * inv;
        if (t1 > t2) std::swap(t1, t2);
        if (t1 > tmin) tmin = t1;
        if (t2 < tmax) tmax = t2;
        if (tmin > tmax) return best;
    }
    return tmin > 0.0f ? tmin : best;
}

py::array_t<float> ray_cast_aabb(
        py::array_t<float, py::array::c_style | py::array::forcecast> origin,
        float heading,
        int num_rays,
        float fov_rad,
        float max_range,
        py::array_t<float, py::array::c_style | py::array::forcecast> obstacles) {

    if (num_rays <= 0) {
        return py::array_t<float>(0);
    }
    if (origin.ndim() != 1 || origin.shape(0) != 2) {
        throw std::runtime_error("origin must be a (2,) float32 array");
    }
    if (obstacles.ndim() != 2 || obstacles.shape(1) != 4) {
        throw std::runtime_error("obstacles must be an (N, 4) float32 array");
    }

    const float ox = origin.at(0);
    const float oy = origin.at(1);
    const std::size_t n_obs = static_cast<std::size_t>(obstacles.shape(0));
    const float* obs_ptr = obstacles.data();

    py::array_t<float> out(num_rays);
    float* out_ptr = out.mutable_data();

    const float half_fov = fov_rad * 0.5f;

    for (int i = 0; i < num_rays; ++i) {
        float angle;
        if (num_rays == 1) {
            angle = heading;
        } else {
            const float t = static_cast<float>(i) / static_cast<float>(num_rays - 1);
            angle = heading - half_fov + t * fov_rad;
        }
        const float dx = std::cos(angle);
        const float dy = std::sin(angle);

        float best = max_range;
        for (std::size_t j = 0; j < n_obs; ++j) {
            const float* row = obs_ptr + j * 4;
            const float t = ray_aabb(ox, oy, dx, dy,
                                     row[0], row[1], row[2], row[3],
                                     best);
            if (t < best) best = t;
        }
        out_ptr[i] = best;
    }
    return out;
}

}  // namespace

PYBIND11_MODULE(_raycaster, m) {
    m.doc() = "Vectorized 2D ray-casting against AABB obstacles "
              "(C++ implementation of panda3d_rl_sim.sensors.ray_cast_aabb).";

    m.def("ray_cast_aabb", &ray_cast_aabb,
          py::arg("origin"), py::arg("heading"), py::arg("num_rays"),
          py::arg("fov_rad"), py::arg("max_range"), py::arg("obstacles"),
          R"pbdoc(
Cast `num_rays` 2D rays from `origin` and return the distance to the nearest
axis-aligned-bounding-box hit for each ray.

See panda3d_rl_sim.sensors.ray_cast_aabb for the reference signature and
semantics; this function is the optional high-performance implementation.
          )pbdoc");
}
