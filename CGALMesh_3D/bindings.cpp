#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include "mesh_image.h"

namespace nb = nanobind;

NB_MODULE(cgalmesh3d, m) {
  m.def("mesh_image", &mesh_image,
        nb::arg("input_path"),
        nb::arg("output_path"),
        nb::arg("sizing_scale") = 0.1,
        nb::arg("edge_size") = 1000.0,
        nb::arg("vx") = 0.0,
        nb::arg("vy") = 0.0,
        nb::arg("vz") = 0.0);
}
