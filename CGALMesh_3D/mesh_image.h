#ifndef MESH_IMAGE_H
#define MESH_IMAGE_H

#include <string>

bool mesh_image(const std::string& input_path,
                const std::string& output_path,
                double sizing_scale = 0.1,
                double edge_size = 1000.0,
                double vx = 0.0,
                double vy = 0.0,
                double vz = 0.0);

#endif
