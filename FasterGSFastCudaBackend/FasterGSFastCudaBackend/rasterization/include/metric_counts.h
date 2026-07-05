#pragma once

#include "buffer_utils.h"
#include "helper_math.h"
#include <functional>

namespace faster_gs::rasterization {

    void metric_counts_from_buffers(
        char* primitive_buffers_blob,
        char* tile_buffers_blob,
        char* instance_buffers_blob,
        const int* metric_map,
        float* counts,
        const int n_primitives,
        const int n_instances,
        const int instance_primitive_indices_selector,
        const int width,
        const int height);

}
