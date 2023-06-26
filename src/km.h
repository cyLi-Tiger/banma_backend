#pragma once

#include<names.h>

namespace triton {
namespace backend {
namespace NAMESPACE {

void km(float* vec1, float* vec2, float cost, 
        float const* matrix, float epsilon);

}
}
}