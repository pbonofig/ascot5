/**
 * @file offload.c
 * @brief Offload functions
 */

#include <stdlib.h>
#include <string.h>
#include "ascot5.h"
#include "offload.h"

void offload_init_offload(offload_package* o, real** offload_array) {
    *offload_array = NULL;
    o->offload_array_length = 0;
    o->unpack_pos = 0;
}

void offload_free_offload(offload_package* o, real** offload_array) {
    free(*offload_array);
    *offload_array = NULL;
    o->offload_array_length = 0;
    o->unpack_pos = 0;
}

void offload_pack(offload_package* o, real** offload_array, real* pack_array,
                  size_t pack_length) {
    size_t new_length = o->offload_array_length + pack_length;

    real* new_array = (real*) malloc(new_length * sizeof(real));

    if(o->offload_array_length > 0) {
        memcpy(new_array, *offload_array, o->offload_array_length*sizeof(real));
    }

    memcpy(new_array+o->offload_array_length, pack_array,
           pack_length*sizeof(real));

    free(*offload_array);

    *offload_array = new_array;
    o->offload_array_length = new_length;
}

real* offload_unpack(offload_package* o, real* offload_array,
                     size_t pack_length) {
    real* ptr = offload_array + o->unpack_pos;

    o->unpack_pos += pack_length;

    return ptr;
}

