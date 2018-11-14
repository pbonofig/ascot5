/**
 * @file hdf5_wall.c
 * @brief Module for reading wall input from HDF5 file
 *
 * Wall data must be read by calling hdf5_wall_init_offload() contained
 * in this module. This module contains reading routines for all wall data
 * types.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>
#include <hdf5_hl.h>
#include "../wall.h"
#include "../wall/wall_2d.h"
#include "../wall/wall_3d.h"
#include "hdf5_wall.h"
#include "hdf5_helpers.h"

int hdf5_wall_read_2D(hid_t f, wall_2d_offload_data* offload_data,
                      real** offload_array, char* qid);
int hdf5_wall_read_3D(hid_t f, wall_3d_offload_data* offload_data,
                      real** offload_array, char* qid);

/**
 * @brief Read wall data from HDF5 file
 *
 * This function reads wall data with given qid while also initializing offload
 * data and allocating and filling offload array. The file is opened and closed
 * outside this function.
 *
 * @param f HDF5 file from which data is read
 * @param offload_data pointer to offload data
 * @param offload_array pointer to offload array
 * @param qid QID of the data
 *
 * @return Zero if reading and initialization of data succeeded
 */
int hdf5_wall_init_offload(hid_t f, wall_offload_data* offload_data,
                           real** offload_array, char* qid) {

    char path[256];
    int err = 1;

    /* Read data the QID corresponds to */

    hdf5_gen_path("/wall/wall_2D-XXXXXXXXXX", qid, path);
    if(hdf5_find_group(f, path) == 0) {
        offload_data->type = wall_type_2D;
        err = hdf5_wall_read_2D(f, &(offload_data->w2d),
                                offload_array, qid);
    }

    hdf5_gen_path("/wall/wall_3D-XXXXXXXXXX", qid, path);
    if(hdf5_find_group(f, path) == 0) {
        offload_data->type = wall_type_3D;
        err = hdf5_wall_read_3D(f, &(offload_data->w3d),
                                offload_array, qid);
    }

    /* Initialize if data was read succesfully */
    if(!err) {
        err = wall_init_offload(offload_data, offload_array);
    }

    return err;
}

/**
 * @brief Read 2D wall data from HDF5 file
 *
 * @param f HDF5 file from which data is read
 * @param offload_data pointer to offload data
 * @param offload_array pointer to offload array
 * @param qid QID of the data
 *
 * @return Zero if reading succeeded
 */
int hdf5_wall_read_2D(hid_t f, wall_2d_offload_data* offload_data,
                      real** offload_array, char* qid) {
    #undef WPATH
    #define WPATH "/wall/wall_2D-XXXXXXXXXX/"

    /* Read number of wall elements and allocate offload array */
    if( hdf5_read_int(WPATH "n", &(offload_data->n),
                      f, qid, __FILE__, __LINE__) ) {return 1;}
    offload_data->offload_array_length = 2 * offload_data->n;
    *offload_array = (real*) malloc(2 * offload_data->n * sizeof(real));

    /* Pointers to beginning of different data series to make code more
     * readable */
    real* r = &(*offload_array)[0];
    real* z = &(*offload_array)[offload_data->n];

    /* Read the wall polygon */
    if( hdf5_read_double(WPATH "r", r,
        f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_double(WPATH "r", z,
        f, qid, __FILE__, __LINE__) ) {return 1;}

    return 0;
}

/**
 * @brief Read 3D wall data from HDF5 file
 *
 * @param f HDF5 file from which data is read
 * @param offload_data pointer to offload data
 * @param offload_array pointer to offload array
 * @param qid QID of the data
 *
 * @return Zero if reading succeeded
 */
int hdf5_wall_read_3D(hid_t f, wall_3d_offload_data* offload_data,
                      real** offload_array, char* qid) {
    #undef WPATH
    #define WPATH "/wall/wall_3D-XXXXXXXXXX/"

    /* Read number of wall elements and allocate offload array to
       store n 3D triangles */
    if( hdf5_read_int(WPATH "n", &(offload_data->n),
                      f, qid, __FILE__, __LINE__) ) {return 1;}
    offload_data->offload_array_length = 9 * offload_data->n;
    *offload_array = (real*) malloc(9 * offload_data->n * sizeof(real));

    if( hdf5_read_double(WPATH "min_x", &(offload_data->xmin),
                         f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_double(WPATH "max_x", &(offload_data->xmax),
                         f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_double(WPATH "min_y", &(offload_data->ymin),
                         f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_double(WPATH "max_y", &(offload_data->ymax),
                         f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_double(WPATH "min_z", &(offload_data->zmin),
                         f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_double(WPATH "max_z", &(offload_data->zmax),
                         f, qid, __FILE__, __LINE__) ) {return 1;}

    /* Allocate temporary arrays for x1x2x3, y1y2y3, z1z2z3 for each triangle */
    real* x1x2x3 = (real*)malloc(3 * offload_data->n * sizeof(real));
    real* y1y2y3 = (real*)malloc(3 * offload_data->n * sizeof(real));
    real* z1z2z3 = (real*)malloc(3 * offload_data->n * sizeof(real));

    if( hdf5_read_double(WPATH "x1x2x3", x1x2x3,
                         f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_double(WPATH "y1y2y3", y1y2y3,
                         f, qid, __FILE__, __LINE__) ) {return 1;}
    if( hdf5_read_double(WPATH "z1z2z3", z1z2z3,
                         f, qid, __FILE__, __LINE__) ) {return 1;}

    /* The data in the offload array is to be in the format
     *  [x1 y1 z1 x2 y2 z2 x3 y3 z3; ... ]
     */
    int i, j;
    for(i = 0; i < offload_data->n; i++) {
        for(j = 0; j < 3; j++) {
            (*offload_array)[i*9 + j*3 + 0] = x1x2x3[3*i+j];
            (*offload_array)[i*9 + j*3 + 1] = y1y2y3[3*i+j];
            (*offload_array)[i*9 + j*3 + 2] = z1z2z3[3*i+j];
        }
    }

    free(x1x2x3);
    free(y1y2y3);
    free(z1z2z3);

    return 0;
}
