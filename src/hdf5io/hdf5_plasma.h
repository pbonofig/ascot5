/**
 * @file hdf5_plasma.h
 * @brief Header file for hdf5_plasma.c
 */
#ifndef HDF5_PLASMA_H
#define HDF5_PLASMA_H
#include <hdf5.h>
#include "../ascot5.h"
#include "../plasma.h"

int hdf5_plasma_init(hid_t f, plasma_data* data, char* qid);
#endif
