#include <stdio.h>
#include <iostream>
#include <limits.h>
#include <stdlib.h>
#include <ctime>
#include <sstream>
#include <string>

#include "gpu_hashtable.hpp"

#define MAX_LOADFACTOR 0.75f

__device__
int my_hash(int data, int limit) {
    return ((long)abs(data) * 184014863) % 203676871 % limit;
}


__device__
void add_entry(entry *data, int data_size, int key, int value)
{
    int hashed_key;
    hashed_key = my_hash(key, data_size);
    int old;

    for (int i = 0; i < data_size; i++) {
        old = atomicCAS(&(data[hashed_key].key), 0, key);
        if (old == 0 || old == key) {
            break;
        }
        hashed_key = (hashed_key + 1) % data_size;
    }
    if (old == 0 || old == key) {
        data[hashed_key].value = value;
    }
}

__device__
int get_entry(entry *data, int data_size, int key)
{
    int hashed_key;
    hashed_key = my_hash(key, data_size);
    int key_detected;

    for (int i = 0; i < data_size; i++) {
        key_detected = data[hashed_key].key;
        if (key_detected == key) {
            break;
        }
        hashed_key = (hashed_key + 1) % data_size;
    }
    if (key_detected == data[hashed_key].key) {
        return data[hashed_key].value;
    }
    return -1;
}

__global__
void GPUinsert(int *keys, int *values, int keys_size, entry *data, int data_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < keys_size; i += stride) {
        add_entry(data, data_size, keys[i], values[i]);
    }
}

__global__
void GPUget(int *keys, int *return_values, int keys_size, entry *data, int data_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < keys_size; i += stride) {
        return_values[i] = get_entry(data, data_size, keys[i]);
    }
}

__global__
void GPUreinsert(entry *dst, entry *src, int dst_size, int src_size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = index; i < src_size; i += stride) {
        if (src[i].key != 0) {
            add_entry(dst, dst_size, src[i].key, src[i].value);
        }
    }
}


/* INIT HASH
 */

GpuHashTable::GpuHashTable(int size) {
    this->size = size;
    int cmr =cudaMalloc((void**) &(this->data), size * sizeof(entry));
    cudaMemset(data, 0, size * sizeof(entry));
}

/* DESTROY HASH
 */
GpuHashTable::~GpuHashTable() {
    cudaFree(data);
}

/* RESHAPE HASH
 */
void GpuHashTable::reshape(int numBucketsReshape) {
    entry* bigger_data;
    cudaMalloc((void**) &bigger_data, numBucketsReshape * sizeof(entry));
    cudaMemset(bigger_data, 0, numBucketsReshape * sizeof(entry));

    //mut totul in bigger_data
    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    cudaDeviceSynchronize(); // make sure previous stuff finished
    GPUreinsert<<<numBlocks, blockSize>>>(bigger_data, data, numBucketsReshape, size);
    cudaDeviceSynchronize(); // wait to finish the process


    cudaFree(data);
    data = bigger_data;
    size = numBucketsReshape;
}

/* INSERT BATCH
 */
bool GpuHashTable::insertBatch(int *keys, int *values, int numKeys) {
    //manage hashtable's size
    int new_size = size;
    int current_occupied = occupied();
    cerr << "size:" << size << endl;
    cerr << "current_occupied:" << current_occupied << endl;
    cerr << "numKeys:" << numKeys << endl;
    cerr << "loadFactor:" << (current_occupied + numKeys) * 1.0f / new_size << endl;
    while ((current_occupied + numKeys) * 1.0f / new_size > MAX_LOADFACTOR) {
        new_size *= 2;
    }
    cerr << "new_size:" << new_size << endl;
    if (new_size != size) {
        reshape(new_size);
    }

    // alloc gpu memory
    int *gpu_keys;
    int *gpu_values;
    cudaMalloc(&gpu_keys, numKeys * sizeof(int));
    cudaMalloc(&gpu_values, numKeys * sizeof(int));

    // transfer from input from cpu to gpu
    cudaMemcpy(gpu_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(gpu_values, values, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    // run insertion
    int blockSize = 256;
    int numBlocks = (numKeys + blockSize - 1) / blockSize;
    GPUinsert<<<numBlocks, blockSize>>>(gpu_keys, gpu_values, numKeys, data, size);

    return true; // ce ar trebui sa intoarca aceasta functie?
}

/* GET BATCH
 */
int* GpuHashTable::getBatch(int* keys, int numKeys) {
    // alloc gpu memory
    int *gpu_keys;
    int *gpu_values;
    cudaMalloc(&gpu_keys, numKeys * sizeof(int));
    cudaMalloc(&gpu_values, numKeys * sizeof(int));

    // transfer input from cpu to gpu
    cudaMemcpy(gpu_keys, keys, numKeys * sizeof(int), cudaMemcpyHostToDevice);

    // run extraction
    int blockSize = 256;
    int numBlocks = (numKeys + blockSize - 1) / blockSize;
    cudaDeviceSynchronize(); // make sure previous stuff finished
    GPUget<<<numBlocks, blockSize>>>(gpu_keys, gpu_values, numKeys, data, size);

    // retrive result
    int *results = (int*)malloc(numKeys * sizeof(int));
    cudaDeviceSynchronize(); // wait  for GPU to process
    cudaMemcpy(results, gpu_values, numKeys * sizeof(int), cudaMemcpyDeviceToHost);

    return results;
}

/* GET LOAD FACTOR
 * num elements / hash total slots elements
 */

__global__
void kernel_occupied(entry *data, int size, int *result) {
    int occupied = 0;
    for (int i = 0; i < size; i++) {
        if (data[i].key != 0) {
            occupied++;
        }
    }
    *result = occupied;
}

int GpuHashTable::occupied() {
    int *result;
    int to_return;
    cudaMalloc(&result, sizeof(int));
    cudaDeviceSynchronize(); // make sure previous stuff finished
    kernel_occupied<<<1, 1>>>(data, size, result);
    cudaDeviceSynchronize(); // wait  for GPU to process
    cudaMemcpy(&to_return, result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(result);
    return to_return;
}


float GpuHashTable::loadFactor() {
    float tmp = 1.0f * occupied() / size; // no larger than 1.0f = 100%
    return tmp;
}

/*********************************************************/

#define HASH_INIT GpuHashTable GpuHashTable(1);
#define HASH_RESERVE(size) GpuHashTable.reshape(size);

#define HASH_BATCH_INSERT(keys, values, numKeys) GpuHashTable.insertBatch(keys, values, numKeys)
#define HASH_BATCH_GET(keys, numKeys) GpuHashTable.getBatch(keys, numKeys)

#define HASH_LOAD_FACTOR GpuHashTable.loadFactor()

#include "test_map.cpp"
