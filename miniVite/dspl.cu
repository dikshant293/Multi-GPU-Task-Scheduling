#pragma once
#ifndef DSPL_HPP
#define DSPL_HPP

#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <numeric>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <map>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>

#include "graph.cu"
#include "utils.cu"

struct Comm {
  GraphElem size;
  GraphWeight degree;

  Comm() : size(0), degree(0.0) {};
};

struct CommInfo {
    GraphElem community;
    GraphElem size;
    GraphWeight degree;
};

const int SizeTag           = 1;
const int VertexTag         = 2;
const int CommunityTag      = 3;
const int CommunitySizeTag  = 4;
const int CommunityDataTag  = 5;

__device__ __host__
GraphElem get_owner(GraphElem vertex, GraphElem nv) {
    // In a single GPU context, we have only one owner
    return 0;
}


// CUDA Kernel
__global__ void louvain_iteration_kernel(
    GraphElem nv,
    GraphElem* d_edgeIndices,
    Edge* d_edgeList,
    GraphWeight* d_vDegree,
    GraphElem* d_currComm,
    GraphElem* d_targetComm,
    GraphWeight* d_clusterWeight,
    Comm* d_localCinfo,
    Comm* d_localCupdate,
    GraphWeight constantForSecondTerm
) {
    GraphElem i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= nv)
        return;

    GraphElem e0 = d_edgeIndices[i];
    GraphElem e1 = d_edgeIndices[i + 1];

    GraphElem selfLoop = 0;
    GraphElem cc = d_currComm[i];
    GraphWeight vDegree = d_vDegree[i];
    GraphWeight ccDegree = d_localCinfo[cc].degree;
    GraphElem ccSize = d_localCinfo[cc].size;

    // Build local map and counter
    __shared__ GraphElem clmap[1024];
    __shared__ GraphWeight counter[1024];

    GraphElem numUniqueClusters = 1;
    clmap[0] = cc;
    counter[0] = 0.0;

    for (GraphElem j = e0; j < e1; j++) {
        Edge edge = d_edgeList[j];
        GraphElem tail = edge.tail_;
        GraphWeight weight = edge.weight_;

        if (tail == i)
            selfLoop += weight;

        GraphElem tcomm = d_currComm[tail];

        bool found = false;
        for (GraphElem k = 0; k < numUniqueClusters; k++) {
            if (clmap[k] == tcomm) {
                atomicAdd(&counter[k], weight);
                found = true;
                break;
            }
        }
        if (!found) {
            clmap[numUniqueClusters] = tcomm;
            counter[numUniqueClusters] = weight;
            numUniqueClusters++;
        }
    }

    d_clusterWeight[i] += counter[0];

    // Compute max gain
    GraphElem maxIndex = cc;
    GraphWeight maxGain = 0.0;
    GraphWeight eix = counter[0] - selfLoop;
    GraphWeight ax = ccDegree - vDegree;

    for (GraphElem k = 0; k < numUniqueClusters; k++) {
        GraphElem c = clmap[k];
        if (c != cc) {
            GraphWeight eiy = counter[k];
            GraphWeight ay = d_localCinfo[c].degree;
            GraphWeight curGain = 2.0 * (eiy - eix) - 2.0 * vDegree * (ay - ax) * constantForSecondTerm;

            if ((curGain > maxGain) || ((curGain == maxGain) && (curGain != 0.0) && (c < maxIndex))) {
                maxGain = curGain;
                maxIndex = c;
            }
        }
    }

    // Update local community updates
    if (maxIndex != cc) {
        atomicAdd(&(d_localCupdate[maxIndex].degree), vDegree);
        atomicAdd(&(d_localCupdate[maxIndex].size), 1);
        atomicAdd(&(d_localCupdate[cc].degree), -vDegree);
        atomicAdd(&(d_localCupdate[cc].size), -1);
    }

    d_targetComm[i] = maxIndex;
}

class Louvain {
public:
    Louvain(Graph* graph) : g(graph) {
        nv = g->get_lnv();
        ne = g->get_lne();

        vDegree.resize(nv, 0.0);
        currComm.resize(nv);
        targetComm.resize(nv);
        clusterWeight.resize(nv, 0.0);
        localCinfo.resize(nv);
        localCupdate.resize(nv);

        // Initialize communities and compute vertex degrees
        initialize();
    }

    ~Louvain() {
        // Destructor
    }

    void run(GraphWeight lower, GraphWeight thresh, int& iters) {
        GraphWeight prevMod = lower;
        GraphWeight currMod = -1.0;
        int numIters = 0;

        constantForSecondTerm = calcConstantForSecondTerm();

        while (true) {
            numIters++;

            // Clean cluster weight and local community updates
            std::fill(clusterWeight.begin(), clusterWeight.end(), 0.0);
            std::fill(localCupdate.begin(), localCupdate.end(), Comm());

            // Execute Louvain iteration
            executeLouvainIteration();

            // Update local community info
            updateLocalCinfo();

            // Compute modularity
            currMod = computeModularity();

            // Exit criteria
            if (currMod - prevMod < thresh)
                break;

            prevMod = currMod;
            if (prevMod < lower)
                prevMod = lower;

            // Update communities
            std::swap(currComm, targetComm);
        }

        iters = numIters;
    }

private:
    Graph* g;
    GraphElem nv, ne;
    GraphWeight constantForSecondTerm;

    std::vector<GraphWeight> vDegree;
    std::vector<GraphElem> currComm;
    std::vector<GraphElem> targetComm;
    std::vector<GraphWeight> clusterWeight;
    std::vector<Comm> localCinfo;
    std::vector<Comm> localCupdate;

    void initialize() {
        // Compute vertex degrees and initialize communities
        for (GraphElem i = 0; i < nv; i++) {
            GraphElem e0, e1;
            g->edge_range(i, e0, e1);
            GraphWeight sum = 0.0;
            for (GraphElem j = e0; j < e1; j++) {
                sum += g->get_edge(j).weight_;
            }
            vDegree[i] = sum;
            localCinfo[i].degree = sum;
            localCinfo[i].size = 1;
            currComm[i] = i;
        }
    }

    GraphWeight calcConstantForSecondTerm() {
        GraphWeight totalEdgeWeightTwice = 0.0;
        totalEdgeWeightTwice = std::accumulate(vDegree.begin(), vDegree.end(), 0.0);
        return (1.0 / totalEdgeWeightTwice);
    }

    void executeLouvainIteration() {
        // Allocate device memory
        GraphWeight* d_vDegree;
        GraphElem* d_currComm;
        GraphElem* d_targetComm;
        GraphWeight* d_clusterWeight;
        Comm* d_localCinfo;
        Comm* d_localCupdate;
        Edge* d_edgeList;
        GraphElem* d_edgeIndices;

        cudaMalloc((void**)&d_vDegree, nv * sizeof(GraphWeight));
        cudaMalloc((void**)&d_currComm, nv * sizeof(GraphElem));
        cudaMalloc((void**)&d_targetComm, nv * sizeof(GraphElem));
        cudaMalloc((void**)&d_clusterWeight, nv * sizeof(GraphWeight));
        cudaMalloc((void**)&d_localCinfo, nv * sizeof(Comm));
        cudaMalloc((void**)&d_localCupdate, nv * sizeof(Comm));
        cudaMalloc((void**)&d_edgeList, ne * sizeof(Edge));
        cudaMalloc((void**)&d_edgeIndices, (nv + 1) * sizeof(GraphElem));

        // Copy data to device
        cudaMemcpy(d_vDegree, vDegree.data(), nv * sizeof(GraphWeight), cudaMemcpyHostToDevice);
        cudaMemcpy(d_currComm, currComm.data(), nv * sizeof(GraphElem), cudaMemcpyHostToDevice);
        cudaMemcpy(d_localCinfo, localCinfo.data(), nv * sizeof(Comm), cudaMemcpyHostToDevice);
        cudaMemcpy(d_edgeList, g->edge_list_.data(), ne * sizeof(Edge), cudaMemcpyHostToDevice);
        cudaMemcpy(d_edgeIndices, g->edge_indices_.data(), (nv + 1) * sizeof(GraphElem), cudaMemcpyHostToDevice);

        // Set clusterWeight and localCupdate to zero
        cudaMemset(d_clusterWeight, 0, nv * sizeof(GraphWeight));
        cudaMemset(d_localCupdate, 0, nv * sizeof(Comm));

        // Launch kernel
        int threads_per_block = 256;
        int num_blocks = (nv + threads_per_block - 1) / threads_per_block;

        louvain_iteration_kernel<<<num_blocks, threads_per_block>>>(
            nv,
            d_edgeIndices,
            d_edgeList,
            d_vDegree,
            d_currComm,
            d_targetComm,
            d_clusterWeight,
            d_localCinfo,
            d_localCupdate,
            constantForSecondTerm
        );

        // Copy results back to host
        cudaMemcpy(targetComm.data(), d_targetComm, nv * sizeof(GraphElem), cudaMemcpyDeviceToHost);
        cudaMemcpy(localCupdate.data(), d_localCupdate, nv * sizeof(Comm), cudaMemcpyDeviceToHost);
        cudaMemcpy(clusterWeight.data(), d_clusterWeight, nv * sizeof(GraphWeight), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_vDegree);
        cudaFree(d_currComm);
        cudaFree(d_targetComm);
        cudaFree(d_clusterWeight);
        cudaFree(d_localCinfo);
        cudaFree(d_localCupdate);
        cudaFree(d_edgeList);
        cudaFree(d_edgeIndices);
    }

    void updateLocalCinfo() {
        for (GraphElem i = 0; i < nv; i++) {
            localCinfo[i].degree += localCupdate[i].degree;
            localCinfo[i].size += localCupdate[i].size;
        }
    }

    GraphWeight computeModularity() {
        GraphWeight le_xx = std::accumulate(clusterWeight.begin(), clusterWeight.end(), 0.0);
        GraphWeight la2_x = 0.0;

        for (GraphElem i = 0; i < nv; i++) {
            la2_x += localCinfo[i].degree * localCinfo[i].degree;
        }

        GraphWeight currMod = fabs((le_xx * constantForSecondTerm) - (la2_x * constantForSecondTerm * constantForSecondTerm));
        return currMod;
    }
};

#endif // DSPL_HPP
