#pragma once
#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <climits>
#include <array>
#include <unordered_map>
#include <random>
#include <cmath>
#include <cassert>
#include <cstring>
#include <stdint.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include "utils.cu"

unsigned seed;

struct Edge
{
    GraphElem tail_;
    GraphWeight weight_;

    Edge() : tail_(-1), weight_(0.0) {}
};

struct EdgeTuple
{
    GraphElem ij_[2];
    GraphWeight w_;

    __host__ __device__
    EdgeTuple(GraphElem i, GraphElem j, GraphWeight w) :
        ij_{ i, j }, w_(w)
    {}
    __host__ __device__
    EdgeTuple(GraphElem i, GraphElem j) :
        ij_{ i, j }, w_(1.0)
    {}
    __host__ __device__
    EdgeTuple() :
        ij_{ -1, -1 }, w_(0.0)
    {}
};

// per process graph instance
class Graph
{
public:
    Graph() :
        lnv_(-1), lne_(-1), nv_(-1),
        ne_(-1)
    {
    }

    Graph(GraphElem lnv, GraphElem lne,
        GraphElem nv, GraphElem ne) :
        lnv_(lnv), lne_(lne),
        nv_(nv), ne_(ne)
    {
        edge_indices_.resize(lnv_ + 1, 0);
        edge_list_.resize(lne_); // this is usually populated later

        parts_.resize(1 + 1); // Only one "process"
        parts_[0] = 0;
        parts_[1] = nv_;
    }

    ~Graph()
    {
        edge_list_.clear();
        edge_indices_.clear();
        parts_.clear();
    }

    // update vertex partition information
    void repart(std::vector<GraphElem> const& parts)
    {
        memcpy(parts_.data(), parts.data(), sizeof(GraphElem) * (1 + 1));
    }

    void set_edge_index(GraphElem const vertex, GraphElem const e0)
    {
#if defined(DEBUG_BUILD)
        assert((vertex >= 0) && (vertex <= lnv_));
        assert((e0 >= 0) && (e0 <= lne_));
        edge_indices_.at(vertex) = e0;
#else
        edge_indices_[vertex] = e0;
#endif
    }

    void edge_range(GraphElem const vertex, GraphElem& e0,
        GraphElem& e1) const
    {
        e0 = edge_indices_[vertex];
        e1 = edge_indices_[vertex + 1];
    }

    void set_nedges(GraphElem lne)
    {
        lne_ = lne;
        edge_list_.resize(lne_);
        ne_ = lne_; // Since there is only one process
    }

    GraphElem get_base(const int rank) const
    {
        return parts_[rank];
    }

    GraphElem get_bound(const int rank) const
    {
        return parts_[rank + 1];
    }

    GraphElem get_range(const int rank) const
    {
        return (parts_[rank + 1] - parts_[rank] + 1);
    }

    int get_owner(const GraphElem vertex) const
    {
        return 0; // Only one "process"
    }

    GraphElem get_lnv() const { return lnv_; }
    GraphElem get_lne() const { return lne_; }
    GraphElem get_nv() const { return nv_; }
    GraphElem get_ne() const { return ne_; }

    // return edge and active info
    // ----------------------------

    Edge const& get_edge(GraphElem const index) const
    {
        return edge_list_[index];
    }

    Edge& set_edge(GraphElem const index)
    {
        return edge_list_[index];
    }

    // local <--> global index translation
    // -----------------------------------
    GraphElem local_to_global(GraphElem idx)
    {
        return (idx + get_base(0));
    }

    GraphElem global_to_local(GraphElem idx)
    {
        return (idx - get_base(0));
    }

    // print edge list (with weights)
    void print(bool print_weight = true) const
    {
        if (lne_ < MAX_PRINT_NEDGE)
        {
            std::cout << "###############" << std::endl;
            std::cout << "Graph:" << std::endl;
            std::cout << "###############" << std::endl;
            GraphElem base = get_base(0);
            for (GraphElem i = 0; i < lnv_; i++)
            {
                GraphElem e0, e1;
                edge_range(i, e0, e1);
                if (print_weight) { // print weights (default)
                    for (GraphElem e = e0; e < e1; e++)
                    {
                        Edge const& edge = get_edge(e);
                        std::cout << i + base << " " << edge.tail_ << " " << edge.weight_ << std::endl;
                    }
                }
                else { // don't print weights
                    for (GraphElem e = e0; e < e1; e++)
                    {
                        Edge const& edge = get_edge(e);
                        std::cout << i + base << " " << edge.tail_ << std::endl;
                    }
                }
            }
        }
        else
        {
            std::cout << "Graph size is {" << lnv_ << ", " << lne_ <<
                "}, which will overwhelm STDOUT." << std::endl;
        }
    }

    // print statistics about edge distribution
    void print_dist_stats()
    {
        long sumdeg = lne_;
        long maxdeg = lne_;
        long my_sq = lne_ * lne_;
        double average = (double)sumdeg;
        double avg_sq = (double)my_sq;
        double var = avg_sq - (average * average);
        double stddev = sqrt(var);

        std::cout << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
        std::cout << "Graph edge distribution characteristics" << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
        std::cout << "Number of vertices: " << nv_ << std::endl;
        std::cout << "Number of edges: " << ne_ << std::endl;
        std::cout << "Maximum number of edges: " << maxdeg << std::endl;
        std::cout << "Average number of edges: " << average << std::endl;
        std::cout << "Expected value of X^2: " << avg_sq << std::endl;
        std::cout << "Variance: " << var << std::endl;
        std::cout << "Standard deviation: " << stddev << std::endl;
        std::cout << "-------------------------------------------------------" << std::endl;
    }

    // public variables
    std::vector<GraphElem> edge_indices_;
    std::vector<Edge> edge_list_;

private:
    GraphElem lnv_, lne_, nv_, ne_;
    std::vector<GraphElem> parts_;
};
// CUDA kernel to generate edges
__global__ void generate_edges_kernel(GraphWeight* d_X, GraphWeight* d_Y, GraphElem n, GraphWeight rn, EdgeTuple* d_edgeList, GraphElem* d_edgeCount, bool unitEdgeWeight)
{
    GraphElem i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n)
        return;

    for (GraphElem j = i + 1; j < n; j++)
    {
        GraphWeight dx = d_X[i] - d_X[j];
        GraphWeight dy = d_Y[i] - d_Y[j];
        GraphWeight ed = sqrt(dx * dx + dy * dy);

        if (ed <= rn)
        {
            GraphElem idx = atomicAdd(d_edgeCount, 2);

            if (unitEdgeWeight)
            {
                d_edgeList[idx] = EdgeTuple(i, j);
                d_edgeList[idx + 1] = EdgeTuple(j, i);
            }
            else
            {
                d_edgeList[idx] = EdgeTuple(i, j, ed);
                d_edgeList[idx + 1] = EdgeTuple(j, i, ed);
            }
        }
    }
}
// RGG graph generator using CUDA
class GenerateRGG
{
public:
    GenerateRGG(GraphElem nv)
    {
        nv_ = nv;

        n_ = nv_;

        // calculate r(n)
        GraphWeight rc = sqrt((GraphWeight)log(nv) / (GraphWeight)(PI * nv));
        GraphWeight rt = sqrt((GraphWeight)2.0736 / (GraphWeight)nv);
        rn_ = (rc + rt) / (GraphWeight)2.0;
    }

    // create RGG and returns Graph
    // use Euclidean distance as edge weight
    // for random edges, choose from (0,1)
    // otherwise, use unit weight throughout
    Graph* generate(bool isLCG, bool unitEdgeWeight = true, GraphWeight randomEdgePercent = 0.0)
    {
        // Generate random coordinate points
        std::vector<GraphWeight> X(n_);
        std::vector<GraphWeight> Y(n_);

        // generate random number within range
        // X: 0, 1
        // Y: 0, 1

        // measure the time to generate random numbers
        seed = (unsigned)reseeder(1);

        if (!isLCG) {
            // Use std::random
            std::default_random_engine generator(seed);
            std::uniform_real_distribution<GraphWeight> distribution(0.0, 1.0);

            for (GraphElem i = 0; i < n_; i++) {
                X[i] = distribution(generator);
                Y[i] = distribution(generator);
            }
        }
        else { // LCG
            // Use LCG from utils.hpp
            LCG xr(/*seed*/1, X.data(), 2 * n_);
            xr.generate();
            xr.rescale(Y.data(), n_, 0.0);
        }

        // Create Graph
        Graph* g = new Graph(n_, 0, nv_, nv_);

        // Allocate device memory
        GraphWeight* d_X;
        GraphWeight* d_Y;
        cudaMalloc((void**)&d_X, n_ * sizeof(GraphWeight));
        cudaMalloc((void**)&d_Y, n_ * sizeof(GraphWeight));

        cudaMemcpy(d_X, X.data(), n_ * sizeof(GraphWeight), cudaMemcpyHostToDevice);
        cudaMemcpy(d_Y, Y.data(), n_ * sizeof(GraphWeight), cudaMemcpyHostToDevice);

        // Prepare edge list
        EdgeTuple* d_edgeList;
        GraphElem* d_edgeCount;

        // Estimate maximum number of edges
        GraphElem max_edges = n_ * n_;

        cudaMalloc((void**)&d_edgeList, max_edges * sizeof(EdgeTuple));
        cudaMalloc((void**)&d_edgeCount, sizeof(GraphElem));

        cudaMemset(d_edgeCount, 0, sizeof(GraphElem));

        // Launch kernel to compute edges
        int threads_per_block = 256;
        int num_blocks = (n_ + threads_per_block - 1) / threads_per_block;

        generate_edges_kernel<<<num_blocks, threads_per_block>>>(d_X, d_Y, n_, rn_, d_edgeList, d_edgeCount, unitEdgeWeight);

        // Copy edge count back
        GraphElem h_edgeCount;
        cudaMemcpy(&h_edgeCount, d_edgeCount, sizeof(GraphElem), cudaMemcpyDeviceToHost);

        // Resize edge list
        std::vector<EdgeTuple> edgeList(h_edgeCount);
        cudaMemcpy(edgeList.data(), d_edgeList, h_edgeCount * sizeof(EdgeTuple), cudaMemcpyDeviceToHost);

        // Set graph edge indices
        std::vector<GraphElem> edge_counts(n_, 0);

        for (auto& e : edgeList)
        {
            edge_counts[e.ij_[0]]++;
        }

        g->edge_indices_[0] = 0;
        for (GraphElem i = 0; i < n_; i++)
        {
            g->edge_indices_[i + 1] = g->edge_indices_[i] + edge_counts[i];
        }

        g->set_nedges(h_edgeCount);

        // Set graph edge list
        g->edge_list_.resize(h_edgeCount);
        std::vector<GraphElem> edge_offsets = g->edge_indices_;

        for (auto& e : edgeList)
        {
            GraphElem idx = edge_offsets[e.ij_[0]]++;
            Edge& edge = g->edge_list_[idx];
            edge.tail_ = e.ij_[1];
            edge.weight_ = e.w_;
        }

        // Free device memory
        cudaFree(d_X);
        cudaFree(d_Y);
        cudaFree(d_edgeList);
        cudaFree(d_edgeCount);

        return g;
    }

    GraphWeight get_d() const { return rn_; }
    GraphElem get_nv() const { return nv_; }

private:
    GraphElem nv_, n_;
    GraphWeight rn_;
};

#endif // GRAPH_HPP
