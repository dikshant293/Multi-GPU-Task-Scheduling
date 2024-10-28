#include <cassert>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>

#include <omp.h>
#include <cuda_runtime.h>

#include "dspl.cu"

// Global variables and options
static std::string inputFileName;
static GraphElem nvRGG = 0;
static bool generateGraph = false;
static bool showGraph = false;
static GraphWeight randomEdgePercent = 0.0;
static bool randomNumberLCG = false;
static bool isUnitEdgeWeight = true;
static GraphWeight threshold = 1.0E-6;

// Function to parse command-line parameters
static void parseCommandLine(const int argc, char * const argv[]);

int main(int argc, char *argv[])
{
    double t0, t1, td0, td1, td;

    parseCommandLine(argc, argv);

    td0 = omp_get_wtime();

    Graph* g = nullptr;

    // Generate or read graph
    if (generateGraph) {
        GenerateRGG gr(nvRGG);
        g = gr.generate(randomNumberLCG, isUnitEdgeWeight, randomEdgePercent);
    }
    else {
        // Implement reading from a file if necessary
        std::cerr << "Reading graph from file is not implemented in this version." << std::endl;
        return -1;
    }

    assert(g != nullptr);
    if (showGraph)
        g->print();

    td1 = omp_get_wtime();
    td = td1 - td0;

    std::cout << "Time to generate graph (in s): " << td << std::endl;

    g->print_dist_stats();
    
    GraphWeight currMod = -1.0;
    double total = 0.0;
    int iters = 0;

    t1 = omp_get_wtime();

    // Run Louvain algorithm
    Louvain louvain(g);
    louvain.run(currMod, threshold, iters);

    t0 = omp_get_wtime();
    total = t0 - t1;

    std::cout << "-------------------------------------------------------" << std::endl;
#ifdef USE_32_BIT_GRAPH
    std::cout << "32-bit datatype" << std::endl;
#else
    std::cout << "64-bit datatype" << std::endl;
#endif
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "Total time (in s): " << total << std::endl;
    std::cout << "Modularity, #Iterations: " << currMod << ", " << iters << std::endl;
    std::cout << "MODS (final modularity * total time): " << (currMod * total) << std::endl;
    std::cout << "-------------------------------------------------------" << std::endl;

    delete g;

    return 0;
}

void parseCommandLine(const int argc, char * const argv[])
{
    int ret;

    while ((ret = getopt(argc, argv, "f:t:n:wlp:s")) != -1) {
        switch (ret) {
        case 'f':
            inputFileName.assign(optarg);
            break;
        case 't':
            threshold = atof(optarg);
            break;
        case 'n':
            nvRGG = atol(optarg);
            if (nvRGG > 0)
                generateGraph = true;
            break;
        case 'w':
            isUnitEdgeWeight = false;
            break;
        case 'l':
            randomNumberLCG = true;
            break;
        case 'p':
            randomEdgePercent = atof(optarg);
            break;
        case 's':
            showGraph = true;
            break;
        default:
            assert(0 && "Option not recognized!!!");
            break;
        }
    }

    if ((argc == 1)) {
        std::cerr << "Must specify some options." << std::endl;
        exit(-1);
    }

    if (!generateGraph && inputFileName.empty()) {
        std::cerr << "Must specify a binary file name with -f or provide parameters for generating a graph." << std::endl;
        exit(-1);
    }

    if (!generateGraph && randomNumberLCG) {
        std::cerr << "Must specify -n for graph generation using LCG." << std::endl;
        exit(-1);
    }

    if (!generateGraph && (randomEdgePercent > 0.0)) {
        std::cerr << "Must specify -n for graph generation first to add random edges to it." << std::endl;
        exit(-1);
    }

    if (!generateGraph && !isUnitEdgeWeight) {
        std::cerr << "Must specify -n for graph generation first before setting edge weights." << std::endl;
        exit(-1);
    }

    if (generateGraph && ((randomEdgePercent < 0) || (randomEdgePercent >= 100))) {
        std::cerr << "Invalid random edge percentage for generated graph!" << std::endl;
        exit(-1);
    }
}
