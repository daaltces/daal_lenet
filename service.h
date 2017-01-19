/* file: service.h */
/*
//               INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in accordance with the terms of that agreement.
//    Copyright (C) 2014-2016 Intel Corporation. All Rights Reserved.
*/

#include "daal.h"

using namespace daal::data_management;

#include <cmath>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdarg>
#include <vector>
#include <queue>

#include "error_handling.h"

using namespace std;

void printPredictedClasses(SharedPtr<prediction::Result> _predictionResult, TensorPtr _testingGroundTruth);
void printWeights(SharedPtr<prediction::Model> _predictionModel);
void printTensorAsArray(const TensorPtr &tensor, size_t size = 0);
void printTensorAsArray(const TensorPtr &tensor, size_t m, size_t n, size_t offset = 0);
bool checkFileIsAvailable(std::string filename, bool needExit = false);
void checkArguments(int argc, char *argv[], int count, ...);

void printPredictedClasses(SharedPtr<prediction::Result> _predictionResult, TensorPtr _testingGroundTruth)
{
    TensorPtr prediction = _predictionResult->get(prediction::prediction);
    const Collection<size_t> &predictionDimensions = prediction->getDimensions();

    SubtensorDescriptor<double> predictionBlock;
    prediction->getSubtensor(0, 0, 0, predictionDimensions[0], readOnly, predictionBlock);
    double *predictionPtr = predictionBlock.getPtr();

    SubtensorDescriptor<int> testGroundTruthBlock;
    _testingGroundTruth->getSubtensor(0, 0, 0, predictionDimensions[0], readOnly, testGroundTruthBlock);
    int *testGroundTruthPtr = testGroundTruthBlock.getPtr();

    // Print predicted classes
    for (size_t i = 0; i < predictionDimensions[0]; i++)
    {
        double maxP = 0;
        size_t maxPIndex = 0;
        for (size_t j = 0; j < predictionDimensions[1]; j++)
        {
            double p = predictionPtr[i * predictionDimensions[1] + j];
            if (maxP < p)
            {
                maxP = p;
                maxPIndex = j;
            }
            printf("%.4f ", p);
        }
        fflush(stdout);

        printf(" -> %d | %d\n", maxPIndex, testGroundTruthPtr[i]);
        fflush(stdout);
    }

    prediction->releaseSubtensor(predictionBlock);
    _testingGroundTruth->releaseSubtensor(testGroundTruthBlock);
}

void printWeights(SharedPtr<prediction::Model> _predictionModel)
{
    SharedPtr<neural_networks::ForwardLayers> forwardLayers = _predictionModel->getLayers();
    for (size_t i = 0; i < forwardLayers->size(); i++)
    {
        SharedPtr<neural_networks::layers::forward::LayerIface> layer = forwardLayers->get(i);
        TensorPtr weights = layer->getLayerInput()->get(neural_networks::layers::forward::weights);
        TensorPtr biases = layer->getLayerInput()->get(neural_networks::layers::forward::biases);

        if (weights && weights->getSize())
        {
            printf("Layer %d weights:\n", i);
            printTensorAsArray(weights);
        }

        if (biases && biases->getSize())
        {
            printf("Layer %d biases:\n", i);
            printTensorAsArray(biases);
        }
    }
}

void printTensorAsArray(const TensorPtr &tensor, size_t size)
{
    SubtensorDescriptor<double> tensorBlock;
    tensor->getSubtensor(0, 0, 0, tensor->getDimensionSize(0), readOnly, tensorBlock);
    double *tensorPtr = tensorBlock.getPtr();

    if (size > 0)
    {
        size = std::min(tensor->getSize(), size);
    }

    for (size_t i = 0; i < size; i++)
    {
        printf("%.4f ", tensorPtr[i]);
    }
    printf("\n");
    fflush(stdout);

    tensor->releaseSubtensor(tensorBlock);
}

void printTensorAsArray(const TensorPtr &tensor, size_t m, size_t n, size_t offset)
{
    SubtensorDescriptor<double> tensorBlock;
    tensor->getSubtensor(0, 0, 0, tensor->getDimensionSize(0), readOnly, tensorBlock);
    double *tensorPtr = tensorBlock.getPtr() + offset * sizeof(double);

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < n; j++)
        {
            std::cout << tensorPtr[i * n + j] << ", ";
        }
        printf("\n");
    }
    fflush(stdout);

    tensor->releaseSubtensor(tensorBlock);
}

bool checkFileIsAvailable(std::string filename, bool needExit)
{
    std::ifstream file(filename.c_str());
    if(file.good())
    {
        return true;
    }
    else
    {
        std::cout << "Can't open file " << filename << std::endl;
        if(needExit)
        {
            exit(fileError);
        }
        return false;
    }
}

void checkArguments(int argc, char *argv[], int count, ...)
{
    std::string **filelist = new std::string*[count];
    va_list ap;
    va_start(ap, count);
    for (int i = 0; i < count; i++)
    {
        filelist[i] = va_arg(ap, std::string *);
    }
    va_end(ap);
    if(argc == 1)
    {
        for (int i = 0; i < count; i++)
        {
            checkFileIsAvailable(*(filelist[i]), true);
        }
    }
    else if(argc == (count + 1))
    {
        bool isAllCorrect = true;
        for (int i = 0; i < count; i++)
        {
            if(!checkFileIsAvailable(argv[i + 1]))
            {
                isAllCorrect = false;
                break;
            }
        }
        if(isAllCorrect == true)
        {
            for(int i = 0; i < count; i++)
            {
                (*filelist[i]) = argv[i + 1];
            }
        }
        else
        {
            std::cout << "Warning: Try to open default datasetFileNames" << std::endl;
            for (int i = 0; i < count; i++)
            {
                checkFileIsAvailable(*(filelist[i]), true);
            }
        }
    }
    else
    {
        std::cout << "Usage: " << argv[0] << " [ ";
        for(int i = 0; i < count; i++)
        {
            std::cout << "<filename_" << i << "> ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Warning: Try to open default datasetFileNames" << std::endl;
        for (int i = 0; i < count; i++)
        {
            checkFileIsAvailable(*(filelist[i]), true);
        }
    }
    delete [] filelist;
}
