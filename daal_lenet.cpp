/* file: daal_lenet.cpp */
/*
//               INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in accordance with the terms of that agreement.
//    Copyright (C) 2014-2016 Intel Corporation. All Rights Reserved.
*/

#include "daal_lenet.h"
#include "service.h"
#include "image_dataset.h"
#include <cmath>
#include <iostream>

using namespace std;

void train();
void test();
bool checkResult();

TensorPtr _trainingData;
TensorPtr _trainingGroundTruth;
TensorPtr _testingData;
TensorPtr _testingGroundTruth;
size_t TrainDataCount = 50000;
size_t TestDataCount = 100;

prediction::ModelPtr _predictionModel;
prediction::ResultPtr _predictionResult;

const string datasetFileNames[] =
{
    "./data/train-images-idx3-ubyte",
    "./data/train-labels-idx1-ubyte",
    "./data/t10k-images-idx3-ubyte",
    "./data/t10k-labels-idx1-ubyte"
};

int main(int argc, char *argv[])
{

    checkArguments(argc, argv, 4, &datasetFileNames[0], &datasetFileNames[1], &datasetFileNames[2], &datasetFileNames[3]);

    printf("Data loading started... \n");

    DatasetReader_MNIST<double> reader;
    reader.setTrainBatch(datasetFileNames[0], datasetFileNames[1], TrainDataCount);
    reader.setTestBatch(datasetFileNames[2], datasetFileNames[3], TestDataCount);
    reader.read();

    printf("Data loaded \n");

    _trainingData = reader.getTrainData();
    _trainingGroundTruth = reader.getTrainGroundTruth();
    _testingData = reader.getTestData();
    _testingGroundTruth = reader.getTestGroundTruth();

    printf("LeNet training started... \n");

    train();

    printf("LeNet training completed \n");
    printf("LeNet testing started \n");

    test();

    if (checkResult())
    {
        return 0;
    }
    else
    {
        return -1;
    }
}

/*LeNet training*/
void train()
{
    const size_t _batchSize = 10;
    double learningRate = 0.01;

    SharedPtr<optimization_solver::sgd::Batch<float> > sgdAlgorithm(new optimization_solver::sgd::Batch<float>());
    (*(HomogenNumericTable<double>::cast(sgdAlgorithm->parameter.learningRateSequence)))[0][0] = learningRate;

    training::TopologyPtr topology = configureNet();

    training::Batch<> net;

    net.parameter.batchSize = _batchSize;
    net.parameter.optimizationSolver = sgdAlgorithm;

    net.initialize(_trainingData->getDimensions(), *topology);

    net.input.set(training::data, _trainingData);
    net.input.set(training::groundTruth, _trainingGroundTruth);
    net.compute();

    _predictionModel = net.getResult()->get(training::model)->getPredictionModel<double>();
}

/*LeNet testing*/
void test()
{
    prediction::Batch<> net;

    net.input.set(prediction::model, _predictionModel);
    net.input.set(prediction::data, _testingData);

    net.compute();

    _predictionResult = net.getResult();

    printPredictedClasses(_predictionResult, _testingGroundTruth);
}

/*check prediction results*/
bool checkResult()
{
    TensorPtr prediction = _predictionResult->get(prediction::prediction);
    const Collection<size_t> &predictionDimensions = prediction->getDimensions();

    SubtensorDescriptor<double> predictionBlock;
    prediction->getSubtensor(0, 0, 0, predictionDimensions[0], readOnly, predictionBlock);
    double *predictionPtr = predictionBlock.getPtr();

    SubtensorDescriptor<int> testGroundTruthBlock;
    _testingGroundTruth->getSubtensor(0, 0, 0, predictionDimensions[0], readOnly, testGroundTruthBlock);
    int *testGroundTruthPtr = testGroundTruthBlock.getPtr();
    size_t maxPIndex = 0;
    size_t trueCount = 0;

    /*validation accuracy finding*/
    for (size_t i = 0; i < predictionDimensions[0]; i++)
    {
        double maxP = 0;
        maxPIndex = 0;
        for (size_t j = 0; j < predictionDimensions[1]; j++)
        {
            double p = predictionPtr[i * predictionDimensions[1] + j];
            if (maxP < p)
            {
                maxP = p;
                maxPIndex = j;
            }
        }
        if ( maxPIndex == testGroundTruthPtr[i] )
        trueCount ++;
    }

    prediction->releaseSubtensor(predictionBlock);
    _testingGroundTruth->releaseSubtensor(testGroundTruthBlock);

    if ( (double)trueCount / (double)TestDataCount > 0.9 )
    {
        return true;
    }
    else
    {
        return false;
    }
}
