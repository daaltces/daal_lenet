/* file: image_dataset.h */
/*
//               INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in accordance with the terms of that agreement.
//    Copyright (C) 2014-2016 Intel Corporation. All Rights Reserved.
*/

#include <vector>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include "daal.h"

using namespace daal;
using namespace daal::services;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;

class DatasetReader
{
public:
    DatasetReader() { }
    virtual ~DatasetReader() { }

    virtual void read() = 0;
    virtual SharedPtr<Tensor> getTrainData() = 0;
    virtual SharedPtr<Tensor> getTrainGroundTruth() = 0;
    virtual SharedPtr<Tensor> getTestData() = 0;
    virtual SharedPtr<Tensor> getTestGroundTruth() = 0;
};


template<typename FPType>
class RGBChannelNormalizer
{
public:
    inline FPType operator()(FPType value) { return value / (FPType)255; }
};

template<typename FPType>
class DummyNormalizer
{
public:
    inline FPType operator()(FPType value) { return value; }
};


template<typename FPType, typename Normalizer = RGBChannelNormalizer<FPType> >
class ImageDatasetReader
{
public:
    size_t numberOfChannels;
    size_t objectHeight;
    size_t objectWidth;

protected:

    Normalizer _normalizer;
    SharedPtr<HomogenTensor<FPType> > _trainData;
    SharedPtr<HomogenTensor<FPType> > _trainGroundTruth;
    SharedPtr<HomogenTensor<FPType> > _testData;
    SharedPtr<HomogenTensor<FPType> > _testGroundTruth;

public:

    virtual ~ImageDatasetReader() { }

    virtual SharedPtr<Tensor> getTrainData() { return _trainData; }
    virtual SharedPtr<Tensor> getTrainGroundTruth() { return _trainGroundTruth; }
    virtual SharedPtr<Tensor> getTestData() { return _testData; }
    virtual SharedPtr<Tensor> getTestGroundTruth() { return _testGroundTruth; }

protected:

    ImageDatasetReader(size_t channelsNum, size_t height, size_t width) :
        numberOfChannels(channelsNum),
        objectHeight(height),
        objectWidth(width) { }

    virtual void allocateTensors()
    {
        size_t numberOfObjects = getNumberOfTrainObjects();
        size_t numberOfTestObjects = getNumberOfTestObjects();

        if (numberOfObjects > 0)
        {
            Collection<size_t> trainDataDims;
            trainDataDims.push_back(numberOfObjects);
            trainDataDims.push_back(numberOfChannels);
            trainDataDims.push_back(objectHeight);
            trainDataDims.push_back(objectWidth);
            _trainData = SharedPtr<HomogenTensor<FPType> >(
                             new HomogenTensor<FPType>(trainDataDims, Tensor::doAllocate, (FPType)0));

            Collection<size_t> trainGroundTruthDims;
            trainGroundTruthDims.push_back(numberOfObjects);
            _trainGroundTruth = SharedPtr<HomogenTensor<FPType> >(
                                    new HomogenTensor<FPType>(trainGroundTruthDims, Tensor::doAllocate));
        }

        if (numberOfTestObjects > 0)
        {
            Collection<size_t> testDataDims;
            testDataDims.push_back(numberOfTestObjects);
            testDataDims.push_back(numberOfChannels);
            testDataDims.push_back(objectHeight);
            testDataDims.push_back(objectWidth);
            _testData = SharedPtr<HomogenTensor<FPType> >(
                            new HomogenTensor<FPType>(testDataDims, Tensor::doAllocate, (FPType)0));

            Collection<size_t> testGroundTruthDims;
            testGroundTruthDims.push_back(numberOfTestObjects);
            _testGroundTruth = SharedPtr<HomogenTensor<FPType> >(
                                   new HomogenTensor<FPType>(testGroundTruthDims, Tensor::doAllocate));
        }
    }

    virtual size_t getNumberOfTrainObjects() = 0;
    virtual size_t getNumberOfTestObjects() = 0;

    void normalizeBuffer(const uint8_t *buffer, FPType *normalized, size_t bufferSize)
    {
        for (size_t i = 0; i < bufferSize; i++)
        {
            normalized[i] = _normalizer((FPType)buffer[i]);
        }
    }

    inline size_t tensorOffset(size_t n, size_t k = 0, size_t h = 0, size_t w = 0)
    {
        return
            n * numberOfChannels * objectHeight * objectWidth +
            k * objectHeight * objectWidth +
            h * objectWidth +
            w;
    }

private:

    ImageDatasetReader() { }

};

template<typename FPType, typename Normalizer = RGBChannelNormalizer<FPType> >
class DatasetReader_MNIST : public ImageDatasetReader<FPType, Normalizer>
{
private:

    const int DATA_MAGIC_NUMBER = 0x00000803;
    const int LABELS_MAGIC_NUMBER = 0x00000801;

    std::string _trainPathData;
    std::string _trainPathLabels;
    std::string _testPathData;
    std::string _testPathLabels;
    size_t _numOfTrainObjects;
    size_t _numOfTestObjects;

public:

    size_t originalObjectHeight;
    size_t originalObjectWidth;
    size_t margins;

public:

    DatasetReader_MNIST(size_t margin = 0) : ImageDatasetReader<FPType, Normalizer>(1, 28 + 2 * margin, 28 + 2 * margin),
        _numOfTrainObjects(0), _numOfTestObjects(0),
        originalObjectWidth(28), originalObjectHeight(28), margins(margin) { }

    virtual ~DatasetReader_MNIST() { }

    inline void setTrainBatch(std::string pathToBatchData, std::string pathToBatchlabels, size_t numOfObjects)
    {
        _trainPathData = std::move(pathToBatchData);
        _trainPathLabels = std::move(pathToBatchlabels);
        _numOfTrainObjects = numOfObjects;
    }

    inline void setTestBatch(std::string pathToBatchData, std::string pathToBatchLabels, size_t numOfObjects)
    {
        _testPathData = std::move(pathToBatchData);
        _testPathLabels = std::move(pathToBatchLabels);
        _numOfTestObjects = numOfObjects;
    }

    virtual void read()
    {
        this->objectWidth = originalObjectWidth + 2 * margins;
        this->objectHeight = originalObjectHeight + 2 * margins;
        this->allocateTensors();

        if (_numOfTrainObjects)
        {
            readBatchDataFile(_trainPathData, this->_trainData, _numOfTrainObjects);
            readBatchLabelsFile(_trainPathLabels, this->_trainGroundTruth, _numOfTrainObjects);
        }

        if (_numOfTestObjects > 0)
        {
            readBatchDataFile(_testPathData, this->_testData, _numOfTestObjects);
            readBatchLabelsFile(_testPathLabels, this->_testGroundTruth, _numOfTestObjects);
        }
    }

protected:

    virtual size_t getNumberOfTrainObjects() { return _numOfTrainObjects; }
    virtual size_t getNumberOfTestObjects() { return _numOfTestObjects; }

private:

    void readBatchDataFile(const std::string &batchPath, SharedPtr<HomogenTensor<FPType> > data, size_t numOfObjects)
    {
        std::ifstream batchStream(batchPath.c_str(), std::ifstream::in | std::ifstream::binary);
        FPType *dataRaw = data->getArray();
        readDataBatch(batchStream, dataRaw, numOfObjects);
        batchStream.close();
    }

    void readBatchLabelsFile(const std::string &batchPath, SharedPtr<HomogenTensor<FPType> > labels, size_t numOfObjects)
    {
        std::ifstream batchStream(batchPath.c_str(), std::ifstream::in | std::ifstream::binary);
        FPType *labelsRaw = labels->getArray();
        readLabelsBatch(batchStream, labelsRaw, numOfObjects);
        batchStream.close();
    }

    void readDataBatch(std::ifstream &stream, FPType *tensorData, size_t numOfObjects)
    {
        uint32_t magicNumber = readDword(stream);
        if (magicNumber != DATA_MAGIC_NUMBER)
        {
            throw std::runtime_error("Invalid data file format");
        }

        uint32_t numberOfImages = readDword(stream);
        if (numberOfImages < numOfObjects)
        {
            throw std::runtime_error("Number of objects too large");
        }

        uint32_t numberOfRows = readDword(stream);
        if (numberOfRows != originalObjectWidth)
        {
            throw std::runtime_error("Batch contains invalid images");
        }

        uint32_t numberOfColumns = readDword(stream);
        if (numberOfColumns != originalObjectHeight)
        {
            throw std::runtime_error("Batch contains invalid images");
        }

        size_t bufferSize = originalObjectWidth * originalObjectHeight;
        uint8_t *channelBuffer = new uint8_t[bufferSize];

        FPType *tensorDataPtr;
        for (size_t objectCounter = 0; objectCounter < numOfObjects && stream.good(); objectCounter++)
        {
            stream.read((char *)channelBuffer, bufferSize);
            tensorDataPtr = tensorData + this->tensorOffset(objectCounter);
            tensorDataPtr += margins * this->objectWidth;
            for (size_t i = 0; i < originalObjectHeight; i++)
            {
                tensorDataPtr += margins;
                this->normalizeBuffer(channelBuffer + i * originalObjectWidth, tensorDataPtr, originalObjectWidth);
                tensorDataPtr += originalObjectWidth + margins;
            }
        }

        delete[] channelBuffer;
    }

    void readLabelsBatch(std::ifstream &stream, FPType *labelsData, size_t numOfObjects)
    {
        uint32_t magicNumber = readDword(stream);
        if (magicNumber != LABELS_MAGIC_NUMBER)
        {
            throw std::runtime_error("Invalid data file format");
        }

        uint32_t numberOfItems = readDword(stream);
        if (numberOfItems < numOfObjects)
        {
            throw std::runtime_error("Number of objects too large");
        }

        char classNumber;
        for (size_t objectCounter = 0; objectCounter < numOfObjects && stream.good(); objectCounter++)
        {
            stream.get(classNumber);
            labelsData[objectCounter] = (FPType)classNumber;
        }
    }

    inline uint32_t readDword(std::ifstream &stream)
    {
        uint32_t dword;
        stream.read((char *)(&dword), sizeof(uint32_t));
        return endianDwordConversion(dword);
    }

    inline uint32_t endianDwordConversion(uint32_t dword)
    {
        return
            ((dword >> 24) & 0x000000FF) |
            ((dword >>  8) & 0x0000FF00) |
            ((dword <<  8) & 0x00FF0000) |
            ((dword << 24) & 0xFF000000);
    }

};
