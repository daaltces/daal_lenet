/* file: daal_lenet.h */
/*
//               INTEL CORPORATION PROPRIETARY INFORMATION
//  This software is supplied under the terms of a license agreement or
//  nondisclosure agreement with Intel Corporation and may not be copied
//  or disclosed except in accordance with the terms of that agreement.
//    Copyright (C) 2014-2016 Intel Corporation. All Rights Reserved.
*/

#include "daal.h"

using namespace daal;
using namespace daal::algorithms;
using namespace daal::data_management;
using namespace daal::algorithms::neural_networks;
using namespace daal::algorithms::neural_networks::layers;
using namespace daal::services;

typedef services::SharedPtr<Tensor> TensorPtr;

typedef initializers::uniform::Batch<> UniformInitializer;
typedef SharedPtr<UniformInitializer> UniformInitializerPtr;
typedef initializers::xavier::Batch<> XavierInitializer;
typedef SharedPtr<XavierInitializer> XavierInitializerPtr;

training::TopologyPtr configureNet()
{
    /*Create convolution layer*/
    SharedPtr<convolution2d::Batch<> > convolution1(new convolution2d::Batch<>() );
    convolution1->parameter.kernelSizes = convolution2d::KernelSizes(3, 3);
    convolution1->parameter.strides = convolution2d::Strides(1, 1);
    convolution1->parameter.nKernels = 32;
    convolution1->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    convolution1->parameter.biasesInitializer = UniformInitializerPtr(new UniformInitializer(0, 0));

    /*Create pooling layer*/
    SharedPtr<maximum_pooling2d::Batch<> > maxpooling1(new maximum_pooling2d::Batch<>(4));
    maxpooling1->parameter.kernelSizes = pooling2d::KernelSizes(2, 2);
    maxpooling1->parameter.paddings = pooling2d::Paddings(0, 0);
    maxpooling1->parameter.strides = pooling2d::Strides(2, 2);

    /*Create convolution layer*/
    SharedPtr<convolution2d::Batch<> > convolution2(new convolution2d::Batch<>());
    convolution2->parameter.kernelSizes = convolution2d::KernelSizes(5, 5);
    convolution2->parameter.strides = convolution2d::Strides(1, 1);
    convolution2->parameter.nKernels = 64;
    convolution2->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    convolution2->parameter.biasesInitializer = UniformInitializerPtr(new UniformInitializer(0, 0));

    /*Create pooling layer*/
    SharedPtr<maximum_pooling2d::Batch<> > maxpooling2(new maximum_pooling2d::Batch<>(4));
    maxpooling2->parameter.kernelSizes = pooling2d::KernelSizes(2, 2);
    maxpooling2->parameter.paddings = pooling2d::Paddings(0, 0);
    maxpooling2->parameter.strides = pooling2d::Strides(2, 2);

    /*Create fullyconnected layer*/
    SharedPtr<fullyconnected::Batch<> > fullyconnected3(new fullyconnected::Batch<>(256));
    fullyconnected3->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    fullyconnected3->parameter.biasesInitializer = UniformInitializerPtr(new UniformInitializer(0, 0));

    /*Create ReLU layer*/
    SharedPtr<relu::Batch<> > relu3(new relu::Batch<>);

    /*Create fully connected layer*/
    SharedPtr<fullyconnected::Batch<> > fullyconnected4(new fullyconnected::Batch<>(10));
    fullyconnected4->parameter.weightsInitializer = XavierInitializerPtr(new XavierInitializer());
    fullyconnected4->parameter.biasesInitializer = UniformInitializerPtr(new UniformInitializer(0, 0));

    /*Create Softmax layer*/
    SharedPtr<loss::softmax_cross::Batch<> > softmax(new loss::softmax_cross::Batch<>());

    /*Create LeNet Topology*/
    training::TopologyPtr topology(new training::Topology());
    const size_t conv1 = topology->add(convolution1);
    const size_t pool1 = topology->add(maxpooling1);  topology->get(conv1).addNext(pool1);
    const size_t conv2 = topology->add(convolution2); topology->get(pool1).addNext(conv2);
    const size_t pool2 = topology->add(maxpooling2); topology->get(conv2).addNext(pool2);
    const size_t fc3 = topology->add(fullyconnected3); topology->get(pool2).addNext(fc3);
    const size_t r3 = topology->add(relu3); topology->get(fc3).addNext(r3);
    const size_t fc4 = topology->add(fullyconnected4); topology->get(r3).addNext(fc4);
    const size_t sm1 = topology->add(softmax); topology->get(fc4).addNext(sm1);
    return topology;
}
