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
    /** EXERCISE 1: Your code here!
     * Create a convolution layer (name it convolution2) using convolution2d::Batch<>(). The layer
     * configuration is as follows:
     *  - The convolution kernel size is 5-by-5.
     *  - A total of 64 kernels are applied to the data at this layer.
     *  - Use the unit convolution stride on each dimension.
     *  - Use the Xavier initializer to initiate weights.
     *  - Use the Uniform initializer to initiate biases.
     */

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

    /** EXERCISE 2: Your code here!
     * Add the other layers to the topology, according to the following specifications:
     *  - Connect 'maxpooling2' to 'convolution2'.
     *  - Connect 'fullyconnected3'to 'maxpooling2'.
     *  - Connect 'relu3' to 'fullyconnected3'.
     *  - Connect 'fullyconnected4' to 'relu3'.
     *  - Connect 'softmax' to "fullyconnected4'
     */

    return topology;
}
