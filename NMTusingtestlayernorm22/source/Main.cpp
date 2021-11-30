/* NiuTrans.NMT - an open-source neural machine translation system.
 * Copyright (C) 2020 NiuTrans Research. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-10
 */

//#define CRTDBG_MAP_ALLOC
//#include <stdlib.h>
//#include <crtdbg.h>

#include "niutensor/tensor/core/CHeader.h"
#include "nmt/quantize/quantize.h"
#include "./nmt/NMT.h"
#include "niutensor/network/XNoder.h"
#include "niutensor/tensor/XTensor.h"
#include "niutensor/tensor/core/movement/Spread.h"
#include <iostream>
#include <chrono>
using namespace nmt;
using namespace nts;
using namespace std;

void outall(XTensor input) {
    if (input.dataType == X_FLOAT) {
        float* data = (float*)input.data;
        int size = input.dimSize[input.order - 1];
        for (int i = 0; i < input.unitNum / size; i++) {
            for (int j = 0; j < size; j++) {
                cout << data[i * size + j] << " ";
            }
            cout << endl;
        }
    }
    else if (input.dataType == X_INT8) {
        int_8* data = (int_8*)input.data;
        int size = input.dimSize[input.order - 1];
        for (int i = 0; i < input.unitNum / size; i++) {
            for (int j = 0; j < size; j++) {
                cout << data[i * size + j]-0<< " ";
            }
            cout << endl;
        }
    }
    
}
void outtensor(XTensor input) {
    if (input.dataType == X_FLOAT) {
        float* data = (float*)input.data;
        for (int i = 0; i < 5; i++) {
            cout << data[i]<<" ";
        }
        cout << endl;
        for (int i = 5; i > 0; i--) {
            cout << data[input.unitNum - i] << " ";
        }
        cout << endl;
    }
    if (input.dataType == X_INT8) {
        int_8* data = (int_8*)input.data;
        for (int i = 0; i < 5; i++) {
            cout << data[i]-0 << " ";
        }
        cout << endl;
        for (int i = 5; i > 0; i--) {
            cout << data[input.unitNum - i]-0<< " ";
        }
        cout << endl;
    }
}

void testSum() {
    XTensor input1, input2, out;
    InitTensor3D(&input1, 3, 6, 50);
    InitTensor3D(&input2, 3, 6, 50);
    InitTensor3D(&out, 3, 6, 50);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    outtensor(input1);
    lower = -1.0;
    higher = 1.0;
    _SetDataRand(&input2, lower, higher);
    outtensor(input2);
    cout << "Sum:" << endl;
    auto beginTime = std::chrono::high_resolution_clock::now();
    _Sum(&input1, &input2, &out);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime);
    double programTimes = ((double)elapsedTime.count());
    LOG("reduceMeanand scale 1.0 elapsed %lf", programTimes);
    outtensor(out);

    quantize(&input1);
    quantize(&input2);
    XTensor out2(input1);
    cout << "quantized Sum:" << endl;
    auto beginTime2 = std::chrono::high_resolution_clock::now();
    _Sum(&input1, &input2, &out2);
    auto endTime2 = std::chrono::high_resolution_clock::now();
    auto elapsedTime2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - beginTime2);
    double programTimes2 = ((double)elapsedTime2.count());
    LOG("reduceMeanand scale 1.0 elapsed %lf", programTimes2);
    dequantize(&out2);
    outtensor(out2);
}
void testquantizeincol() {
    XTensor input1;
    InitTensor2D(&input1, 32, 4);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    outtensor(input1);
    quantize(&input1);
    cout << "quantize and dequantize:" << endl;
    dequantize(&input1);
    outtensor(input1);
    quantizeMatrix2DinCol(&input1);
    cout << "quantizeincol ande dequantize:" << endl;
    dequantizeMatrix2DinCol(&input1);
    outtensor(input1);
}

void testMatrix() {
    XTensor input1, input2, out, out2;
    int size=1300;
    InitTensor2D(&input1, 121, 512);
    InitTensor2D(&input2, 512, 512);
    InitTensor2D(&out, 121, 512);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    lower = -1.0;
    higher = 1.0;
    _SetDataRand(&input2, lower, higher);
    outtensor(input1);
    outtensor(input2);
    auto beginTime = std::chrono::high_resolution_clock::now();
    _MatrixMul2D(&input1, X_NOTRANS, &input2, X_NOTRANS, &out);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime);
    double programTimes = ((double)elapsedTime.count());
    LOG("reduceMeanand scale 1.0 elapsed %lf", programTimes);
    cout << "matrix mul" << endl;
    outtensor(out);

    nmt::quantize(&input1);
    nmt::quantize(&input2);
    //XTensor out(input1);
    InitTensor2D(&out2, 121, 512);
    int dimsize[1] = { size };
    out2.scale.initScale(1, dimsize);
    out2 = ConvertDataType(out, X_INT8);
    auto beginTime2 = std::chrono::high_resolution_clock::now();
    _MatrixMul2D(&input1, X_NOTRANS, &input2, X_NOTRANS, &out2);
    auto endTime2 = std::chrono::high_resolution_clock::now();
    auto elapsedTime2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - beginTime2);
    double programTimes2 = ((double)elapsedTime2.count());
    LOG("reduceMeanand scale 1.0 elapsed %lf", programTimes2);
    cout << endl;
    cout << "dequantize" << endl;
    nmt::dequantize(&out2);
    outtensor(out2);
    cout << "加速比:" << (programTimes - programTimes2) / programTimes;
}

void testreducemeanandscale() {
    XTensor input1,mean,mean2;
    InitTensor3D(&input1,2,3,4);
    float lower = -10.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    outall(input1);
    mean = ReduceMean(input1, input1.order - 1);
    cout << "mean:" << endl;
    outtensor(mean);
    nmt::quantize(&input1);
    mean2 = ReduceMeanandScale_int8(input1, input1.order - 1,NULL, 1.0);
    cout << endl<<"quantized mean:" << endl;
    nmt::dequantize(&mean2);
    outtensor(mean2);
}
void testl1layer() {
    XTensor x,mean,mean2;
    InitTensor3D(&x, 2, 3, 4);
    float lower = -10.0;
    float higher = 1.0;
    _SetDataRand(&x, lower, higher);
    outall(x);
    mean = ReduceMean(x, x.order - 1);
    cout << endl << "mean:" << endl;
    outall(mean);
    nmt::quantize(&x);
    mean2 = ReduceMean_int8(x, x.order - 1, NULL, 1.0);
    float dr = (!x.isSparse) ? 1.0F : x.denseRatio;
    XTensor meanFilled(x.order, x.dimSize, x.dataType, dr, x.devID, x.mem);
    meanFilled.enableGrad = x.enableGrad;
    meanFilled.SetTMPFlag();
    XTensor l1Filled(x.order, x.dimSize, x.dataType, dr, x.devID, x.mem);
    l1Filled.enableGrad = x.enableGrad;
    l1Filled.SetTMPFlag();

    int stride = x.dimSize[x.order - 1];
    int num = 1;
    int_8* xdata = (int_8*)x.data;
    int_8* data = (int_8*)meanFilled.data;
    int_8* meandata = (int_8*)mean2.data;
    int* absdata = new int[x.unitNum];
    float sum;
    for (int i = 0; i < x.order - 1; i++) {
        num *= x.dimSize[i];
    }
    for (int i = 0; i < num; i++) {
        sum = 0;
        for (int j = 0; j < stride; j++) {
            data[i * stride + j] = xdata[i * stride + j] - meandata[i];
            sum = sum + abs(data[i * stride + j]);
        }
        sum = sum / stride;
        for (int j = 0; j < stride; j++) {
            absdata[i * stride + j] = round(sum);
        }
    }
    l1Filled.SetDataInt8(absdata, l1Filled.unitNum);
    meanFilled.initScale(x.scale.order, x.scale.dimsize, x.scale.values);
    l1Filled.initScale(x.scale.order, x.scale.dimsize, x.scale.values);
    delete[] absdata;
    dequantize(&l1Filled);
    dequantize(&meanFilled);
    cout << endl << "meanfilled" << endl;
    outall(meanFilled);
    cout << endl << "f1filled" << endl;
    outall(l1Filled);
}
void testMultiply() {
    XTensor input1, input2, out,out2;
    InitTensor2D(&input1, 100, 500);
    InitTensor2D(&input2, 100, 500);
    //InitTensor3D(&out, 2, 3, 4);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    outtensor(input1);
    lower = -1.0;
    higher = 1.0;
    _SetDataRand(&input2, lower, higher);
    outtensor(input2);
    auto beginTime = std::chrono::high_resolution_clock::now();
    out = input1 * input2;
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime);
    double programTimes = ((double)elapsedTime.count());
    cout << "mul" << endl;
    outtensor(out);
    quantize(&input1);
    quantize(&input2);
    auto beginTime2 = std::chrono::high_resolution_clock::now();
    out2 = input1 * input2;
    
    auto endTime2 = std::chrono::high_resolution_clock::now();
    auto elapsedTime2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - beginTime2);
    double programTimes2 = ((double)elapsedTime2.count());
    cout << "加速比:" << (programTimes - programTimes2) / programTimes<<endl;
    cout << "quantized mul" << endl;
    dequantize(&out2);
    outtensor(out2);
    
}
void testMultiplydim() {
    XTensor input1, input2, out, out2;
    InitTensor3D(&input1, 2, 3, 4);
    InitTensor1D(&input2,4);
    //InitTensor3D(&out, 2, 3, 4);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    outtensor(input1);
    lower = -3.0;
    higher = 1.0;
    _SetDataRand(&input2, lower, higher);
    outtensor(input2);
    out = input1 * input2;
    cout << "mul" << endl;
    outtensor(out);
    quantize(&input1);
    quantize(&input2);
    out2 = input1 * input2;
    cout << "quantized mul" << endl;
    dequantize(&out2);
    outtensor(out2);
}

void testdiv() {
    XTensor input1, input2, out, out2;
    InitTensor3D(&input1, 2, 3, 4);
    InitTensor3D(&input2, 2, 3, 4);
    //InitTensor3D(&out, 2, 3, 4);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    outtensor(input1);
    lower = -3.0;
    higher = 1.0;
    _SetDataRand(&input2, lower, higher);
    outtensor(input2);
    out = input1 / input2;
    cout << "div" << endl;
    outtensor(out);
    quantize(&input1);
    quantize(&input2);
    out2 = input1 / input2;
    cout << "quantized div" << endl;
    dequantize(&out2);
    outtensor(out2);
}
void testsumdim() {
    XTensor input1, input2, out, out2;
    InitTensor3D(&input1, 2, 3, 4);
    InitTensor1D(&input2, 4);
    //InitTensor3D(&out, 2, 3, 4);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    outtensor(input1);
    lower = -3.0;
    higher = 1.0;
    _SetDataRand(&input2, lower, higher);
    outtensor(input2);
    out = input1 +input2;
    cout << "sum" << endl;
    outtensor(out);
    quantize(&input1);
    quantize(&input2);
    out2 = input1 + input2;
    cout << "quantized sum" << endl;
    dequantize(&out2);
    outtensor(out2);
}
void testBatchMatrix() {
    XTensor input1, input2;
    int size = 1300;
    InitTensor3D(&input1, 2, 1300,1300);
    InitTensor3D(&input2, 2, 1300,1300);
    //InitTensor3D(&out, 2,3,5);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    lower = -3.0;
    higher = 1.0;
    _SetDataRand(&input2, lower, higher);
    outtensor(input1);
    outtensor(input2);

    int order = input1.order;
    int sub = 0;
    int* dimSize = new int[order];
    for (int i = 0; i < input1.order - 2; i++)
        dimSize[sub++] = input1.dimSize[i];
    dimSize[sub++] = input1.dimSize[input1.order-2];
    dimSize[sub++] = input2.dimSize[input2.order-1];

    float dr = 1.0F;
    XTensor out(order, dimSize, input1.dataType, dr, input1.devID, input1.mem);
    out.SetTMPFlag();
    if (out.dataType == X_INT8) {
        int* dim = new int[out.order - 1];
        for (int i = 0; i < out.order - 1; i++) {
            dim[i] = out.dimSize[i];
        }
        out.scale.initScale(out.order - 1, dim);
        delete[]dim;
    }
    auto beginTime = std::chrono::high_resolution_clock::now();
    _MatrixMulBatched(&input1, X_NOTRANS, &input2, X_NOTRANS,&out);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime);
    double programTimes = ((double)elapsedTime.count());
    LOG("reduceMeanand scale 1.0 elapsed %lf", programTimes);
    cout << "matrix mul" << endl;
    outtensor(out);

    nmt::quantize(&input1);
    nmt::quantize(&input2);
    //XTensor out(input1);
    //InitTensor3D(&out2, 2,3,5);
    //int dimsize[1] = { size };
    //out2.scale.initScale(1, dimsize);
    //out2 = ConvertDataType(out, X_INT8);
    sub = 0;
    int* dimSize2 = new int[order];
    for (int i = 0; i < input1.order - 2; i++)
        dimSize2[sub++] = input1.dimSize[i];
    dimSize2[sub++] = input1.order - 2;
    dimSize2[sub++] = input2.order - 1;

    XTensor out2(order, dimSize, input1.dataType, dr, input1.devID, input1.mem);
    out2.SetTMPFlag();
    if (out2.dataType == X_INT8) {
        int* dim = new int[out2.order - 1];
        for (int i = 0; i < out2.order - 1; i++) {
            dim[i] = out2.dimSize[i];
        }
        out2.scale.initScale(out2.order - 1, dim);
        delete[]dim;
    }
    auto beginTime2 = std::chrono::high_resolution_clock::now();
    _MatrixMulBatched(&input1, X_NOTRANS, &input2, X_NOTRANS,&out2);
    auto endTime2 = std::chrono::high_resolution_clock::now();
    auto elapsedTime2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - beginTime2);
    double programTimes2 = ((double)elapsedTime2.count());
    LOG("reduceMeanand scale 1.0 elapsed %lf", programTimes2);
    cout << endl;
    cout << "dequantize" << endl;
    nmt::dequantize(&out2);
    outtensor(out2);
    cout << "加速比:" << (programTimes - programTimes2) / programTimes;
}
void testscaleandshift() {
    XTensor input1, out;
    InitTensor2D(&input1, 128, 121);
    InitTensor2D(&out, 128, 121);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    outtensor(input1);
    auto beginTime = std::chrono::high_resolution_clock::now();
    _ScaleAndShift(&input1,&out, 3.0F, 2.5F);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime);
    double programTimes = ((double)elapsedTime.count());
    cout << "scale and shift" << endl;
    outtensor(out);
    quantize(&input1);
    XTensor out2(input1);
    auto beginTime2 = std::chrono::high_resolution_clock::now();
    _ScaleAndShift(&input1, &out2, 3.0F, 2.5F);
    auto endTime2 = std::chrono::high_resolution_clock::now();
    auto elapsedTime2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - beginTime2);
    double programTimes2 = ((double)elapsedTime2.count());
    cout << "加速比:" << (programTimes - programTimes2) / programTimes<<endl;
    cout << "quantized scale and shift" << endl;
    dequantize(&out2);
    outtensor(out2);
}
void testsplitandmerge() {
    XTensor input1, out,out2;
    InitTensor3D(&input1,5, 3, 8);
    InitTensor3D(&out,5, 3, 8);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    outall(input1);
    out = Split(&input1, input1.order - 1, 4);
    out = Merge(&out, out.order - 1);
    cout << "split and merge:" << endl;
    outall(out);
    quantize(&input1);
    XTensor out3(input1);
    dequantize(&out3);
    cout << "out3" << endl;
    outall (out3);
    outall(input1);
    out2 = Split(&input1, input1.order - 1, 2);
    outall(out2);
    out2 = Merge(&out2, out2.order - 1);
    cout << "quantized split and merge:" << endl;
    dequantize(&out2);
    outall(out2);

}
void testconcatenate() {
    XTensor input1, input2, out,out2;
    InitTensor3D(&input1, 2, 5, 4);
    InitTensor3D(&input2, 2, 3, 4);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    _SetDataRand(&input2, lower, higher);
    outall(input1);
    cout << endl;
    outall(input2);
    out = Concatenate(&input1, &input2, 1);
    cout << "concatenate:" << endl;
    outall(out);
    quantize(&input1);
    quantize(&input2);
    out2 = Concatenate(&input1, &input2, 1);
    dequantize(&out2);
    cout << "dequantized concatenate:" << endl;
    outall(out2);
    
}
void testquantize() {
    XTensor input1;
    InitTensor3D(&input1, 2, 5, 4);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    outtensor(input1);
    quantize(&input1);
    dequantize(&input1);
    outtensor(input1);
}
void testscale() {
    XTensor input1;
    InitTensor2D(&input1, 121,500);
    float lower = -1.0;
    float higher = 1.0;
    _SetDataRand(&input1, lower, higher);
    outtensor(input1);
    XTensor out1(input1);
    auto beginTime = std::chrono::high_resolution_clock::now();
    _Scale(&input1, &out1, 2.5);
    auto endTime = std::chrono::high_resolution_clock::now();
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime);
    double programTimes = ((double)elapsedTime.count());
    LOG("reduceMeanand scale 1.0 elapsed %lf", programTimes);
    outtensor(out1);
    quantize(&input1);
    XTensor out2(input1);
    auto beginTime2 = std::chrono::high_resolution_clock::now();
    _Scale(&input1, &out2,2.5);
    auto endTime2 = std::chrono::high_resolution_clock::now();
    auto elapsedTime2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - beginTime2);
    double programTimes2 = ((double)elapsedTime2.count());
    LOG("reduceMeanand scale 1.0 elapsed %lf", programTimes2);
    dequantize(&out2);
    outtensor(out2);
    cout << "加速比:" << (programTimes - programTimes2) / programTimes;
}

int main(int argc, const char** argv)
{
    DISABLE_GRAD;
    //testconcatenate();
    //testreducemeanandscale();
    //testSum();
    NMTMain(argc - 1, argv + 1);
    //testMatrix();
    //testquantizeincol();
    //_CrtDumpMemoryLeaks();
    //testl1layer();
    //testMultiply();
    //testMultiplydim();
    //testdiv();
    //testsumdim();
    //testBatchMatrix();
    //testscaleandshift();
    //testsplitandmerge();
    //testscale();
    return 0;
}

