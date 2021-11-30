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
 * $Created by: XIAO Tong (xiaotong@mail.neu.edu.cn) 2018-07-31
 * $Modified by: HU Chi (huchinlp@gmail.com) 2020-04
 */

#include "Embedding.h"
#include "LayerNorm.h"
#include "../Utility.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include <chrono>
#include "../quantize/quantize.h"

namespace nmt
{

/* constructor */
LN::LN()
{
    devID = -1;
    d = 0;
}

/* de-constructor */
LN::~LN()
{
}

/*
initialize the model
>> argc - number of arguments
>> argv - list of pointers to the arguments
>> config - configurations of the model
*/
void LN::InitModel(Config& config)
{
    devID = config.devID;

    d = config.modelSize;

    InitTensor1D(&w, d, X_FLOAT, devID);
    InitTensor1D(&b, d, X_FLOAT, devID);
    w.SetDataRand(1.0F, 1.0F);
    b.SetZeroAll();

    w.SetDataFixed(1);
}

/*
make the network
>> input - the input tensor
>> return - layer normalization output
*/
XTensor LN::Make(XTensor& input)
{
    XTensor& x = input;
    XTensor xn;
    XTensor mean;
    XTensor variance;
    XTensor standard;
    XTensor meanFilled;
    XTensor standardFilled;

    TENSOR_DATA_TYPE dataType = input.dataType;

    if (dataType == X_FLOAT16) {
        /* reduce functions can only run with FP32 */
        x = ConvertDataType(input, X_FLOAT);
    }

    /* \mu = (sum_i x_i)/m */
    mean = ReduceMean(x, x.order - 1);

    /* \sigma = (sum_i (x_i - \mu)^2)/m */
    variance = ReduceVariance(x, x.order - 1, mean);

    /* standard = sqrt(variance) */
    standard = Power(variance, 0.5F);

    /* unsqueeze mean and standard deviation to fit them into
       the same shape of x */
    meanFilled = Unsqueeze(mean, x.order - 1, x.GetDim(-1));
    standardFilled = Unsqueeze(standard, x.order - 1, x.GetDim(-1));

    /* x' = (x - \mu)/standard */
    xn = (x - meanFilled) / standardFilled;

    if (dataType != mean.dataType) {
        x = ConvertDataType(x, dataType);
        xn = ConvertDataType(xn, dataType);
    }

    /* result = x' * w + b   */
    return xn * w + b;
}
XTensor LN::Make_int8(XTensor& input)
{
    XTensor& x = input;
    XTensor xn;
    XTensor mean;
    XTensor l1;
    XTensor standard;

    TENSOR_DATA_TYPE dataType = input.dataType;

    if (dataType == X_FLOAT16) {
        /* reduce functions can only run with FP32 */
        x = ConvertDataType(input, X_FLOAT);
    }
    //auto beginTime = std::chrono::high_resolution_clock::now();
    /* \mu = (sum_i x_i)/m */
    mean = ReduceMean_int8(x, x.order - 1,NULL,1.0);
    //auto endTime = std::chrono::high_resolution_clock::now();
    //auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(endTime - beginTime);
    //double programTimes = ((double)elapsedTime.count());
    //LOG("reduceMeanand scale 1.0 elapsed %lf", programTimes);
    /* \sigma = (sum_i abs(x_i - \mu))/m */
    //auto beginTime1 = std::chrono::high_resolution_clock::now();
    
    
    //l1 = ReduceMeanandScale_int8(x, x.order - 1, &mean,3.0);
    
    float dr = (!x.isSparse) ? 1.0F : x.denseRatio;
    XTensor meanFilled(x.order, x.dimSize, x.dataType, dr, x.devID, x.mem);
    meanFilled.enableGrad = x.enableGrad;
    meanFilled.SetTMPFlag();
    XTensor l1Filled(x.order, x.dimSize, x.dataType, dr, x.devID, x.mem);
    l1Filled.enableGrad = x.enableGrad;
    l1Filled.SetTMPFlag();

    int stride = x.dimSize[x.order - 1];
    int num = 1;
    int_8* data = (int_8*)meanFilled.data;
    int_8* meandata = (int_8*)mean.data;
    int_8* xdata = (int_8*)x.data;
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
            absdata[i * stride + j] =round(sum);
        }
    }
    l1Filled.SetDataInt8(absdata, l1Filled.unitNum);
    meanFilled.initScale(x.scale.order, x.scale.dimsize, x.scale.values);
    l1Filled.initScale(x.scale.order, x.scale.dimsize, x.scale.values);
    delete[] absdata;
    //auto endTime1 = std::chrono::high_resolution_clock::now();
    //auto elapsedTime1 = std::chrono::duration_cast<std::chrono::microseconds>(endTime1 - beginTime1);
    //double programTimes1 = ((double)elapsedTime1.count());
    //LOG("reduceMeanand scale 3.0 elapsed %lf", programTimes1);
    float* s = l1Filled.scale.values;
    float pi = 3.14159265358979323846;

    //auto beginTime2 = std::chrono::high_resolution_clock::now();
    /* l1*c    (c=sqrt(pi))  */
    for (int i = 0; i < l1Filled.scale.unitnum; i++) {
        s[i] = s[i] / sqrt(pi);
    }
    
    /*float* sx = x.scale.values;
    float* smean = mean.scale.values;
    int row = x.scale.dimsize[0];
    int col = x.scale.dimsize[1];
    for (int i = 0; i < row; i++) {
        for (int j = 0; j < col; j++)
        {
            sx[i * col + j] = (sx[i * col + j] - smean[i]) / 2;
        }
    }*/
    //auto endTime2 = std::chrono::high_resolution_clock::now();
    //auto elapsedTime2 = std::chrono::duration_cast<std::chrono::microseconds>(endTime2 - beginTime2);
    //double programTimes2 = ((double)elapsedTime2.count());
    //LOG(" xi-mean %lf", programTimes2);
    /* unsqueeze mean and standard deviation to fit them into
       the same shape of x */
    //auto beginTime3 = std::chrono::high_resolution_clock::now();
    //meanFilled = Unsqueeze(mean, x.order - 1, x.GetDim(-1));
    //auto endTime3 = std::chrono::high_resolution_clock::now();
    //auto elapsedTime3 = std::chrono::duration_cast<std::chrono::microseconds>(endTime3 - beginTime3);
    //double programTimes3 = ((double)elapsedTime3.count());
    //LOG(" usqueezemean %lf", programTimes3);

    //auto beginTime4 = std::chrono::high_resolution_clock::now();
    //l1Filled = Unsqueeze(l1, x.order - 1, x.GetDim(-1));
    //auto endTime4 = std::chrono::high_resolution_clock::now();
    //auto elapsedTime4 = std::chrono::duration_cast<std::chrono::microseconds>(endTime4 - beginTime4);
    //double programTimes4 = ((double)elapsedTime4.count());
    //LOG(" usqueezemean %lf", programTimes4);

    /* x' = (x - \mu)/standard */
    xn = meanFilled / l1Filled;

    if (dataType != mean.dataType) {
        x = ConvertDataType(x, dataType);
        xn = ConvertDataType(xn, dataType);
    }
    if (w.dataType != X_INT8) {
        nmt::quantize(&w);
    }
    if (b.dataType != X_INT8) {
        nmt::quantize(&b);
    }
    /* result = x' * w + b   */
    return xn * w + b;
}

}