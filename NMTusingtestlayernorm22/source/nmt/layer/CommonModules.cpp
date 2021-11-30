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
 * $Created by: Bei Li (libei_neu@outlook.com) 2020-02-05
 * This file includes some common modules of the Transformer model
 */

#include "CommonModules.h"
#include "../../niutensor/tensor/core/CHeader.h"
#include "../../niutensor/tensor/function/FHeader.h"
#include "../quantize/quantize.h"
namespace nmt
{

/* 
flexible layer normalization for the Transformer 
>> input - input tensor
>> ln - the layernorm network
>> prenorm - whether we use prenorm or not
>> before - whether we use layernorm before attention/fnn
>> after - whether we use layernorm after attention/fnn
*/
XTensor LayerNorm(XTensor& input, LN& ln, bool prenorm, bool before, bool after)
{
    if (after ^ prenorm) {
        if (input.dataType == X_INT8) {
            return ln.Make_int8(input);
        }
        //return ln.Make(input);
        else {
            quantize(&input);
            XTensor output;
            output = ln.Make_int8(input);
            //output = ln.Make(input);
            dequantize(&output);
            dequantize(&ln.w);
            dequantize(&ln.b);
            dequantize(&input);
            return output;
        }
    }
    else
        return input;
}

}