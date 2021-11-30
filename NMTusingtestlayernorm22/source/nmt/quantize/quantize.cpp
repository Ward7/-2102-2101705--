#include "quantize.h"
namespace nmt {



    void quantize(XTensor* input) {
        int unitNum = input->unitNum;
        int* converted = new int[unitNum];
        int dimsize[5] = { 0 };
        int stride = 1;
        for (int i = 0; i < input->order - 1; i++) {
            stride *= input->dimSize[i];
            dimsize[i] = input->dimSize[i];
        }
        int blocksize = input->dimSize[input->order - 1];
        if (input->dataType == DEFAULT_DTYPE)
        {
            float* scales = new float[stride];
            float scale = 0;
            float* d = (float*)input->data;
            float maxd = 0;
            for (int i = 0; i < stride; i++) {
                for (int j = 0; j < blocksize; j++) {
                    if (maxd <= fabs(d[i * blocksize + j])) {
                        maxd = fabs(d[i * blocksize + j]);
                    }
                }
                if (maxd == 0) { scale = 1; }
                else {
                    scale = 127.0 / maxd;
                }
                for (int j = 0; j < blocksize; j++) {
                    converted[i * blocksize + j] = round(d[i * blocksize + j] * scale);
                }
                scales[i] = scale;
            }
            input->initScale(input->order - 1, dimsize, scales);
            *input = ConvertDataType(input, X_INT8);
            input->SetDataInt8(converted, unitNum);
            delete[] converted;
            delete[] scales;

        }
        else if (input->dataType == X_INT) {
            float* scales = new float[stride];
            float scale = 0;
            int* d = (int*)input->data;
            float maxd = 0;
            for (int i = 0; i < stride; i++) {
                for (int j = 0; j < blocksize; j++) {
                    if (maxd <= fabs(d[i * blocksize + j])) {
                        maxd = fabs(d[i * blocksize + j]);
                    }
                }
                if (maxd == 0) { scale = 1; }
                else {
                    scale = 127.0 / maxd;
                }
                for (int j = 0; j < blocksize; j++) {
                    converted[i * blocksize + j] =round(d[i * blocksize + j] * scale);
                }
                scales[i] = scale;
            }
            input->initScale(input->order - 1, dimsize, scales);
            *input = ConvertDataType(input, X_INT8);
            input->SetDataInt8(converted, unitNum);
            delete[] converted;
            delete[] scales;
            delete[] dimsize;
        }

    }

    void quantize(XTensor* input, float* d) {
        int unitNum = input->unitNum;
        int* converted = new int[unitNum];
        int stride = 1;
        int* dim=new int[input->order - 1];
        for (int i = 0; i < input->order - 1; i++) {
            stride *= input->dimSize[i];
            dim[i] =input-> dimSize[i];
        }
        int blocksize = input->dimSize[input->order - 1];
        float* scales = new float[stride];
        float scale = 0;
        float maxd = 0;
        for (int i = 0; i < stride; i++) {
            for (int j = 0; j < blocksize; j++) {
                if (maxd <= fabs(d[i * blocksize + j])) {
                    maxd = fabs(d[i * blocksize + j]);
                }
            }
            if (maxd == 0) { scale = 1; }
            else {
                scale = 127.0 / maxd;
            }
            for (int j = 0; j < blocksize; j++) {
                converted[i * blocksize + j] =round(d[i * blocksize + j] * scale);
            }
            scales[i] = scale;
        }
        input->initScale(input->order-1,dim,scales);
        //input->scale.setData(scales, unitNum / input->dimSize[input->order - 1]);
        input->SetDataInt8(converted, unitNum);
        delete[] converted;
        delete[] scales;
        delete[] dim;



    }

    void requantize(XTensor* input, DTYPE* d) {
        int unitNum = input->unitNum;
        int* converted = new int[unitNum];
        int stride = 1;
        for (int i = 0; i < input->order - 1; i++) {
            stride *= input->dimSize[i];
        }
        int blocksize = input->dimSize[input->order - 1];

        float* scales = input->scale.values;
        float scale = 0;
        float maxd = 0;
        for (int i = 0; i < stride; i++) {
            for (int j = 0; j < blocksize; j++) {
                if (maxd <= fabs(d[i * blocksize + j])) {
                    maxd = fabs(d[i * blocksize + j]);
                }
            }
            if (maxd == 0) { scale = 1; }
            else {
                scale = 127.0 / maxd;
            }
            for (int j = 0; j < blocksize; j++) {
                converted[i * blocksize + j] = round(d[i * blocksize + j] * scale);
            }
            scales[i] *= scale;
        }
        input->SetDataInt8(converted, unitNum);
        delete[] converted;
    }
    void dequantize(XTensor* input) {
        float* real = new float[input->unitNum];
        int_8* now = (int_8*)input->data;
        int stride = 1;
        float* scales = input->scale.values;
        for (int i = 0; i < input->order - 1; i++) {
            stride *= input->dimSize[i];
        }
        int blocksize = input->dimSize[input->order - 1];
        for (int i = 0; i < stride; i++) {
            for (int j = 0; j < blocksize; j++) {
                real[i * blocksize + j] = now[i * blocksize + j] / scales[i];
            }
        }
        *input = ConvertDataType(input, X_FLOAT);
        input->SetData(real, input->unitNum);
        delete[]real;
    }
    void dequantizeMatrix2DinRow(XTensor* input) {
        float* converted = new float[input->unitNum];
        int num = 0;
        int dimsize0 = input->dimSize[0];
        int dimsize1 = input->dimSize[1];
        int_8* d = (int_8*)input->data;
        float* s = input->scale.values;
        for (int i = 0; i < dimsize0; i++) {
            for (int j = 0; j < dimsize1; j++) {
                converted[num++] = d[i * dimsize1 + j] / s[i];
            }
        }
        *input = ConvertDataType(input, X_FLOAT);
        input->SetData(converted, input->unitNum);
        input->scale.allocated = false;
        delete[] converted;
    }
    float* dequantizeMatrix2DinRowtoData(XTensor* input) {
        float* converted = new float[input->unitNum];
        int num = 0;
        int dimsize0 = input->dimSize[0];
        int dimsize1 = input->dimSize[1];
        int_8* d = (int_8*)input->data;
        float* s = input->scale.values;
        for (int i = 0; i < dimsize0; i++) {
            for (int j = 0; j < dimsize1; j++) {
                converted[num++] = d[i * dimsize1 + j] / s[i];
            }
        }
        input->scale.allocated = false;
        return converted;
    }
    void dequantizeMatrix2DinCol(XTensor* input) {
        float* converted=new float[input->unitNum];
        int num = 0;
        int dimsize0 = input->dimSize[0];
        int dimsize1 = input->dimSize[1];
        int_8* d = (int_8*)input->data;
        float* s = input->scale.values;
        for (int i = 0; i < dimsize1; i++) {
            for (int j = 0; j < dimsize0; j++) {
                converted[j * dimsize1 + i] = d[j * dimsize1 + i] / s[i];
            }
        }
        *input = ConvertDataType(input, X_FLOAT);
        input->SetData(converted, input->unitNum);
        input->scale.allocated = false;
        delete[] converted;
    }
    float* dequantizeMatrix2DinColtoData(XTensor* input) {
        float* converted;
        int num = 0;
        int dimsize0 = input->dimSize[0];
        int dimsize1 = input->dimSize[1];
        int_8* d = (int_8*)input->data;
        float* s = input->scale.values;
        for (int i = 0; i < dimsize0; i++) {
            for (int j = 0; j < dimsize1; j++) {
                converted[num++] = d[j * dimsize1 + i] / s[i];
            }
        }
        input->scale.allocated = false;
        return converted;
    }
    void dequantizeMatrix2D(XTensor* input) {

        if (input->dataType == X_FLOAT) {
            float* converted = new float[input->unitNum];
            float* d = (float*)input->data;
            float* s = input->scale.values;
            for (int i = 0; i < input->unitNum; i++) {
                converted[i] = d[i] / s[i];
            }
            *input = ConvertDataType(input, X_FLOAT);
            input->SetData(converted, input->unitNum);
            delete[] converted;
        }
        else if (input->dataType == X_INT8) {
            float* converted = new float[input->unitNum];
            int_8* d = (int_8*)input->data;
            float* s = input->scale.values;
            for (int i = 0; i < input->unitNum; i++) {
                converted[i] = d[i] / s[i];
            }
            *input = ConvertDataType(input, X_FLOAT);
            input->SetData(converted, input->unitNum);
            delete[] converted;
        }
        else {
            ShowNTErrors("todo");
        }

    }



    void dequantizeMatrix2D(XTensor* input, float* d) {

        float* converted = new float[input->unitNum];
        float* s = input->scale.values;
        for (int i = 0; i < input->unitNum; i++) {
            converted[i] = d[i] / s[i];
        }
        /*ConvertDataType(*input, *input, X_FLOAT);*/
        *input = ConvertDataType(input, X_FLOAT);
        input->SetData(converted, input->unitNum);
        delete[]converted;
    }
    void dequantizeMatrix2D(XTensor* input, float* d, float* s) {

        DTYPE* converted = new DTYPE[input->unitNum];
        for (int i = 0; i < input->unitNum; i++) {
            converted[i] = d[i] / s[i];
        }
        /*ConvertDataType(*input, *input, X_FLOAT);*/
        *input = ConvertDataType(input, X_FLOAT);
        input->SetData(converted, input->unitNum);
        delete[]converted;
    }
    float* dequantizeMatrix2D(int unitNum, float* d, float* s) {

        DTYPE* converted = new DTYPE[unitNum];
        for (int i = 0; i < unitNum; i++) {
            converted[i] = d[i] / s[i];
        }
        /*ConvertDataType(*input, *input, X_FLOAT);*/
        return converted;
    }
    void quantizeMatrix2DinCol(XTensor* input) {
        
        int* converted = new int[input->unitNum];
        int scalenum = 0;
        int row = input->dimSize[0];
        int col = input->dimSize[1];
        float* scales = new float[col];
        int dimSize[1];
        dimSize[0] = col;
        for (int i = 0; i < col; i++) {
            int num = input->dimSize[1];
            DTYPE* d = (DTYPE*)input->data;
            float maxd = fabs(d[0]);
            for (int j = 0; j < row; j++) {
                if (maxd <= fabs(d[i + j * col])) {
                    maxd = fabs(d[i + j * col]);
                }
            }
            float scale;
            if (maxd == 0) {
                scale = 1;
            }
            else {
                scale = 127.0 / maxd;
            }
            for (int j = 0; j < row; j++) {
                converted[i + j * col] =round( d[i + j * col] * scale);
            }

            scales[i] = scale;

        }

        input->initScale(1, dimSize, scales);
        *input = ConvertDataType(input, X_INT8);
        input->SetDataInt8(converted, input->unitNum);
        delete[] converted;
        delete[] scales;
    }
    void quantizeMatrix2DinCol(XTensor* input, float* d) {

        int* converted = new int[input->unitNum];
        int scalenum = 0;
        int row = input->dimSize[0];
        int col = input->dimSize[1];
        int dimSize[1];
        dimSize[0] = col;
        float* scales = new float[col];
        for (int i = 0; i < col; i++) {
            int num = input->dimSize[1];
            float maxd = fabs(d[0]);
            for (int j = 0; j < row; j++) {
                if (maxd <= fabs(d[i + j * col])) {
                    maxd = fabs(d[i + j * col]);
                }
            }
            float scale;
            if (maxd == 0) {
                scale = 1;
            }
            else {
                scale = 127.0 / maxd;
            }
            for (int j = 0; j < row; j++) {
                converted[i + j * col] =round( d[i + j * col] * scale);
            }

            scales[i] = scale;

        }
        input->SetDataInt8(converted, input->unitNum);
        delete[]input->scale.values;
        input->scale.allocated = false;
        input->initScale(1, dimSize, scales);
        delete[] converted;
    }
    float* quantizeMatrix2DinColtoScale(XTensor* input, float* d) {

        int scalenum = 0;
        int row = input->dimSize[0];
        int col = input->dimSize[1];
        float* scales = new float[col];
        int dimSize[1];
        dimSize[0] = col;
        for (int i = 0; i < col; i++) {
            float maxd = fabs(d[0]);
            for (int j = 0; j < row; j++) {
                if (maxd <= fabs(d[i + j * col])) {
                    maxd = fabs(d[i + j * col]);
                }
            }
            float scale;
            if (maxd == 0) {
                scale = 1;
            }
            else {
                scale = 127.0 / maxd;
            }

            scales[i] = scale;

        }
        return scales;
    }

    void CopyDataAndScale(XTensor* s, XTensor* t) {
        CheckNTErrors(s->dataType == t->dataType, "datatype not matched!")
            memcpy(t->scale.values, s->scale.values, s->scale.unitnum);
        memcpy((char*)t->data, (char*)s->data, s->unitNum * s->unitSize);
    }
//void matchscale(XTensor* a, XTensor* b) {
//        float* sa = a->scale.values;
//        float* sb = b->scale.values;
//        float scale;
//        float* scalea = new float[a->scale.unitnum];
//        float* scaleb = new float[b->scale.unitnum];
//        for (int i = 0; i < a->scale.unitnum; i++) {
//            scale = sa[i] < sb[i] ? sa[i] : sb[i];
//            scalea[i] = sa[i] / scale;
//            scaleb[i] = sb[i] / scale;
//            sa[i] = scale;
//            sb[i] = scale;
//            
//        }
//        int* converteda = new int[a->unitNum];
//        int* convertedb = new int[b->unitNum];
//        int dimsize[5] = { 0 };
//        int stridea = a->scale.unitnum;
//        int blocksizea = a->dimSize[a->order - 1];
//        int strideb = b->scale.unitnum;
//        int blocksizeb = b->dimSize[b->order - 1];
//            int_8* da = (int_8*)a->data;
//            int_8* db = (int_8*)b->data;
//            float maxd = 0;
//            for (int i = 0; i < stridea; i++) {
//                for (int j = 0; j < blocksizea; j++) {
//                    converteda[i * blocksizea + j] = da[i * blocksizea + j] / scalea[i];
//                }
//            }
//            for (int i = 0; i < strideb; i++) {
//                for (int j = 0; j < blocksizeb; j++) {
//                    convertedb[i * blocksizeb + j] = db[i * blocksizeb + j] / scaleb[i];
//                }
//            }
//            a->SetDataInt8(converteda, a->unitNum);
//            b->SetDataInt8(convertedb, b->unitNum);
//            delete[] converteda;
//            delete[] convertedb;
//            delete[] scalea;
//            delete[] scaleb;
//    }
float* dequantizeevery(float* data, float* scale, int num) {
    float* real = new float[num];
    for (int i = 0; i < num; i++) {
        real[i] = data[i] / scale[i];
    }
    return real;
}
void matchscale(XTensor* a, XTensor* b) {
    if (a->order == b->order) {
        float* sa = a->scale.values;
        float* sb = b->scale.values;
        float scale;
        float* scalea = new float[a->scale.unitnum];
        float* scaleb = new float[b->scale.unitnum];
        for (int i = 0; i < a->scale.unitnum; i++) {
            scale = sa[i] < sb[i] ? sa[i] : sb[i];
            scalea[i] = sa[i] / scale;
            scaleb[i] = sb[i] / scale;
            sa[i] = scale;
            sb[i] = scale;

        }
        int* converteda = new int[a->unitNum];
        int* convertedb = new int[b->unitNum];
        int dimsize[5] = { 0 };
        int stridea = a->scale.unitnum;
        int blocksizea = a->dimSize[a->order - 1];
        int strideb = b->scale.unitnum;
        int blocksizeb = b->dimSize[b->order - 1];
        int_8* da = (int_8*)a->data;
        int_8* db = (int_8*)b->data;
        float maxd = 0;
        for (int i = 0; i < stridea; i++) {
            for (int j = 0; j < blocksizea; j++) {
                converteda[i * blocksizea + j] = da[i * blocksizea + j] / scalea[i];
            }
        }
        for (int i = 0; i < strideb; i++) {
            for (int j = 0; j < blocksizeb; j++) {
                convertedb[i * blocksizeb + j] = db[i * blocksizeb + j] / scaleb[i];
            }
        }
        a->SetDataInt8(converteda, a->unitNum);
        b->SetDataInt8(convertedb, b->unitNum);
        delete[] converteda;
        delete[] convertedb;
        delete[] scalea;
        delete[] scaleb;
    }
    else {
        CheckNTErrors(a->dimSize[a->order - 1] == b->dimSize[0], "scale not match");
        CheckNTErrors(b->dimSize[0], "scale not match");
        float* sa = a->scale.values;
        float* sb = b->scale.values;
        float scale = 10000;
        float* scalea = new float[a->scale.unitnum];
        float* scaleb = new float[b->scale.unitnum];
        for (int i = 0; i < a->scale.unitnum; i++) {
            if (sa[i] < scale) {
                scale = sa[i];
            }
        }
        if (sb[0] < scale) {
            scale = sb[0];
            for (int i = 0; i < a->scale.unitnum; i++) {
                scalea[i] = sa[i] / scale;
                sa[i] = scale;
            }
            int* converteda = new int[a->unitNum];
            int dimsize[5] = { 0 };
            int stridea = a->scale.unitnum;
            int blocksizea = a->dimSize[a->order - 1];
            int_8* da = (int_8*)a->data;
            for (int i = 0; i < stridea; i++) {
                for (int j = 0; j < blocksizea; j++) {
                    converteda[i * blocksizea + j] = round(da[i * blocksizea + j] / scalea[i]);
                }
            }
            a->SetDataInt8(converteda, a->unitNum);
            delete[] converteda;
            delete[] scalea;
            delete[] scaleb;
        }
        else {
            for (int i = 0; i < a->scale.unitnum; i++) {
                scalea[i] = sa[i] / scale;
                sa[i] = scale;
            }
            scaleb[0] = sb[0] / scale;
            sb[0] = scale;
            int* converteda = new int[a->unitNum];
            int* convertedb = new int[b->unitNum];
            int stridea = a->scale.unitnum;
            int blocksizea = a->dimSize[a->order - 1];
            int strideb = b->scale.unitnum;
            int blocksizeb = b->dimSize[b->order - 1];
            int_8* da = (int_8*)a->data;
            int_8* db = (int_8*)b->data;
            float maxd = 0;
            for (int i = 0; i < stridea; i++) {
                for (int j = 0; j < blocksizea; j++) {
                    converteda[i * blocksizea + j] = round(da[i * blocksizea + j] / scalea[i]);
                }
            }
            for (int i = 0; i < b->unitNum; i++) {
                convertedb[i] = round(db[i] / scaleb[0]);
            }
            a->SetDataInt8(converteda, a->unitNum);
            b->SetDataInt8(convertedb, b->unitNum);
            delete[] converteda;
            delete[] convertedb;
            delete[] scalea;
            delete[] scaleb;
        }

    }
}
}