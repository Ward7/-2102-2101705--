#ifndef __QUANTIZE_H__
#define __QUANTIZE_H__
#include "../../niutensor/tensor/XList.h"
#include "../../niutensor/tensor/XTensor.h"
#include "../../niutensor/tensor/XGlobal.h"
#include "../Model.h"
using namespace std;
using namespace nts;
namespace nmt {
	
		/* quantize tensor from fp32 to int8
		input:the tensor we want to quantize
		*/
		void quantize(XTensor* input);
		void quantize(XTensor* input, float* d);
		/* dequantize tensor from int 8 to fp32 
		input:the tensor we want to dequantize
		*/
		void dequantize(XTensor* input);
		void requantize(XTensor* input, float* data);
		void dequantizeMatrix2DinRow(XTensor* input);
		float* dequantizeMatrix2DinRowtoData(XTensor* input);
		void dequantizeMatrix2DinCol(XTensor* input);
		float* dequantizeMatrix2DinColtoData(XTensor* input);
		void dequantizeMatrix2D(XTensor* input);
		void quantizeMatrix2DinCol(XTensor* input);
		void quantizeMatrix2DinCol(XTensor* input, float* d);
		float* quantizeMatrix2DinColtoScale(XTensor* input, float* d);
		float* dequantizeMatrix2D(int unitNum, float* d, float* s);
		void CopyDataAndScale(XTensor* s, XTensor* t);
		void dequantizeMatrix2D(XTensor* input, float* d);
		void dequantizeMatrix2D(XTensor* input, float* d, float* s);
		void matchscale(XTensor* a, XTensor* b);
		float* dequantizeevery(float* data, float* scale, int num);
}
#endif
