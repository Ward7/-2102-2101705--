#include "scale.h"
#include <string.h>
namespace nmt {

Scale::Scale() {
	/*switch (input.order)
	{
	case 1:{
		InitTensor0D(&value, X_FLOAT, input.devID, false);
		break;
	}
	case 2: {
		InitTensor1D(&value, input.dimSize[0], X_FLOAT, input.devID, false);
		break;
	}
	case 3: {
		InitTensor2D(&value, input.dimSize[0], input.dimSize[1], X_FLOAT, input.devID, false);
		break;
	}
	default:
		break;
	}*/
}
Scale::~Scale() {

}
void Scale::setData(float *d , int unitnum) {
	memcpy(values, d, unitnum*sizeof(float));
}
void Scale::initScale(int order, const int* dimsize, float* d) {

	if (allocated == false) {
		unitnum = 1;
		this->order = order;
		for (int i = 0; i < order; i++) {
			this->dimsize[i] = dimsize[i];
			unitnum *= dimsize[i];
		}
		values = new float[unitnum];
		setData(d, unitnum);
		this->allocated = true;
	}
	else {
		setData(d, unitnum);
	}
	

}
void Scale::initScale(int order, const int* dimsize) {
	unitnum = 1;
	this->order = order;
	for (int i = 0; i < order; i++) {
		this->dimsize[i] = dimsize[i];
		unitnum *= dimsize[i];
	}
	values = new float[unitnum];
	this->allocated = true;
}

}