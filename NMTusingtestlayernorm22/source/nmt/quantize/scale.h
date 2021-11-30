#ifndef __SCALE_H__
#define __SCALE_H__
//#include "../../niutensor/tensor/XList.h"
//#include "../../niutensor/tensor/XGlobal.h"
//#include "../Model.h"
#include <stdio.h>
using namespace std;
#define MAX_SCALE_DIM 5
//using namespace nts;
namespace nmt {
	class Scale {
	public:
		float *  values = NULL;
		int order;
		int unitnum;
		int dimsize[MAX_SCALE_DIM] = {0};
		bool allocated=false;
		/* constructer */
		Scale();
		/* de-constructer */
		~Scale();
		void setData(float *d,int unitnum);
		void initScale(int order,const int * dimsize, float *d);
		void initScale(int order, const int* dimsize);
	};
}
#endif