#ifndef KEYPOINT_H
#define KEYPOINT_H

#include <opencv2/calib3d.hpp>
#include <vector>
#include <math.h>

class Keypoint;
typedef std::vector<Keypoint> KVector;
typedef std::vector<std::pair<Keypoint, Keypoint> > KPairVector;



class Keypoint
{
	public:
		Keypoint(){
			this->orientacion=0;
		}
		~Keypoint(){}

	float operator-(const Keypoint &aux)
	{
		float sum=0.f;
		for(int i=0; i < 64; ++i)
		  sum += (this->descriptor[i] - aux.descriptor[i])*(this->descriptor[i] - aux.descriptor[i]);
		return sqrt(sum);
	};

	float x, y;

	float scale;

	float orientacion;

	int laplacian;

	float descriptor[64];

	float dx, dy;

	int clusterIndex;
};


#endif