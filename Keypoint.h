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


void getEnc(KVector &kp1, KVector &kp2, KPairVector &matches)
{
  float dist, d1, d2;
  Keypoint *match;

  matches.clear();

  for(int i = 0; i < kp1.size(); i++) 
  {
    d1 = d2 = FLT_MAX; //max

    for(int j = 0; j < kp2.size(); j++) 
    {
      dist = kp1[i] - kp2[j];  

      if(dist<d1)
      {
        d2 = d1;
        d1 = dist;
        match = &kp2[j];
      }
      else if(dist<d2)
      {
        d2 = dist;
      }
    }
    if(d1/d2 < 0.65) 
    { 
      kp1[i].dx = match->x - kp1[i].x; 
      kp1[i].dy = match->y - kp1[i].y;
      matches.push_back(std::make_pair(kp1[i], *match));
    }
  }
}


#endif