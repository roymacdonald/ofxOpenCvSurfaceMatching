//
//  ofxSurfaceMatching
//
//  Created by Roy Macdonald on 5/15/21.
//
//
#pragma once

#include "ofMain.h"
#include "opencv2/surface_matching.hpp"
#include "opencv2/surface_matching/ppf_helpers.hpp"
#include "opencv2/core/utility.hpp"

class ofxSurfaceMatching{
public:
	ofxSurfaceMatching();
	
	void train(std::string modelPath);
	
	void match(std::string scenePath);
	
	bool isTraining() const;
	
private:
	std::atomic<bool> _bIsTraining;
	int64_t tick1, tick2;
	
	unique_ptr<cv::ppf_match_3d::PPF3DDetector> _detector = nullptr;
	cv::Mat _modelMat;
};
