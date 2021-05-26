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
	
	void trainAsync(std::string modelPath);
	
	void match(std::string scenePath);
	
	bool isTraining() const;
	
	glm::mat4 getPose(size_t index) const;
	
	size_t getNumPoses() const;
	
	ofEvent<void> trainingEndEvent;
	
	static ofMesh transformMeshAndSave(const ofMesh& mesh, glm::mat4 matrix, string savePath);
	
	
	bool beginApplyingPose(ofCamera&cam, size_t poseIndex =0, ofRectangle viewport = ofRectangle());
	void endApplyingPose();
	
	
private:
	std::atomic<bool> _bIsTraining;
	
	int64_t tick1, tick2;
	
	unique_ptr<cv::ppf_match_3d::PPF3DDetector> _detector = nullptr;
	cv::Mat _modelMat;
	
	vector<glm::mat4> _poses;
	
	void _update(ofEventArgs&);

	void _train(std::string modelPath);

	class ThreadHelper : public ofThread{
	public:
		ThreadHelper(ofxSurfaceMatching & sm, string _modelPath):SM(sm), modelPath(_modelPath){
			startThread();
		}
		~ThreadHelper(){
			if(isThreadRunning()) waitForThread(true);
		}
		virtual void threadedFunction() override;
		ofxSurfaceMatching& SM;
		string modelPath;
	};
	
	bool removeThreadHelper();
	
	shared_ptr<ThreadHelper> threadHelper = nullptr;
	
	bool _bPoseWasApplied = false;
	
};
