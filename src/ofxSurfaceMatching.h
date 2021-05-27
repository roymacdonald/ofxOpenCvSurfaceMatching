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
	void train(const ofMesh& model);

	void trainAsync(std::string modelPath);
	void trainAsync(const ofMesh& model);
	
	void match(std::string scenePath);
	void match(const ofMesh& scene);
	
	
	bool isTraining() const;
	
	struct Pose{
		double alpha = 0;
		double residual = 0;
		size_t modelIndex = 0;
		size_t numVotes = 0;
		double angle = 0;

		glm::mat4 matrix;
	};
	
	
	glm::mat4 getPoseMatrix(size_t index) const;
	
	Pose getPose(size_t index) const;
	
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
	
	vector<Pose> _poses;
	
	void _update(ofEventArgs&);

	void _train(std::string modelPath);
	void _train(const ofMesh & mesh);
	void _train();
	
	void _match(const cv::Mat& pcTest);
	
	class ThreadHelper : public ofThread{
	public:
		ThreadHelper(ofxSurfaceMatching & sm, string _modelPath):SM(sm), modelPath(_modelPath){
			startThread();
		}
		ThreadHelper(ofxSurfaceMatching & sm, const ofMesh& _mesh):SM(sm), modelPath(""), mesh(_mesh){
			startThread();
		}
		
		
		~ThreadHelper(){
			if(isThreadRunning()) waitForThread(true);
		}
		virtual void threadedFunction() override;
		ofxSurfaceMatching& SM;
		string modelPath;
		ofMesh mesh;
	};
	
	bool removeThreadHelper();
	
	shared_ptr<ThreadHelper> threadHelper = nullptr;
	
	bool _bPoseWasApplied = false;
	
};
