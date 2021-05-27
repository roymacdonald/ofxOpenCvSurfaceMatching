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
	
	
	///\brief training parameters
    ///\param relativeSamplingStep Sampling distance relative to the object's diameter. 
    ///                            Models are first sampled uniformly in order to improve efficiency. 
    ///                            Decreasing this value leads to a denser model, and a more accurate pose estimation 
    ///                            but the larger the model, the slower the training. Increasing the value leads to a
    ///                            less accurate pose computation but a smaller model and faster model generation and 
    ///                            matching. Beware of the memory consumption when using small values.
    ///\param relativeDistanceStep The discretization distance of the point pair distance relative to the model's 
    ///                            diameter. This value has a direct impact on the hashtable. Using small values would
    ///                            lead to too fine discretization, and thus ambiguity in the bins of hashtable. Too
    ///                            large values would lead to no discrimination over the feature vectors and different
    ///                            point pair features would be assigned to the same bin. This argument defaults to 
    ///                            the value of RelativeSamplingStep. For noisy scenes, the value can be increased to
    ///                            improve the robustness of the matching against noisy points.
    ///\param numAngles            Set the discretization of the point pair orientation as the number of subdivisions 
    ///                            of the angle. This value is the equivalent of RelativeDistanceStep for the 
    ///                            orientations. Increasing the value increases the precision of the matching but 
    ///                            decreases the robustness against incorrect normal directions. Decreasing the value 
    ///                            decreases the precision of the matching but increases the robustness against 
    ///                            incorrect normal directions. For very noisy scenes where the normal directions can
    ///                            not be computed accurately, the value can be set to 25 or 20.
    
	
	///\brief train
	///\param modelPath            File path to a PLY file that represents the model with which to train the detector.
	///\param movel                ofMesh object tha represents the model with which to train the detector.
	///\param relativeSamplingStep  read above in "training parameters"
	///\param relativeDistanceStep 	read above in "training parameters"
	///\param numAngles 			read above in "training parameters"

	void train(std::string modelPath, double relativeSamplingStep = 0.025, const double relativeDistanceStep=0.05, const double numAngles=30);
	void train(const ofMesh& model, double relativeSamplingStep = 0.025, const double relativeDistanceStep=0.05, const double numAngles=30);


	///\brief Asynchronous version of the train function. Runs on a different thread.
	///\ parameters are the same as the regular train function
	///\ when using one of these make sure you register a listener to this object's trainingEndEvent
	void trainAsync(std::string modelPath, double relativeSamplingStep = 0.025, const double relativeDistanceStep=0.05, const double numAngles=30);
	void trainAsync(const ofMesh& model, double relativeSamplingStep = 0.025, const double relativeDistanceStep=0.05, const double numAngles=30);
	
	
	
	 ///\brief Matches a trained model across a provided scene.
	 ///\param scene Point cloud for the scene  as an ofMesh object.
	 ///\param scenePath file path to a PLY file that represents the point cloud
	 ///
	 ///\param relativeSceneSampleStep The ratio of scene points to be used for the matching after sampling with relativeSceneDistance. 
	 ///                               For example, if this value is set to 1.0/5.0, every 5th point from the scene is used for pose estimation.
	 ///                               This parameter allows an easy trade-off between speed and accuracy of the matching.
	 ///                               Increasing the value leads to less points being used and in turn to a faster but less accurate pose computation.
	 ///                               Decreasing the value has the inverse effect.
	 ///
	 ///\param relativeSceneDistance Set the distance threshold relative to the diameter of the model. 
	 ///                             This parameter is equivalent to relativeSamplingStep in the training stage. 
	 ///                             This parameter acts like a prior sampling with the relativeSceneSampleStep parameter.
	  
	
	void match(std::string scenePath, const double relativeSceneSampleStep=1.0/5.0, const double relativeSceneDistance=0.03);
	void match(const ofMesh& scene, const double relativeSceneSampleStep=1.0/5.0, const double relativeSceneDistance=0.03);
	
	
	 ///\brief Set parameters for ICP matching.
	 ///\param iterations
	 ///\param tolerence Controls the accuracy of registration at each iteration of ICP.
	 ///\param rejectionScale Robust outlier rejection is applied for robustness. This value
     ///	    actually corresponds to the standard deviation coefficient. Points with
	 ///		rejectionScale * &sigma are ignored during registration.
	 ///\param numLevels Number of pyramid levels to proceed. Deep pyramids increase speed but
	 ///		decrease accuracy. Too coarse pyramids might have computational overhead on top of the
	 ///		inaccurate registrtaion. This parameter should be chosen to optimize a balance.
	 ///		Typical values range from 4 to 10.
	 ///\param sampleType Currently this parameter is ignored and only uniform sampling is applied. Leave it as 0.
	 ///\param numMaxCorr Currently this parameter is ignored and only PickyICP is applied. Leave it as 1.
	 
	void setIcpParams(const int iterations, const float tolerence = 0.05f, const float rejectionScale = 2.5f, const int numLevels = 6);
	
	
	
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
	
	
	///\brief ofEvent triggered when the training ends. This works on the threaded or non-threaded train functions
	ofEvent<void> trainingEndEvent;
	

	///\brief use this function to apply the pose transformation to whatever you draw between calling this function and endApplyPose();
	/// remember to call endApplyingPose(); after this function.
	bool beginApplyingPose(ofCamera&cam, size_t poseIndex =0, ofRectangle viewport = ofRectangle());
	void endApplyingPose();
	

	static ofMesh transformMeshAndSave(const ofMesh& mesh, glm::mat4 matrix, string savePath);
	
	
private:
	
	void _setTrainingParams( double relativeSamplingStep, const double relativeDistanceStep, const double numAngles);
	
	std::atomic<bool> _bIsTraining;
	
	int64_t tick1, tick2;
	
	unique_ptr<cv::ppf_match_3d::PPF3DDetector> _detector = nullptr;
	
	cv::Mat _modelMat;
	
	vector<Pose> _poses;
	
	void _update(ofEventArgs&);

	void _train(std::string modelPath);
	void _train(const ofMesh & mesh);
	void _train();
	
	std::atomic<double> _relativeSamplingStep;
	std::atomic<double> _relativeDistanceStep;
	std::atomic<double> _numAngles;
		
	int _icpIterations = 100;
	float _icpTolerence = 0.005f;
	float _icpRejectionScale = 2.5f;
	int _icpNumLevels = 8;
	
	
	
	void _match(const cv::Mat& pcTest, const double relativeSceneSampleStep, const double relativeSceneDistance);
	
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
