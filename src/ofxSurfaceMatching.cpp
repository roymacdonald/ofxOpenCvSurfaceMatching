
//
//  ofxSurfaceMatching
//
//  Created by Roy Macdonald on 5/15/21.
//
//

#include "ofxSurfaceMatching.h"

using namespace cv;
using namespace ppf_match_3d;

#define TICK_1 tick1=cv::getTickCount();
#define TICK_2 tick2=cv::getTickCount();
#define ELAPSED_TIME (double)(tick2-tick1)/ cv::getTickFrequency()

ofxSurfaceMatching::ofxSurfaceMatching(){
	_bIsTraining = false;
}

void ofxSurfaceMatching::train(std::string modelPath){
	_bIsTraining = true;
	
	

		_modelMat = loadPLYSimple(ofToDataPath(modelPath, true).c_str(), 1);
		
		cout <<  "channels: " << _modelMat.channels() << endl;
		cout <<  "rows: " << _modelMat.rows  << endl;
		cout <<  "cols: " << _modelMat.cols  << endl;
		cout <<  "depth: " << _modelMat.depth() << endl;
		cout <<  "size: " << _modelMat.size  << endl;
		cout <<  "dims: " << _modelMat.dims  << endl;
	
		// Now train the model
		ofLogVerbose("ofxSurfaceMatching::train") << "Training...";
		TICK_1
	
		_detector = make_unique<ppf_match_3d::PPF3DDetector> (0.025, 0.05);
		_detector->trainModel(_modelMat);
	
		TICK_2
	
		ofLogVerbose("ofxSurfaceMatching::train")  << "Training complete in " << ELAPSED_TIME << " sec";
	_bIsTraining = false;
}

void ofxSurfaceMatching::match(std::string scenePath){
	if(!_bIsTraining && _detector != nullptr){
		
		string sceneFileName = ofToDataPath(scenePath, true);

			// Read the scene
			Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);
			cout <<  "channels: " << pcTest.channels() << endl;
			cout <<  "rows: " << pcTest.rows  << endl;
			cout <<  "cols: " << pcTest.cols  << endl;
			cout <<  "depth: " << pcTest.depth() << endl;
			cout <<  "size: " << pcTest.size  << endl;
			cout <<  "dims: " << pcTest.dims  << endl;
			
			// Match the model to the scene and get the pose
			ofLogVerbose("ofxSurfaceMatching::match") << "Starting matching...";
		
			vector<Pose3DPtr> results;
			TICK_1
			_detector->match(pcTest, results, 1.0/40.0, 0.05);
			TICK_2
		ofLogVerbose("ofxSurfaceMatching::match")  << "PPF Elapsed Time " << ELAPSED_TIME << " sec" ;

			//check results size from match call above
			size_t results_size = results.size();
			ofLogVerbose("ofxSurfaceMatching::match")  << "Number of matching poses: " << results_size;
			if (results_size == 0) {
				ofLogVerbose("ofxSurfaceMatching::match") << "No matching poses found. Exiting.";
				return;
			}

			// Get only first N results - but adjust to results size if num of results are less than that specified by N
			size_t N = 2;
			if (results_size < N) {
				ofLogVerbose("ofxSurfaceMatching::match") << "Reducing matching poses to be reported (as specified in code): "
				<< N << " to the number of matches found: " << results_size ;
				N = results_size;
			}
			vector<Pose3DPtr> resultsSub(results.begin(),results.begin()+N);
			
			// Create an instance of ICP
			ICP icp(100, 0.005f, 2.5f, 8);
		
		
			TICK_1
			
			// Register for all selected poses
		ofLogVerbose("ofxSurfaceMatching::match") << "Performing ICP on " << N << " poses...";
			icp.registerModelToScene(_modelMat, pcTest, resultsSub);
			TICK_2
			
			ofLogVerbose("ofxSurfaceMatching::match") << "ICP Elapsed Time " << ELAPSED_TIME << " sec";
				 
			cout << "Poses: " << endl;
			// debug first five poses
			for (size_t i=0; i<resultsSub.size(); i++)
			{
				Pose3DPtr result = resultsSub[i];
				cout << "Pose Result " << i << endl;
				result->printPose();
//				if (i==0)
//				{
//					Mat pct = transformPCPose(_modelMat, result->pose);
//					writePLY(pct, "para6700PCTrans.ply");
//				}
			}
			
	}
}

bool ofxSurfaceMatching::isTraining() const{
	return _bIsTraining;
}

////compite normals
//string modelFileName = (string)argv[1];
//  string outputFileName = (string)argv[2];
//  cv::Mat points, pointsAndNormals;
//
//  cout << "Loading points\n";
//  cv::ppf_match_3d::loadPLYSimple(modelFileName.c_str(), 1).copyTo(points);
//
//  cout << "Computing normals\n";
//  cv::Vec3d viewpoint(0, 0, 0);
//  cv::ppf_match_3d::computeNormalsPC3d(points, pointsAndNormals, 6, false, viewpoint);
//
//  std::cout << "Writing points\n";
//  cv::ppf_match_3d::writePLY(pointsAndNormals, outputFileName.c_str());
//  //the following function can also be used for debugging purposes
//  //cv::ppf_match_3d::writePLYVisibleNormals(pointsAndNormals, outputFileName.c_str());
//
//  std::cout << "Done\n";
//  return 0;
//}
//

