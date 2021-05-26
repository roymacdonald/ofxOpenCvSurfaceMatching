
//
//  ofxSurfaceMatching
//
//  Created by Roy Macdonald on 5/15/21.
//
//

#include "ofxSurfaceMatching.h"

using namespace cv;
using namespace ppf_match_3d;
#include <future>

#define TICK_1 tick1=cv::getTickCount();
#define TICK_2 tick2=cv::getTickCount();
#define ELAPSED_TIME (double)(tick2-tick1)/ cv::getTickFrequency()

void fromCV2GLM(const cv::Matx44d& cvmat, glm::mat4& glmmat) {
    if (cvmat.cols != 4 || cvmat.rows != 4 ) {
        cout << "Matrix conversion error!" << endl;
        return;
    }
	for (int i = 0; i < 16; i++) {
		glm::value_ptr( glmmat )[i] = (float)cvmat.val[i];
	}
	glmmat = glm::transpose(glmmat);
	
	
//    memcpy(glm::value_ptr(*glmmat), cvmat.data, 16 * sizeof(float));
}

ofMesh ofxSurfaceMatching::transformMeshAndSave(const ofMesh& mesh, glm::mat4 matrix, string savePath){
		
	glm::mat3 R(matrix);
	glm::vec3 t(matrix[3]);
	
	cout << "transformMeshAndSave : \nR: " << R << "\nT: " << t << "\nMatrix: " << matrix << endl;
	
	ofMesh tempMesh;
	
	for (size_t i = 0; i < mesh.getVertices().size(); i++)
	   {
		const auto& v = mesh.getVertices()[i];
		   
		 glm::vec4 p = matrix * glm::vec4(v, 1);
	
		 // p2[3] should normally be 1
		   if(!ofIsFloatEqual(p.w, 1.0f)){
			   cout << "P.w: " << p.w <<endl;
		   }
		 if (fabs(p.w) > std::numeric_limits<float>::epsilon()){
			 tempMesh.addVertex((1.0 / p.w) * glm::vec3(p));
		 }

		 // If the point cloud has normals,
		 // then rotate them as well
		 if (mesh.hasNormals() && mesh.getNormals().size() > i)
		 {
			 glm::vec3 n = mesh.getNormals()[i];
			 glm::vec3 n2 = R * n;

		   double nNorm = glm::l2Norm(n2);

		   if (nNorm > std::numeric_limits<float>::epsilon() )
		   {
			   tempMesh.addNormal( (1.0 / nNorm) * n2);
		   }
		 }
	   }
	cout << "centroid: " << tempMesh.getCentroid() << endl;
	if(!savePath.empty()){
		tempMesh.save(savePath);
	}
	return tempMesh;
}


ofxSurfaceMatching::ofxSurfaceMatching(){
	_bIsTraining = false;
	
}

void ofxSurfaceMatching::train(std::string modelPath){
	_train(modelPath);
	ofNotifyEvent(trainingEndEvent, this);
}

void ofxSurfaceMatching::_train(std::string modelPath){
	_bIsTraining = true;
	
	

		_modelMat = loadPLYSimple(ofToDataPath(modelPath, true).c_str(), 1);
		
//		cout <<  "channels: " << _modelMat.channels() << endl;
//		cout <<  "rows: " << _modelMat.rows  << endl;
//		cout <<  "cols: " << _modelMat.cols  << endl;
//		cout <<  "depth: " << _modelMat.depth() << endl;
//		cout <<  "size: " << _modelMat.size  << endl;
//		cout <<  "dims: " << _modelMat.dims  << endl;
//
		// Now train the model
		ofLogVerbose("ofxSurfaceMatching::train") << "Training...";
//		TICK_1
	
		_detector = make_unique<ppf_match_3d::PPF3DDetector> (0.025, 0.05);
		_detector->trainModel(_modelMat);
	
//		TICK_2
	
	ofLogVerbose("ofxSurfaceMatching::train")  << "Training complete in ";
//	<< ELAPSED_TIME << " sec";
	
//	FileStorage fs(ofToDataPath("saved_training", true), FileStorage::WRITE );
	
//	_detector->write(fs);
	
	
	_bIsTraining = false;

}

void ofxSurfaceMatching::trainAsync(std::string modelPath){
	if(!_bIsTraining && threadHelper == nullptr){
		threadHelper = make_shared<ThreadHelper>(*this, modelPath);
	
//		cout << "trainAsync\n";
	}else{
		ofLogWarning("ofxSurfaceMatching::trainAsync") << "can not train when there is another training still happening";
	}
	
}
void ofxSurfaceMatching::match(std::string scenePath){
	
	if(!_bIsTraining && _detector != nullptr){
		
		string sceneFileName = ofToDataPath(scenePath, true);

		cout << "sceneFilePath: " << sceneFileName <<endl;
		
			// Read the scene
			Mat pcTest = loadPLYSimple(sceneFileName.c_str(), 1);

			// Match the model to the scene and get the pose
			ofLogVerbose("ofxSurfaceMatching::match") << "Starting matching...";
		
			vector<Pose3DPtr> results;
			TICK_1
			_detector->match(pcTest, results, 1.0/40.0, 0.05);
			TICK_2
		ofLogVerbose("ofxSurfaceMatching::match")  << "PPF Elapsed Time : "<< ELAPSED_TIME << " sec" ;

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
			
		ofLogVerbose("ofxSurfaceMatching::match") << "ICP Elapsed Time : " << ELAPSED_TIME << " sec";
				 
		_poses.clear();
		_poses.resize(resultsSub.size());
		
			cout << "Poses: " << endl;
			
			for (size_t i=0; i<resultsSub.size(); i++)
			{
				Pose3DPtr result = resultsSub[i];
				cout << "Pose Result " << i << endl;
				result->printPose();
				fromCV2GLM(result->pose, _poses[i]);
			}
	}
}

bool ofxSurfaceMatching::beginApplyingPose(ofCamera&cam, size_t poseIndex, ofRectangle viewport){
	_bPoseWasApplied = false;
	if(!isTraining() && getNumPoses() > 0){
		ofPushMatrix();
		ofLoadIdentityMatrix();
		
		ofSetMatrixMode(OF_MATRIX_PROJECTION);
		ofLoadMatrix(cam.getProjectionMatrix(viewport.isEmpty()?ofRectangle(0, 0, ofGetWidth(), ofGetHeight()):viewport));
		ofSetMatrixMode(OF_MATRIX_MODELVIEW);
		ofLoadViewMatrix(cam.getModelViewMatrix()* getPose(poseIndex));
		_bPoseWasApplied =true;
		
	}
	return _bPoseWasApplied;
}

void ofxSurfaceMatching::endApplyingPose(){
	if(_bPoseWasApplied){
		ofPopMatrix();
	}
}

size_t ofxSurfaceMatching::getNumPoses() const{
	return _poses.size();
}

glm::mat4 ofxSurfaceMatching::getPose(size_t index) const{
	if(index >= _poses.size()){
		ofLogWarning("ofxSurfaceMatching::getPose") << "index is out of bounds. index: " << index << " num poses: " << _poses.size();
		return  glm::mat4( 1.0 );
	}
	return _poses[index];
}

bool ofxSurfaceMatching::isTraining() const{
	return _bIsTraining;
}

void ofxSurfaceMatching::_update(ofEventArgs&){
	if(removeThreadHelper()){
		ofNotifyEvent(trainingEndEvent, this);
	}
}

//--------------------------------------------------------------
void ofxSurfaceMatching::ThreadHelper::threadedFunction(){
	if(isThreadRunning()){
		ofAddListener(ofEvents().update, &SM, &ofxSurfaceMatching::_update);
		SM._train(modelPath);
	}
}
//--------------------------------------------------------------
bool ofxSurfaceMatching::removeThreadHelper(){
	if(threadHelper && !threadHelper->isThreadRunning()){
		ofRemoveListener(ofEvents().update, this, &ofxSurfaceMatching::_update);
		threadHelper.reset();
		threadHelper = nullptr;
		return true;
	}
	return false;
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
