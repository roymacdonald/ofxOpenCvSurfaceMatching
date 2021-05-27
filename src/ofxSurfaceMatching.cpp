
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

void ofMeshToCvMat(const ofMesh & mesh, cv::Mat & mat){
	ofLogVerbose("ofMeshToCvMat start");
	
	bool bHasNormals = (mesh.hasNormals() && mesh.getNumVertices() == mesh.getNumNormals() );
	mat.create(mesh.getNumVertices(), bHasNormals? 6 : 3, CV_32FC1);
	glm::vec3 v, n;
	for (size_t i = 0; i < mesh.getNumVertices(); i++)
	{
	   float* data = mat.ptr<float>(i);
		 v = mesh.getVertex(i);
		 data[0] = v.x;
		 data[1] = v.y;
		 data[2] = v.z;
		if(bHasNormals){
			n = glm::normalize(mesh.getNormal(i));
			data[3] = n.x;
			data[4] = n.y;
			data[5] = n.z;
		}
	}
	
	if(!bHasNormals){
		cv::Mat PCNormals;
		cv::Vec3d viewpoint(0, 0, 0);
		cv::ppf_match_3d::computeNormalsPC3d(mat, PCNormals, 6, false, viewpoint);
		mat =  PCNormals;
	}
	ofLogVerbose("ofMeshToCvMat end");
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

void ofxSurfaceMatching::train(std::string modelPath, double relativeSamplingStep, const double relativeDistanceStep, const double numAngles){
	_setTrainingParams( relativeSamplingStep, relativeDistanceStep, numAngles);
	_train(modelPath);
	ofNotifyEvent(trainingEndEvent, this);
}
void ofxSurfaceMatching::train(const ofMesh& model, double relativeSamplingStep, const double relativeDistanceStep, const double numAngles){
	_setTrainingParams( relativeSamplingStep, relativeDistanceStep, numAngles);
	_train(model);
	ofNotifyEvent(trainingEndEvent, this);
}

void ofxSurfaceMatching::_train(std::string modelPath){
	_bIsTraining = true;
	
	_modelMat = loadPLYSimple(ofToDataPath(modelPath, true).c_str(), 1);
	_train();

}

void ofxSurfaceMatching::_train(const ofMesh & mesh){
	_bIsTraining = true;
	ofMeshToCvMat(mesh, _modelMat);
	_train();
}

void ofxSurfaceMatching::_train(){
	_bIsTraining = true;
		
	ofLogVerbose("ofxSurfaceMatching::train") << "Training...";
	TICK_1
	
	_detector = make_unique<ppf_match_3d::PPF3DDetector> (_relativeSamplingStep , _relativeDistanceStep, _numAngles);
	_detector->trainModel(_modelMat);
	
	TICK_2
	
	ofLogVerbose("ofxSurfaceMatching::train")  << "Training complete in "	<< ELAPSED_TIME << " sec";
	
	_bIsTraining = false;

}

void ofxSurfaceMatching::trainAsync(std::string modelPath, double relativeSamplingStep , const double relativeDistanceStep, const double numAngles){
	if(!_bIsTraining && threadHelper == nullptr){
		_setTrainingParams( relativeSamplingStep, relativeDistanceStep, numAngles);
		threadHelper = make_shared<ThreadHelper>(*this, modelPath);
	}else{
		ofLogWarning("ofxSurfaceMatching::trainAsync") << "can not train when there is another training still happening";
	}
}

void ofxSurfaceMatching::trainAsync(const ofMesh& model, double relativeSamplingStep , const double relativeDistanceStep, const double numAngles){
	if(!_bIsTraining && threadHelper == nullptr){
			_setTrainingParams( relativeSamplingStep, relativeDistanceStep, numAngles);
			threadHelper = make_shared<ThreadHelper>(*this, model);
		}else{
			ofLogWarning("ofxSurfaceMatching::trainAsync") << "can not train when there is another training still happening";
		}
}

void ofxSurfaceMatching::_setTrainingParams( double relativeSamplingStep, const double relativeDistanceStep, const double numAngles){
	_relativeSamplingStep = relativeSamplingStep;
	_relativeDistanceStep = relativeDistanceStep;
	_numAngles = numAngles;
}


void ofxSurfaceMatching::_match(const cv::Mat& pcTest, const double relativeSceneSampleStep, const double relativeSceneDistance){
	if(!_bIsTraining && _detector != nullptr){
	// Match the model to the scene and get the pose
		ofLogVerbose("ofxSurfaceMatching::match") << "Starting matching...";
	
		vector<Pose3DPtr> results;
		TICK_1
		_detector->match(pcTest, results, relativeSceneSampleStep, relativeSceneDistance);
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
	
		ICP icp(_icpIterations,
				_icpTolerence,
				_icpRejectionScale,
				_icpNumLevels);
		
	
	
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
			
			fromCV2GLM(result->pose, _poses[i].matrix);
			_poses[i].alpha = result->alpha;
			_poses[i].residual = result->residual;
			_poses[i].modelIndex = result->modelIndex;
			_poses[i].numVotes = result->numVotes;
			_poses[i].angle = result->angle;
			
			
			
		}
	}
}
void ofxSurfaceMatching::match(std::string scenePath, const double relativeSceneSampleStep, const double relativeSceneDistance){
	
	if(!_bIsTraining && _detector != nullptr){
		// Read the scene
		Mat pcTest = loadPLYSimple(ofToDataPath(scenePath, true).c_str(), 1);
		_match(pcTest, relativeSceneSampleStep, relativeSceneDistance);
	}
}
void ofxSurfaceMatching::match(const ofMesh& scene, const double relativeSceneSampleStep, const double relativeSceneDistance){
	
	if(!_bIsTraining && _detector != nullptr){
		cv::Mat pcTest;
		ofMeshToCvMat(scene, pcTest);
		_match(pcTest, relativeSceneSampleStep, relativeSceneDistance);
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
		ofLoadViewMatrix(cam.getModelViewMatrix()* getPoseMatrix(poseIndex));
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

glm::mat4 ofxSurfaceMatching::getPoseMatrix(size_t index) const{
	if(index >= _poses.size()){
		ofLogWarning("ofxSurfaceMatching::getPoseMatrix") << "index is out of bounds. index: " << index << " num poses: " << _poses.size();
		return  glm::mat4( 1.0 );
	}
	return _poses[index].matrix;
}
ofxSurfaceMatching::Pose ofxSurfaceMatching::getPose(size_t index) const{
	if(index >= _poses.size()){
		ofLogWarning("ofxSurfaceMatching::getPose") << "index is out of bounds. index: " << index << " num poses: " << _poses.size();
		return  ofxSurfaceMatching::Pose();
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

void ofxSurfaceMatching::setIcpParams(const int iterations, const float tolerence, const float rejectionScale, const int numLevels){
	_icpIterations = iterations;
	_icpTolerence = tolerence;
	_icpRejectionScale = rejectionScale;
	_icpNumLevels = numLevels;
}

//--------------------------------------------------------------
void ofxSurfaceMatching::ThreadHelper::threadedFunction(){
	if(isThreadRunning()){
		ofAddListener(ofEvents().update, &SM, &ofxSurfaceMatching::_update);
		if(modelPath.empty()){
			SM._train(mesh);
		}else{
			SM._train(modelPath);
		}
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


